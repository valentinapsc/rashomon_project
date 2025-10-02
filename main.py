import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
from tqdm import tqdm
import os
import glob

from captum.attr import Saliency, IntegratedGradients, Lime

from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

import matplotlib.pyplot as plt
from statistics import mean, pstdev

# PARAMETRI
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

NUM_MODELS = 10          # Numero modelli iniziali da addestrare
RASHOMON_THRESH = 0.01   # Soglia (1%) per selezione Rashomon set
EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 3             # Early stopping patience

MIN_DELTA = 1e-4         # miglioramento minimo richiesto sulla val loss

IG_BASELINE_MODE = "dataset_mean"

SAVE_DIR = "rashomon_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# DATASET 
train_raw = datasets.MNIST('.', train=True, download=True,
                           transform=transforms.ToTensor())

def compute_dataset_stats(ds, batch_size=1024):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    s, ss, n = 0.0, 0.0, 0
    for x, _ in loader:                     # x: [B,1,28,28] in [0,1]
        b = x.size(0)
        x = x.view(b, -1)
        s  += x.sum().item()
        ss += (x * x).sum().item()
        n  += x.numel()
    mean = s / n
    var  = ss / n - mean**2
    std  = var**0.5
    return float(mean), float(std)

mean, std = compute_dataset_stats(train_raw)
print(f"MNIST train stats → mean={mean:.6f}, std={std:.6f}")

# trasformazione con Normalize fissata (μ,σ) del TRAIN
normalize = transforms.Normalize(mean=[mean], std=[std])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# ricreo i dataset normalizzati (train/test) e poi faccio lo split train/val
mnist_train = datasets.MNIST('.', train=True,  download=True, transform=transform)
mnist_test  = datasets.MNIST('.', train=False, download=True, transform=transform)

# split: train 80%, val 20% del train originale (entrambi già normalizzati con gli stessi parametri)
train_size = int(0.8 * len(mnist_train))
val_size   = len(mnist_train) - train_size
train_dataset, val_dataset = random_split(mnist_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(mnist_test,    batch_size=BATCH_SIZE)

# MODELLO 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# TRAINING + EARLY STOPPING 
def train_one_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # tengono conto della migliore epoca su validation
    best_val_loss = None
    best_val_acc  = 0.0
    best_weights  = None
    
    patience_counter = 0

    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # VALIDATION
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                batch_loss = criterion(output, target).item()
                val_loss_sum += batch_loss * data.size(0)  # somma pesata
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total   += target.size(0)

        val_loss = val_loss_sum / max(total, 1)
        val_acc  = correct / max(total, 1)

        # EARLY STOPPING: migliora se la val_loss diminuisce di almeno MIN_DELTA
        if (best_val_loss is None) or (best_val_loss - val_loss > MIN_DELTA):
            # MIGLIORAMENTO SUFFICIENTE
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_weights  = model.state_dict()  # checkpoint
            patience_counter = 0
        else:
            # NESSUN miglioramento “abbastanza grande”
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break # ferma il training

    # ripristina il checkpoint migliore su VALIDATION
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, best_val_acc

# CARICAMENTO O TRAIN 
rashomon_models = []
all_val_acc = []

# prova a caricare modelli già salvati
model_files = sorted(glob.glob(os.path.join(SAVE_DIR, "rashomon_model_*.pt")))

if len(model_files) > 0:
    print("Caricamento modelli Rashomon dal disco...")
    for model_path in model_files:
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        rashomon_models.append(model)
    print(f"Modelli Rashomon caricati: {len(rashomon_models)}")
else:
    print("Training modelli (con early stopping) per Rashomon set...")
    temp_models = []
    temp_accs = []
    for i in tqdm(range(NUM_MODELS)):
        seed = SEED + i
        model, val_acc = train_one_model(seed)
        temp_models.append(model)
        temp_accs.append(val_acc)
        print(f"Modello {i}: Val accuracy={val_acc:.4f}")
    best_acc = max(temp_accs)
    rashomon_threshold = best_acc - RASHOMON_THRESH
    for i, (model, acc) in enumerate(zip(temp_models, temp_accs)):
        if acc >= rashomon_threshold:
            print(f"[✓] Modello {i} selezionato (val acc={acc:.4f})")
            rashomon_models.append(model)
            # salva il modello Rashomon su disco
            model_path = os.path.join(SAVE_DIR, f"rashomon_model_{i}.pt")
            torch.save(model.state_dict(), model_path)
        else:
            print(f"[ ] Modello {i} scartato (val acc={acc:.4f})")
    print(f"\nMigliore accuracy: {best_acc:.4f}")
    print(f"Soglia Rashomon: {rashomon_threshold:.4f}")
    print(f"Modelli Rashomon selezionati: {len(rashomon_models)}/{NUM_MODELS}")
    
# VALUTAZIONE RASHOMON SET
print("\n" + "="*50)
print("Rashomon set: validazione della 'equivalenza' (accuracies)")
print("="*50)

rashomon_metrics = []
for i, m in enumerate(rashomon_models):
    val_acc  = evaluate_accuracy(m, val_loader, device='cpu')
    test_acc = evaluate_accuracy(m, test_loader, device='cpu')
    rashomon_metrics.append({
        "model_id": i,
        "val_acc": val_acc,
        "test_acc": test_acc
    })

# stampa tabella ordinata per val_acc decrescente
rashomon_metrics.sort(key=lambda r: r["val_acc"], reverse=True)
print(f"{'Model':>5}  {'ValAcc':>8}  {'TestAcc':>8}")
for r in rashomon_metrics:
    print(f"{r['model_id']:>5}  {r['val_acc']*100:8.2f}%  {r['test_acc']*100:8.2f}%")

# riassunto (media ± std, min/max)
val_list  = [r["val_acc"] for r in rashomon_metrics]
test_list = [r["test_acc"] for r in rashomon_metrics]

def pct(x): return f"{x*100:.2f}%"
def pm(m,s): return f"{m*100:.2f}% ± {s*100:.2f}%"

print("\nRiepilogo:")
print(f"- Val acc (media±std):  {pm(mean(val_list),  pstdev(val_list) if len(val_list)>1 else 0.0)}")
print(f"- Test acc (media±std): {pm(mean(test_list), pstdev(test_list) if len(test_list)>1 else 0.0)}")
print(f"- Val acc min..max:     {pct(min(val_list))} .. {pct(max(val_list))}")
print(f"- Test acc min..max:    {pct(min(test_list))} .. {pct(max(test_list))}")

# GENERAZIONE SPIEGAZIONI
print("\n" + "="*50)
print("Generazione Spiegazioni")
print("="*50)

# parametri spiegazioni
EXPL_METHODS = ['saliency', 'ig', 'lime']

def generate_saliency(model, sample, label):
    explainer = Saliency(model)
    attr = explainer.attribute(sample, target=label)
    arr = attr.squeeze().detach().cpu().numpy()
    # normalizza [0,1] per confronto/visualizzazione
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def generate_ig(model, sample, label):
    explainer = IntegratedGradients(model)
    if IG_BASELINE_MODE == "black":
        base_val = (0.0 - mean) / (std + 1e-8)  # valore normalizzato del pixel "nero"
    else:
        base_val = 0.0                           # baseline = media dataset nello spazio normalizzato
    baselines = torch.full_like(sample, base_val)

    attr = explainer.attribute(sample, target=label, baselines=baselines)
    arr = attr.squeeze().detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

# esempio di “XAI model-agnostic”
def generate_lime(model, sample, label):
    explainer = Lime(model)
    baseline_val = torch.mean(sample) 

    attr = explainer.attribute(
        sample,
        target=label,
        n_samples=200,                    
        feature_mask=FEATURE_MASK.to(sample.device),
        perturbations_per_eval=50,         # batch su CPU
        baselines=baseline_val
    )
    arr = attr.squeeze().cpu().detach().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def generate_explanations(model, sample, label):
    explanations = {}
    if 'saliency' in EXPL_METHODS:
        explanations['saliency'] = generate_saliency(model, sample, label)
    if 'ig' in EXPL_METHODS:
        explanations['ig'] = generate_ig(model, sample, label)
    if 'lime' in EXPL_METHODS:
        explanations['lime'] = generate_lime(model, sample, label)    
    return explanations


SAMPLE_SIZE = 10
# immagini casuali dal test set
sample_indices = np.random.choice(len(mnist_test), SAMPLE_SIZE, replace=False)
sample_imgs = torch.stack([mnist_test[i][0] for i in sample_indices])
sample_labels = torch.tensor([mnist_test[i][1] for i in sample_indices])

device = torch.device("cpu")

# struttura per risultati
explanations_all = []

for model_idx, model in enumerate(rashomon_models):
    model.eval()
    model.to(device)
    model_results = []
    print(f"\nGenerazione spiegazioni per il modello {model_idx+1}/{len(rashomon_models)}")
    for img_idx in range(SAMPLE_SIZE):
        sample = sample_imgs[img_idx].unsqueeze(0).to(device)  # [1, 1, 28, 28]
        label = sample_labels[img_idx].item()
        exp = generate_explanations(model, sample, label)
        model_results.append({
            'img_index': int(sample_indices[img_idx]),
            'true_label': int(label),
            'explanations': exp
        })
        print(f"  Img {img_idx}: spiegazioni generate ({', '.join(exp.keys())})")
    explanations_all.append({
        'model_id': model_idx,
        'model_seed': SEED + model_idx,
        'explanations': model_results
    })
    
# MISURE DI SIMILARITÀ 
SIM_METRICS = ['SSIM', 'Pearson', 'Spearman', 'Cosine', 'MAE']

def calculate_similarity(exp1, exp2):
    # Normalizza le spiegazioni
    exp1_norm = (exp1 - np.min(exp1)) / (np.max(exp1) - np.min(exp1) + 1e-10)
    exp2_norm = (exp2 - np.min(exp2)) / (np.max(exp2) - np.min(exp2) + 1e-10)
    # Appiattisci
    flat1 = exp1_norm.flatten()
    flat2 = exp2_norm.flatten()
    # 1. SSIM
    try:
        ssim_val = ssim(exp1_norm, exp2_norm, data_range=1.0)
    except Exception:
        ssim_val = np.nan
    # 2. Pearson
    if np.std(flat1) == 0 or np.std(flat2) == 0:
        pearson_val = np.nan
    else:
        pearson_val, _ = pearsonr(flat1, flat2)
    # 3. Spearman
    if np.std(flat1) == 0 or np.std(flat2) == 0:
        spearman_val = np.nan
    else:
        spearman_val, _ = spearmanr(flat1, flat2)
    # 4. Cosine
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    if norm1 == 0 or norm2 == 0:
        cosine_val = np.nan
    else:
        cosine_val = np.dot(flat1, flat2) / (norm1 * norm2)
    # 5. MAE
    mae_val = np.mean(np.abs(exp1_norm - exp2_norm))
    return [ssim_val, pearson_val, spearman_val, cosine_val, mae_val]


all_explanations = {}
for model in explanations_all:
    all_explanations[model['model_id']] = [
        exp['explanations'] for exp in model['explanations']
    ]
    
print("\n" + "="*50)
print("Calcolo similarità tra spiegazioni")
print("="*50)

similarity_results = {m: {'same_model': {met: [] for met in SIM_METRICS},
                          'diff_model': {met: [] for met in SIM_METRICS}} for m in EXPL_METHODS}
for m1, m2 in combinations(EXPL_METHODS, 2):
    similarity_results[f"{m1}-{m2}"] = {'same_model': {met: [] for met in SIM_METRICS}}

# Per ogni immagine del campione
for img_idx in tqdm(range(SAMPLE_SIZE)):
    # intra-modello (confronta metodi diversi sullo stesso modello)
    for model_id, model_exps in all_explanations.items():
        methods = list(model_exps[img_idx].keys())
        for m1, m2 in combinations(methods, 2):
            sim_vals = calculate_similarity(
                model_exps[img_idx][m1], 
                model_exps[img_idx][m2]
            )
            for i, metric in enumerate(SIM_METRICS):
                similarity_results[f"{m1}-{m2}"]['same_model'][metric].append(sim_vals[i])

    # inter-modello (confronta lo stesso metodo tra modelli diversi)
    for method in EXPL_METHODS:
        model_exps = [all_explanations[model_id][img_idx][method] for model_id in all_explanations]
        for exp1, exp2 in combinations(model_exps, 2):
            sim_vals = calculate_similarity(exp1, exp2)
            for i, metric in enumerate(SIM_METRICS):
                similarity_results[method]['diff_model'][metric].append(sim_vals[i])
                
# Stampa risultati medi
# INTRA:
print("\nRisultati similarità INTRA-modello (tra metodi, stesso modello):")
pair_keys = [f"{m1}-{m2}" for m1, m2 in combinations(EXPL_METHODS, 2)]
for key in pair_keys:
    print(f"{key}:")
    for metric in SIM_METRICS:
        arr = np.array(similarity_results[key]['same_model'][metric], dtype=float)
        if arr.size > 0 and not np.isnan(np.nanmean(arr)):
            m = np.nanmean(arr)
            s = np.nanstd(arr)
            n = np.sum(~np.isnan(arr))
            print(f"  {metric}: {m:.3f} ± {s:.3f} (n={n})")
        else:
            print("  {0}: n/d".format(metric))

# INTER: 
print("\nRisultati similarità INTER-modello (stesso metodo, modelli diversi):")
for method in EXPL_METHODS:
    print(f"{method}:")
    for metric in SIM_METRICS:
        arr = np.array(similarity_results[method]['diff_model'][metric], dtype=float)
        if arr.size > 0 and not np.isnan(np.nanmean(arr)):
            m = np.nanmean(arr)
            s = np.nanstd(arr)
            n = np.sum(~np.isnan(arr))
            print(f"  {metric}: {m:.3f} ± {s:.3f} (n={n})")
        else:
            print("  {0}: n/d".format(metric))


# VALUTAZIONE QUALITÀ (MoRF)
def morf_curve_aopc(model, image, explanation, true_class, steps=10, device='cpu'):

    model.eval()
    img = image.clone().detach().to(device)
    baseline_val = torch.mean(img)  # baseline: media dei pixel

    flat_exp = explanation.flatten()
    idx_sorted = np.argsort(flat_exp)[::-1]  # feature più importanti prima

    probas = []
    for step in range(steps + 1):  # steps+1 per includere la baseline
        num_to_remove = int((step) / steps * len(flat_exp))
        masked = img.clone()
        if num_to_remove > 0:
            remove_indices = idx_sorted[:num_to_remove]
            for idx in remove_indices:
                h, w = np.unravel_index(idx, explanation.shape)
                masked[0, h, w] = baseline_val
        # Predizione
        with torch.no_grad():
            output = model(masked.unsqueeze(0).to(device))  # [1, 1, 28, 28]
            prob = torch.softmax(output, dim=1)[0, true_class].item()
        probas.append(prob)
    # Calcolo AOPC
    f0 = probas[0]
    diffs = [f0 - p for p in probas]
    aopc = np.mean(diffs)
    return aopc, probas

# calcolo MoRF per ogni modello, metodo, immagine campione

print("\n" + "="*50)
print("VALUTAZIONE QUALITÀ SPIEGAZIONI (MoRF)")
print("="*50)

STEPS_MORF = 10  # step della curva

quality_results = {m: [] for m in EXPL_METHODS}

device = torch.device("cpu") 

for model_idx, model in enumerate(rashomon_models):
    model.to(device)
    model.eval()
    for img_idx in range(SAMPLE_SIZE):
        sample_img = sample_imgs[img_idx].to(device)  # [1, 28, 28]
        true_class = sample_labels[img_idx].item()
        for method in EXPL_METHODS:
            # explanation: [28, 28], numpy
            explanation = explanations_all[model_idx]['explanations'][img_idx]['explanations'][method]
            aopc, probas = morf_curve_aopc(
                model, sample_img, explanation, true_class, steps=STEPS_MORF, device=device
            )
            quality_results[method].append(aopc)

# Output: media e std AOPC per ogni metodo
print("\nRisultati AOPC (più alto = spiegazione più efficace):")
for method in EXPL_METHODS:
    arr = np.array(quality_results[method], dtype=float)
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    n = np.sum(~np.isnan(arr))
    print(f"- {method}: {m:.4f} ± {s:.4f} (n={n})")


# VISUALIZZAZIONE

def denorm_img(t):
    # t shape [1,H,W] o [B,1,H,W]; riporta ai valori [0,1] prima della Normalize
    return (t * std + mean).clamp(0.0, 1.0)

def plot_explanations_grid(
    explanations_all,
    sample_imgs,
    sample_labels,
    methods=['saliency', 'ig', 'lime'],
    lime_heatmaps=None
):
    num_models = len(explanations_all)
    num_imgs = len(sample_imgs)
    num_methods = len(methods)
    
    total_cols = 1 + num_models * num_methods  # 1 originale + heatmap per ogni modello/metodo

    fig, axes = plt.subplots(num_imgs, total_cols, figsize=(3*total_cols, 3*num_imgs))

    if num_imgs == 1:
        axes = axes.reshape(1, -1)
    if total_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_imgs):
        # Colonna 0: originale
        axes[i, 0].imshow(denorm_img(sample_imgs[i])[0], cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Original\nLabel: {sample_labels[i].item()}")
        col = 1
        for m_idx, model_info in enumerate(explanations_all):
            for method in methods:
                if method == "lime" and lime_heatmaps is not None:
                    heatmap = lime_heatmaps[i]  # heatmap o immagine colorata da LIME
                else:
                    heatmap = model_info['explanations'][i]['explanations'][method]
                axes[i, col].imshow(heatmap, cmap='hot')
                axes[i, col].axis('off')
                axes[i, col].set_title(f"Model {model_info['model_id']+1}\n{method.capitalize()}")
                col += 1

    plt.tight_layout()
    plt.show()

plot_explanations_grid(explanations_all, sample_imgs, sample_labels, methods=['saliency', 'ig', 'lime'])

# CALCOLO E PLOT DELLE CURVE MORF MEDIE

avg_curves = {m: np.zeros(STEPS_MORF + 1) for m in EXPL_METHODS}
counts = {m: 0 for m in EXPL_METHODS}

for model_idx, model in enumerate(rashomon_models):
    model.eval()
    for img_idx in range(SAMPLE_SIZE):
        img = sample_imgs[img_idx]
        label = sample_labels[img_idx].item()
        for method in EXPL_METHODS:
            exp = explanations_all[model_idx]['explanations'][img_idx]['explanations'][method]
            _, probas = morf_curve_aopc(model, img, exp, label, steps=STEPS_MORF)
            avg_curves[method] += np.array(probas)
            counts[method] += 1

# media sulle immagini e modelli
for m in EXPL_METHODS:
    avg_curves[m] /= counts[m]

# plot
steps = np.linspace(0, 100, STEPS_MORF + 1)  # % feature rimosse
plt.figure(figsize=(6, 4))
for method in EXPL_METHODS:
    plt.plot(steps, avg_curves[method], marker='o', label=method.capitalize())

plt.xlabel("Percentuale di feature rimosse")
plt.ylabel("Probabilità classe corretta")
plt.title("Curve MoRF medie")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("morf_curves.png", dpi=300) 
plt.show()
