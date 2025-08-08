import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
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

SAVE_DIR = "rashomon_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# DATASET 
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST('.', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('.', train=False, download=True, transform=transform)

# Split: train 80%, val 20% del train originale
train_size = int(0.8 * len(mnist_train))
val_size = len(mnist_train) - train_size
train_dataset, val_dataset = random_split(mnist_train, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE)

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

    best_val_acc = 0.0
    patience_counter = 0
    best_weights = None

    for epoch in range(EPOCHS):
        # Training
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        val_acc = correct / total

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    # Carica i pesi migliori
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

# GENERAZIONE SPIEGAZIONI
print("\n" + "="*50)
print("Generazione Spiegazioni")
print("="*50)

# Parametri spiegazioni
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
    attr = explainer.attribute(sample, target=label, baselines=torch.zeros_like(sample))
    arr = attr.squeeze().detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

# esempio di “XAI model-agnostic”
def generate_lime(model, sample, label):
    explainer = Lime(model)
    attr = explainer.attribute(sample, target=label, n_samples=100)
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

# Esempio di utilizzo sul Rashomon set

SAMPLE_SIZE = 10
# scegli immagini casuali dal test set
sample_indices = np.random.choice(len(mnist_test), SAMPLE_SIZE, replace=False)
sample_imgs = torch.stack([mnist_test[i][0] for i in sample_indices])
sample_labels = torch.tensor([mnist_test[i][1] for i in sample_indices])

device = torch.device("cpu")

# Struttura per risultati
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
    # Intra-modello (confronta metodi diversi sullo stesso modello)
    for model_id, model_exps in all_explanations.items():
        methods = list(model_exps[img_idx].keys())
        for m1, m2 in combinations(methods, 2):
            sim_vals = calculate_similarity(
                model_exps[img_idx][m1], 
                model_exps[img_idx][m2]
            )
            for i, metric in enumerate(SIM_METRICS):
                similarity_results[f"{m1}-{m2}"]['same_model'][metric].append(sim_vals[i])

    # Inter-modello (confronta lo stesso metodo tra modelli diversi)
    for method in EXPL_METHODS:
        model_exps = [all_explanations[model_id][img_idx][method] for model_id in all_explanations]
        for exp1, exp2 in combinations(model_exps, 2):
            sim_vals = calculate_similarity(exp1, exp2)
            for i, metric in enumerate(SIM_METRICS):
                similarity_results[method]['diff_model'][metric].append(sim_vals[i])
                
# Stampa risultati medi
# ---- INTRA: tra metodi, stesso modello (SOLO COPPIE) ----
print("\nRisultati medi similarità INTRA-modello (tra metodi, stesso modello):")
pair_keys = [f"{m1}-{m2}" for m1, m2 in combinations(EXPL_METHODS, 2)]
for key in pair_keys:
    print(f"{key}:")
    for metric in SIM_METRICS:
        vals = similarity_results[key]['same_model'][metric]
        if len(vals) > 0 and not np.isnan(np.nanmean(vals)):
            print(f"  {metric}: {np.nanmean(vals):.3f} (n={len(vals)})")
        else:
            print(f"  {metric}: n/d")

# ---- INTER: stesso metodo, modelli diversi (SOLO METODI) ----
print("\nRisultati medi similarità INTER-modello (stesso metodo, modelli diversi):")
for method in EXPL_METHODS:
    print(f"{method}:")
    for metric in SIM_METRICS:
        vals = similarity_results[method]['diff_model'][metric]
        if len(vals) > 0 and not np.isnan(np.nanmean(vals)):
            print(f"  {metric}: {np.nanmean(vals):.3f} (n={len(vals)})")
        else:
            print(f"  {metric}: n/d")


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

# === Calcolo MoRF per ogni modello, metodo, immagine campione ===

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
    aopcs = quality_results[method]
    print(f"- {method}: mean={np.mean(aopcs):.4f}, std={np.std(aopcs):.4f} (n={len(aopcs)})")


# VISUALIZZAZIONE
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
        axes[i, 0].imshow(sample_imgs[i][0], cmap='gray')
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