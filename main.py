import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import random
from tqdm import tqdm

from captum.attr import Saliency, IntegratedGradients

from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr, spearmanr

# PARAMETRI
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

NUM_MODELS = 10          # Numero modelli iniziali da addestrare
RASHOMON_THRESH = 0.01   # Soglia (es: 1%) per selezione Rashomon set
EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 5             # Early stopping patience

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

# ADDDESTRAMENTO  
all_models = []
all_val_acc = []

print("Training modelli (con early stopping) per Rashomon set...")
for i in tqdm(range(NUM_MODELS)):
    seed = SEED + i  # seed diverso per ogni modello
    model, val_acc = train_one_model(seed)
    all_models.append(model)
    all_val_acc.append(val_acc)
    print(f"Modello {i}: Val accuracy={val_acc:.4f}")

# SELEZIONE RASHOMON 
best_acc = max(all_val_acc)
rashomon_threshold = best_acc - RASHOMON_THRESH

rashomon_models = []
for i, (model, acc) in enumerate(zip(all_models, all_val_acc)):
    if acc >= rashomon_threshold:
        print(f"[✓] Modello {i} selezionato (val acc={acc:.4f})")
        rashomon_models.append(model)
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
EXPL_METHODS = ['saliency', 'ig']

def generate_saliency(model, sample, label):
    explainer = Saliency(model)
    attr = explainer.attribute(sample, target=label)
    arr = attr.squeeze().detach().cpu().numpy()
    # Normalizza [0,1] per confronto/visualizzazione
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def generate_ig(model, sample, label):
    explainer = IntegratedGradients(model)
    attr = explainer.attribute(sample, target=label, baselines=torch.zeros_like(sample))
    arr = attr.squeeze().detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def generate_explanations(model, sample, label):
    explanations = {}
    if 'saliency' in EXPL_METHODS:
        explanations['saliency'] = generate_saliency(model, sample, label)
    if 'ig' in EXPL_METHODS:
        explanations['ig'] = generate_ig(model, sample, label)
    # Puoi aggiungere altri metodi qui...
    return explanations

# Esempio di utilizzo sul Rashomon set

SAMPLE_SIZE = 5
# Scegli immagini casuali dal test set
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