import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import random
from tqdm import tqdm

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

# ======= SELEZIONE RASHOMON =========
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