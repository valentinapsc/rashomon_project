"""
STUDIO SULLA STABILITÀ DELLE SPIEGAZIONI NEL RASHOMON SET

L'obiettivo è:
1. Costruire un Rashomon Set di modelli con prestazioni equivalenti
2. Generare spiegazioni con diversi algoritmi (Grad-CAM, LIME, SHAP)
3. Valutare la similarità tra spiegazioni con 5 metriche diverse
4. Quantificare la qualità delle spiegazioni con le curve MoRF
5. Analizzare la relazione tra similarità e qualità delle spiegazioni

"""

#1. IMPORTAZIONE LIBRERIE
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr, spearmanr
import lime
from lime import lime_image
import shap
from tqdm import tqdm
from itertools import combinations
import time

#2. PARAMETRI GLOBALI
# Configurazione sperimentale
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Parametri dataset
DATASET = "cifar10"
NUM_CLASSES = 10
IMG_SIZE = (32, 32, 3)

# Parametri modello
MODEL_CONFIG = [
    Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
]

# Parametri training
EPOCHS = 50
BATCH_SIZE = 64
RASHOMON_THRESHOLD = 0.01  # Soglia 1% per Rashomon Set
INIT_MODELS = 10           # Modelli iniziali da addestrare

# Parametri spiegazioni
EXPL_METHODS = ['gradcam', 'lime', 'shap']
SAMPLE_IMAGES = 5          # Immagini campione per analisi

# Parametri similarità
SIM_METRICS = ['SSIM', 'Pearson', 'Spearman', 'Cosine', 'MAE']

# 3. PREPARAZIONE DATI
print(f"\n{'='*50}")
print(f"{'PREPARAZIONE DATASET':^50}")
print(f"{'='*50}")

def load_and_preprocess_data():
    """Carica e preprocessa il dataset"""
    # Caricamento dataset
    if DATASET == "cifar10":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError(f"Dataset non supportato: {DATASET}")
    
    # Normalizzazione [0,1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Suddivisione validation set (10% del test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, 
        test_size=0.9, 
        random_state=SEED
    )
    
    print(f"- Dimensione dataset:")
    print(f"  Training:   {X_train.shape[0]} immagini")
    print(f"  Validation: {X_val.shape[0]} immagini")
    print(f"  Test:       {X_test.shape[0]} immagini")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Caricamento dati
X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

# Selezione immagini campione
sample_indices = np.random.choice(len(X_test), SAMPLE_IMAGES, replace=False)
X_sample = X_test[sample_indices]
y_sample = y_test[sample_indices]

# 4. DEFINIZIONE MODELLO
def build_model():
    """Costruisce il modello CNN"""
    model = Sequential(MODEL_CONFIG)
    model.compile(
        optimizer='adam', # Adaptive Moment Estimation
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Visualizzazione architettura
print(f"\n{'='*50}")
print(f"{'ARCHITETTURA MODELLO':^50}")
print(f"{'='*50}")
model = build_model()
model.summary()

# 5. COSTRUZIONE RASHOMON SET
print(f"\n{'='*50}")
print(f"{'COSTRUZIONE RASHOMON SET':^50}")
print(f"{'='*50}")

def train_rashomon_models():
    """Addestra modelli e seleziona Rashomon Set"""
    # Configurazione early stopping
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=0
    )
    
    all_models = []
    all_val_acc = []
    
    print(f"Addestramento di {INIT_MODELS} modelli con early stopping...")
    for i in tqdm(range(INIT_MODELS)):
        # Costruzione e addestramento modello
        model = build_model()
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Registrazione risultati
        best_val_acc = max(history.history['val_accuracy'])
        all_val_acc.append(best_val_acc)
        all_models.append(model)
    
    # Determinazione soglia Rashomon
    global_best_acc = max(all_val_acc)
    rashomon_threshold = global_best_acc - RASHOMON_THRESHOLD
    
    # Selezione modelli Rashomon
    rashomon_models = []
    for model, acc in zip(all_models, all_val_acc):
        if acc >= rashomon_threshold:
            model.id = f"Model_{len(rashomon_models)+1}"
            rashomon_models.append(model)
    
    print("\n- Risultati selezione Rashomon:")
    print(f"  Migliore accuracy: {global_best_acc:.4f}")
    print(f"  Soglia: {rashomon_threshold:.4f}")
    print(f"  Modelli selezionati: {len(rashomon_models)}/{INIT_MODELS}")
    print(f"  Range accuracy: {min(all_val_acc):.4f} - {global_best_acc:.4f}")
    
    return rashomon_models, global_best_acc

# Addestramento modelli
rashomon_models, best_acc = train_rashomon_models()