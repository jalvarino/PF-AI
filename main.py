import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# CONFIGURACIÓN GENERAL
# =========================
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 20

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cpu")  # Mantener CPU para reproducibilidad


# =========================
# PASO 4 – FUNCIÓN DE CARGA Y PREPROCESAMIENTO
# =========================
def load_prepare_split(csv_path, target="FLOOR", random_state=42):
    cols_to_drop = [
        "LONGITUDE", "LATITUDE", "SPACEID",
        "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"
    ]

    df = pd.read_csv(csv_path)
    df = df.drop(columns=cols_to_drop)

    X = df.filter(regex="^WAP")
    y = df[target]
    # Mantener solo pisos válidos (0–3)
    valid_floors = [0, 1, 2, 3]
    mask = y.isin(valid_floors)

    X = X[mask]
    y = y[mask]

    # Reemplazar ausencia de señal
    X[X == 100] = -100

    # Normalización MinMax
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long),
        torch.tensor(y_val.values, dtype=torch.long),
        scaler
    )


# =========================
# MODELO ANN
# =========================
class ANN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modules = []

        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.ReLU())

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


# =========================
# FUNCIÓN DE ENTRENAMIENTO
# =========================
def train_model(model, train_loader, val_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        running_train_loss = 0.0

        for Xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validación
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                outputs = model(Xb)
                loss = criterion(outputs, yb)
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

    total_time = time.time() - start_time
    return train_losses, val_losses, total_time


# =========================
# FUNCIÓN DE EVALUACIÓN
# =========================
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    y_true = y_test.cpu().numpy()

    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, average="macro"),
        "Recall": recall_score(y_true, preds, average="macro"),
        "F1-score": f1_score(y_true, preds, average="macro")
    }


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    os.makedirs("outputs/losses", exist_ok=True)

    print("\nCargando y preparando datos...")
    X_train, X_val, y_train, y_val, scaler = load_prepare_split(
        "datasets/UJIIndoorLoc/trainingData.csv"
    )

    # Cargar TEST
    test_df = pd.read_csv("datasets/UJIIndoorLoc/validationData.csv")
    test_df[test_df == 100] = -100

    X_test = scaler.transform(test_df.filter(regex="^WAP"))
    y_test = test_df["FLOOR"].values

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE
    )

    # =========================
    # ARQUITECTURAS
    # =========================
    ANN_ARCHITECTURES = {
        "Arquitectura_1": [520, 128, 4],
        "Arquitectura_2": [520, 256, 128, 4],
        "Arquitectura_3": [520, 256, 128, 64, 4],
        "Arquitectura_4": [520, 512, 256, 128, 64, 4],
        "Arquitectura_5": [520, 1024, 512, 256, 128, 64, 4],
    }

    all_results = []

    # =========================
    # ENTRENAMIENTO PRINCIPAL
    # =========================
    for name, layers in ANN_ARCHITECTURES.items():
        print(f"\nEntrenando {name} ...")

        model = ANN(layers).to(DEVICE)

        train_losses, val_losses, train_time = train_model(
            model, train_loader, val_loader, epochs=EPOCHS
        )

        metrics = evaluate_model(model, X_test, y_test)

        all_results.append({
            "Arquitectura": name,
            **metrics,
            "Tiempo_entrenamiento_s": round(train_time, 1)
        })

        # Guardar gráfica de pérdidas
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.title(name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"outputs/losses/{name}.png")
        plt.close()

    # =========================
    # RESULTADOS FINALES
    # =========================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("outputs/ann_results.csv", index=False)

    print("\nRESULTADOS FINALES:")
    print(results_df)
    print("\nResultados guardados en outputs/ann_results.csv")