#!/usr/bin/env python3
"""
STGNN Training Script (GPU Forced + Early Stopping + Metrics)

✅ Forces CUDA GPU
✅ Progress bars (tqdm)
✅ Early stopping for energy efficiency
✅ Tracks MSE, RMSE, MAE
✅ Saves best validation model
✅ Designed for 1–2 hour training window
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import numpy as np

from data_preprocessing import get_base_dataloaders
from model_stgnn import build_stgnn

# ============================
# FORCE GPU
# ============================

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA GPU not detected. Training is GPU-only.")

DEVICE = torch.device("cuda")
print(f"\n✅ Training on GPU: {torch.cuda.get_device_name(0)}")

# ============================
# CONFIGURATION
# ============================

EPOCHS = 30                # Fits 1–2 hour window
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 6   # Stop after 6 bad val epochs

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "stgnn_best.pt"

# ============================
# METRICS
# ============================

def rmse(pred, true):
    return torch.sqrt(nn.functional.mse_loss(pred, true))

def mae(pred, true):
    return torch.mean(torch.abs(pred - true))

# ============================
# TRAIN LOOP
# ============================

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for X, Y in tqdm(loader, desc="Training", leave=False):
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ============================
# VALIDATION LOOP
# ============================

@torch.no_grad()
def validate(model, loader):
    model.eval()

    total_mse = 0
    total_rmse = 0
    total_mae = 0

    for X, Y in tqdm(loader, desc="Validation", leave=False):
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)

        preds = model(X)

        total_mse += nn.functional.mse_loss(preds, Y).item()
        total_rmse += rmse(preds, Y).item()
        total_mae += mae(preds, Y).item()

    n = len(loader)
    return total_mse / n, total_rmse / n, total_mae / n

# ============================
# MAIN TRAINING PIPELINE
# ============================

def main():
    print("\n📦 Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_base_dataloaders()

    print("\n🧠 Building STGNN Model...")
    model = build_stgnn().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    patience_counter = 0

    print("\n🚀 Starting Training...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{EPOCHS} ==========")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_mse, val_rmse, val_mae = validate(model, val_loader)

        print(f"Train MSE: {train_loss:.6f}")
        print(f"Val   MSE: {val_mse:.6f}")
        print(f"Val   RMSE: {val_rmse:.6f}")
        print(f"Val   MAE: {val_mae:.6f}")

        # ✅ Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Best Model Saved → {MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"⏳ EarlyStopping Counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        # ✅ EARLY STOPPING
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\n🛑 Early stopping triggered to save energy.")
            break

    print("\n✅ Training Completed!")
    print(f"🏆 Best Validation RMSE: {best_val_rmse:.6f}")

    # ============================
    # FINAL TEST EVALUATION
    # ============================

    print("\n🔍 Running Final Test Evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH))

    test_mse, test_rmse, test_mae = validate(model, test_loader)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test MSE : {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE : {test_mae:.6f}")
    print("=======================================\n")

# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
