#!/usr/bin/env python3
"""
STGNN Training Script (Stable + Memory-Safe)
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Reduce fragmentation issues
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import config as cfg
from src.data_preprocessing import get_base_dataloaders
from src.model_stgnn import build_stgnn

# ============================
# DEVICE SELECTION
# ============================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"\n✅ Training on CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"\n✅ Training on Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("\n⚠️ Training on CPU (no GPU detected)")

# ============================
# LOSS FUNCTION
# ============================

def get_loss_fn():
    name = cfg.LOSS_FN.lower()
    if name == "mse":
        print("🔧 Using MSELoss")
        return nn.MSELoss()
    if name == "mae":
        print("🔧 Using L1Loss (MAE)")
        return nn.L1Loss()
    if name == "huber":
        print(f"🔧 Using SmoothL1Loss (Huber), delta={cfg.HUBER_DELTA}")
        return nn.SmoothL1Loss(beta=cfg.HUBER_DELTA)
    print(f"⚠️ Unknown LOSS_FN='{cfg.LOSS_FN}', falling back to MSELoss")
    return nn.MSELoss()

# ============================
# METRICS
# ============================

def rmse(pred, true):
    return torch.sqrt(F.mse_loss(pred, true))

def mae(pred, true):
    return torch.mean(torch.abs(pred - true))

# ============================
# TRAIN LOOP
# ============================

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for X, Y in tqdm(loader, desc="Training", leave=False):
        # X, Y come from CPU; move small batch to DEVICE
        X = X.to(DEVICE, non_blocking=True)  # [B, W, N, 1]
        Y = Y.to(DEVICE, non_blocking=True)  # [B, H, N, 1]

        optimizer.zero_grad()
        preds = model(X)            # ensure model accepts [B, W, N, 1]
        loss = loss_fn(preds, Y)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        if isinstance(grad_norm, torch.Tensor):
            total_grad_norm += grad_norm.item()
        else:
            total_grad_norm += float(grad_norm)
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0

    return total_loss / num_batches, total_grad_norm / num_batches

# ============================
# VALIDATION LOOP
# ============================

@torch.no_grad()
def evaluate(model, loader, desc="Validation"):
    model.eval()

    total_mse = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    num_batches = 0

    for X, Y in tqdm(loader, desc=desc, leave=False):
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)

        preds = model(X)

        total_mse  += F.mse_loss(preds, Y).item()
        total_rmse += rmse(preds, Y).item()
        total_mae  += mae(preds, Y).item()
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_mse  / num_batches,
        total_rmse / num_batches,
        total_mae  / num_batches,
    )

# ============================
# MAIN TRAINING
# ============================

def main():
    # Print config summary
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"  Max Epochs:              {cfg.EPOCHS}")
    print(f"  Learning Rate:           {cfg.LEARNING_RATE}")
    print(f"  Weight Decay:            {cfg.WEIGHT_DECAY}")
    print(f"  Gradient Clip Norm:      {cfg.GRADIENT_CLIP_NORM}")
    print(f"  Early Stopping Patience: {cfg.EARLY_STOPPING_PATIENCE}")
    print(f"  Batch Size:              {cfg.BATCH_SIZE}")
    print(f"  Loss Function:           {cfg.LOSS_FN}")
    print()
    print(f"  Spatial Dropout:         {cfg.SPATIAL_DROPOUT}")
    print(f"  Temporal Dropout:        {cfg.TEMPORAL_DROPOUT}")
    print(f"  Final Dropout:           {cfg.FINAL_DROPOUT}")
    print()
    print(f"  Scheduler Mode:          {cfg.SCHEDULER_MODE}")
    print(f"  Scheduler Factor:        {cfg.SCHEDULER_FACTOR}")
    print(f"  Scheduler Patience:      {cfg.SCHEDULER_PATIENCE}")
    print(f"  Scheduler Min LR:        {cfg.SCHEDULER_MIN_LR}")
    print("="*80 + "\n")

    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_PATH = MODEL_DIR / "stgnn_best.pt"

    print("\n📦 Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_base_dataloaders()

    print("\n🧠 Building STGNN Model...")
    model = build_stgnn().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {total_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.SCHEDULER_MODE,
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
        min_lr=cfg.SCHEDULER_MIN_LR,
        verbose=cfg.SCHEDULER_VERBOSE,
    )
    print(f"[scheduler] ReduceLROnPlateau configured (factor={cfg.SCHEDULER_FACTOR}, patience={cfg.SCHEDULER_PATIENCE})")

    loss_fn = get_loss_fn()

    best_val_rmse = float("inf")
    patience_counter = 0

    print("\n🚀 Starting Training...\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{cfg.EPOCHS} ==========")

        train_loss, avg_grad_norm = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_mse, val_rmse, val_mae = evaluate(model, val_loader, desc="Validation")

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss ({cfg.LOSS_FN.upper()}): {train_loss:.6f}")
        print(f"Avg Grad Norm:              {avg_grad_norm:.6f}")
        print(f"Val   MSE:                  {val_mse:.6f}")
        print(f"Val   RMSE:                 {val_rmse:.6f}")
        print(f"Val   MAE:                  {val_mae:.6f}")
        print(f"Current LR:                 {current_lr:.2e}")

        # Scheduler on val RMSE
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Best Model Saved → {MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"⏳ EarlyStopping Counter: {patience_counter}/{cfg.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print("\n🛑 Early stopping triggered.")
            break

    print("\n✅ Training Completed!")
    print(f"🏆 Best Validation RMSE: {best_val_rmse:.6f}")

    print("\n🔍 Running Final Test Evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    test_mse, test_rmse, test_mae = evaluate(model, test_loader, desc="Test")

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test MSE : {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE : {test_mae:.6f}")
    print("=======================================\n")

    return {
        "best_val_rmse": float(best_val_rmse),
        "test_mse": float(test_mse),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
    }

if __name__ == "__main__":
    main()
