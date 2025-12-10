#!/usr/bin/env python3
"""
STGNN Training Script (Improved with Regularization)

✅ Forces CUDA GPU
✅ Progress bars (tqdm)
✅ Enhanced regularization (dropout, weight decay, gradient clipping)
✅ Learning rate scheduler (ReduceLROnPlateau)
✅ Early stopping for energy efficiency
✅ Tracks MSE, RMSE, MAE
✅ Saves best validation model
✅ Centralized configuration
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import config as cfg
from utils.ema import ModelEMA
from src.data_preprocessing import get_base_dataloaders
from src.model_stgnn import build_stgnn

# ============================
# DEVICE SELECTION (CUDA or MPS)
# ============================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"\n✅ Training on CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"\n✅ Training on Apple Silicon GPU (MPS)")
else:
    raise RuntimeError("❌ No GPU detected. Training requires either CUDA or MPS GPU.")

# ============================
# CONFIGURATION (from utils.config)
# ============================

print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print(f"  Max Epochs:              {cfg.EPOCHS}")
print(f"  Learning Rate:           {cfg.LEARNING_RATE}")
print(f"  Weight Decay:            {cfg.WEIGHT_DECAY}")
print(f"  Gradient Clip Norm:      {cfg.GRADIENT_CLIP_NORM}")
print(f"  Early Stopping Patience: {cfg.EARLY_STOPPING_PATIENCE}")
print(f"  Batch Size:              {cfg.BATCH_SIZE}")
print(f"")
print(f"  Spatial Dropout:         {cfg.SPATIAL_DROPOUT}")
print(f"  Temporal Dropout:        {cfg.TEMPORAL_DROPOUT}")
print(f"  Final Dropout:           {cfg.FINAL_DROPOUT}")
print(f"")
print(f"  Scheduler Mode:          {cfg.SCHEDULER_MODE}")
print(f"  Scheduler Factor:        {cfg.SCHEDULER_FACTOR}")
print(f"  Scheduler Patience:      {cfg.SCHEDULER_PATIENCE}")
print(f"  Scheduler Min LR:        {cfg.SCHEDULER_MIN_LR}")
print(f"")
print(f"  Use EMA:                 {cfg.USE_EMA}")
if cfg.USE_EMA:
    print(f"  EMA Decay:               {cfg.EMA_DECAY}")
    print(f"  EMA Update After Step:   {cfg.EMA_UPDATE_AFTER_STEP}")
print("="*80 + "\n")

MODEL_DIR = Path(__file__).parent / "models"
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

def train_one_epoch(model, loader, optimizer, loss_fn, ema=None, epoch=0):
    model.train()
    total_loss = 0
    
    # Calculate global step offset for this epoch
    steps_per_epoch = len(loader)
    step_offset = (epoch - 1) * steps_per_epoch if epoch > 0 else 0

    for batch_idx, (X, Y) in enumerate(tqdm(loader, desc="Training", leave=False)):
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, Y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
        
        optimizer.step()
        
        # Update EMA after optimizer step
        if ema is not None:
            global_step = step_offset + batch_idx
            ema.update(model, global_step)

        total_loss += loss.item()

    return total_loss / len(loader)

# ============================
# VALIDATION LOOP
# ============================

@torch.no_grad()
def validate(model, loader, ema=None, use_ema=False):
    model.eval()

    total_mse = 0
    total_rmse = 0
    total_mae = 0
    
    # Use EMA weights if requested and available
    if use_ema and ema is not None:
        context = ema.average_parameters(model)
    else:
        from contextlib import nullcontext
        context = nullcontext()
    
    with context:
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
    
    # Initialize EMA if enabled
    ema = None
    if cfg.USE_EMA:
        ema = ModelEMA(
            model,
            decay=cfg.EMA_DECAY,
            update_after_step=cfg.EMA_UPDATE_AFTER_STEP,
            device=DEVICE
        )
        print(f"[EMA] Enabled with decay={cfg.EMA_DECAY}, update_after_step={cfg.EMA_UPDATE_AFTER_STEP}")

    # Optimizer with weight decay (L2 regularization)
    # Start with warmup LR, will be adjusted during warmup phase
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.WARMUP_START_LR,  # Start with warmup LR
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    # Learning rate scheduler - reduces LR when validation plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.SCHEDULER_MODE,
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
        min_lr=cfg.SCHEDULER_MIN_LR
    )
    print(f"[scheduler] ReduceLROnPlateau configured (factor={cfg.SCHEDULER_FACTOR}, patience={cfg.SCHEDULER_PATIENCE})")
    print(f"[warmup] Warmup enabled for {cfg.WARMUP_EPOCHS} epochs: {cfg.WARMUP_START_LR:.2e} → {cfg.LEARNING_RATE:.2e}")
    
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    patience_counter = 0

    print("\n🚀 Starting Training...\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{cfg.EPOCHS} ==========")

        # Warmup learning rate schedule
        if epoch <= cfg.WARMUP_EPOCHS:
            # Linear warmup from WARMUP_START_LR to LEARNING_RATE
            warmup_lr = cfg.WARMUP_START_LR + (cfg.LEARNING_RATE - cfg.WARMUP_START_LR) * (epoch / cfg.WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"[warmup] Epoch {epoch}/{cfg.WARMUP_EPOCHS}: LR set to {warmup_lr:.2e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, ema=ema, epoch=epoch)
        
        # Validate with regular weights
        val_mse, val_rmse, val_mae = validate(model, val_loader, ema=ema, use_ema=False)
        
        # Validate with EMA weights if enabled
        if ema is not None:
            val_mse_ema, val_rmse_ema, val_mae_ema = validate(model, val_loader, ema=ema, use_ema=True)

        print(f"Train MSE: {train_loss:.6f}")
        print(f"Val   MSE: {val_mse:.6f} | RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f}")
        if ema is not None:
            print(f"Val (EMA) MSE: {val_mse_ema:.6f} | RMSE: {val_rmse_ema:.6f} | MAE: {val_mae_ema:.6f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Only step scheduler after warmup period
        # Use EMA metrics for scheduler if available
        metric_for_scheduler = val_rmse_ema if ema is not None else val_rmse
        if epoch > cfg.WARMUP_EPOCHS:
            scheduler.step(metric_for_scheduler)

        # ✅ Save best model (using EMA metrics if available)
        current_val_rmse = val_rmse_ema if ema is not None else val_rmse
        if current_val_rmse < best_val_rmse:
            best_val_rmse = current_val_rmse
            patience_counter = 0
            
            # Save checkpoint with both model and EMA state
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_rmse': best_val_rmse,
            }
            if ema is not None:
                checkpoint['ema_state_dict'] = ema.state_dict()
            
            torch.save(checkpoint, MODEL_PATH)
            print(f"💾 Best Model Saved → {MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"⏳ EarlyStopping Counter: {patience_counter}/{cfg.EARLY_STOPPING_PATIENCE}")

        # ✅ EARLY STOPPING
        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print("\n🛑 Early stopping triggered to save energy.")
            break

    print("\n✅ Training Completed!")
    print(f"🏆 Best Validation RMSE: {best_val_rmse:.6f}")

    # ============================
    # FINAL TEST EVALUATION
    # ============================

    print("\n🔍 Running Final Test Evaluation...")
    checkpoint = torch.load(MODEL_PATH)
    
    # Load model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load EMA state if available
        if ema is not None and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            print("[EMA] Loaded EMA state from checkpoint")
    else:
        # Legacy checkpoint format (just state dict)
        model.load_state_dict(checkpoint)
    
    # Test with regular weights
    test_mse, test_rmse, test_mae = validate(model, test_loader, ema=ema, use_ema=False)
    
    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test MSE : {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE : {test_mae:.6f}")
    
    # Test with EMA weights if available
    if ema is not None:
        test_mse_ema, test_rmse_ema, test_mae_ema = validate(model, test_loader, ema=ema, use_ema=True)
        print("\n========== FINAL TEST RESULTS (EMA) ====")
        print(f"Test MSE : {test_mse_ema:.6f}")
        print(f"Test RMSE: {test_rmse_ema:.6f}")
        print(f"Test MAE : {test_mae_ema:.6f}")
    
    print("=======================================\n")

# ============================
# ENTRY POINT
# ============================

if __name__ == "__main__":
    main()
