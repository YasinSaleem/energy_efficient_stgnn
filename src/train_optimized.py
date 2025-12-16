#!/usr/bin/env python3
"""
Energy-Optimized STGNN Training Script

Uses the same optimizations as energy-optimized continual learning:
- Enhanced early stopping (patience=5, min_delta=1e-4, warmup=3)
- Structured pruning (30% after training completes)

Saves to: src/models/stgnn_best_opt.pt
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
# ENERGY OPTIMIZATION CONFIGS
# ============================

# Enhanced Early Stopping Parameters
EARLY_STOP_PATIENCE = 5      # More patient than CL (5 vs 3)
EARLY_STOP_MIN_DELTA = 1e-4  # Same as CL
EARLY_STOP_WARMUP = 3        # More warmup than CL (3 vs 1)

# Pruning Parameters
PRUNING_AMOUNT = 0.3         # 30% pruning, same as CL

# ============================
# ENHANCED EARLY STOPPING
# ============================

class EnhancedEarlyStopping:
    """Enhanced early stopping with min_delta and warmup."""
    
    def __init__(self, patience=5, min_delta=1e-4, warmup=3):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, epoch, val_loss):
        """Returns True if training should stop."""
        if epoch < self.warmup:
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            print(f"\n  🛑 Enhanced early stopping triggered at epoch {epoch + 1}")
            print(f"  📊 Best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch + 1}")
            self.should_stop = True
            return True
        
        return False


# ============================
# STRUCTURED PRUNING
# ============================

def apply_structured_pruning(model, amount=0.3):
    """Apply L1 unstructured pruning to Linear layers."""
    try:
        import torch.nn.utils.prune as prune
        
        params_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                params_to_prune.append((module, 'weight'))
        
        for module, param_name in params_to_prune:
            prune.l1_unstructured(module, name=param_name, amount=amount)
            prune.remove(module, param_name)
        
        print(f"\n  ✂️  Applied pruning to {len(params_to_prune)} Linear layers ({amount*100:.0f}%)")
        return model
    except Exception as e:
        print(f"\n  ⚠️  Pruning failed: {e}")
        return model


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
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        preds = model(X)
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
    print("\n" + "="*80)
    print("ENERGY-OPTIMIZED TRAINING CONFIGURATION")
    print("="*80)
    print(f"  Max Epochs:              {cfg.EPOCHS}")
    print(f"  Learning Rate:           {cfg.LEARNING_RATE}")
    print(f"  Weight Decay:            {cfg.WEIGHT_DECAY}")
    print(f"  Gradient Clip Norm:      {cfg.GRADIENT_CLIP_NORM}")
    print(f"  Batch Size:              {cfg.BATCH_SIZE}")
    print(f"  Loss Function:           {cfg.LOSS_FN}")
    print()
    print("  🔋 Energy Optimizations:")
    print(f"    Enhanced Early Stopping:")
    print(f"      - Patience:   {EARLY_STOP_PATIENCE}")
    print(f"      - Min Delta:  {EARLY_STOP_MIN_DELTA}")
    print(f"      - Warmup:     {EARLY_STOP_WARMUP}")
    print(f"    Structured Pruning:")
    print(f"      - Amount:     {PRUNING_AMOUNT * 100:.0f}%")
    print()
    print(f"  Scheduler Mode:          {cfg.SCHEDULER_MODE}")
    print(f"  Scheduler Factor:        {cfg.SCHEDULER_FACTOR}")
    print(f"  Scheduler Patience:      {cfg.SCHEDULER_PATIENCE}")
    print(f"  Scheduler Min LR:        {cfg.SCHEDULER_MIN_LR}")
    print("="*80 + "\n")

    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_PATH = MODEL_DIR / "stgnn_best_opt.pt"

    print("📦 Loading DataLoaders...")
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

    # Enhanced early stopping
    early_stopping = EnhancedEarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        warmup=EARLY_STOP_WARMUP
    )
    print(f"\n[early_stop] Enhanced mode: patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}, warmup={EARLY_STOP_WARMUP}")

    best_val_rmse = float("inf")
    best_model_state = None

    print("\n🚀 Starting Energy-Optimized Training...\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{cfg.EPOCHS} ==========")

        # Train
        train_loss, grad_norm = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"[train] loss={train_loss:.6f}, grad_norm={grad_norm:.4f}")

        # Validate
        val_mse, val_rmse, val_mae = evaluate(model, val_loader, desc="Validation")
        print(f"[val]   mse={val_mse:.6f}, rmse={val_rmse:.6f}, mae={val_mae:.6f}")

        # Scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_rmse)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"[scheduler] LR reduced: {current_lr:.2e} → {new_lr:.2e}")

        # Track best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            print(f"✅ New best validation RMSE: {best_val_rmse:.6f}")

        # Check enhanced early stopping
        if early_stopping(epoch - 1, val_rmse):
            print(f"\n⏹️  Training stopped early at epoch {epoch}")
            break

    # Load best model state before pruning
    if best_model_state is not None:
        print(f"\n📥 Loading best model state (val_rmse={best_val_rmse:.6f})")
        model.load_state_dict(best_model_state)
    
    # Apply structured pruning
    print(f"\n✂️  Applying structured pruning ({PRUNING_AMOUNT*100:.0f}%)...")
    model = apply_structured_pruning(model, amount=PRUNING_AMOUNT)

    # Final evaluation on test set
    print("\n📊 Evaluating pruned model on test set...")
    test_mse, test_rmse, test_mae = evaluate(model, test_loader, desc="Test")
    print(f"[test] mse={test_mse:.6f}, rmse={test_rmse:.6f}, mae={test_mae:.6f}")

    # Save the optimized model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n💾 Saved optimized model to: {MODEL_PATH}")

    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"  Best Val RMSE:      {best_val_rmse:.6f}")
    print(f"  Final Test RMSE:    {test_rmse:.6f}")
    print(f"  Epochs Run:         {epoch}/{cfg.EPOCHS}")
    print(f"  Early Stopped:      {early_stopping.should_stop}")
    print(f"  Pruning Applied:    {PRUNING_AMOUNT*100:.0f}%")
    print(f"  Model Saved:        {MODEL_PATH}")
    print("="*80 + "\n")

    print("✅ Energy-optimized training complete!")
    print("   Ready for continual learning with:")
    print("   python energy_optimized_continual_learning.py\n")


if __name__ == "__main__":
    main()
