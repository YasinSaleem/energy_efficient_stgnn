#!/usr/bin/env python3
"""
Energy-Optimized Continual Learning for STGNN

Implements continual learning with energy optimizations:
- Enhanced early stopping (patience=3, min_delta=1e-4)
- Structured pruning (30% of weights)

Loads the energy-optimized base model: stgnn_best_opt.pt

Author: Energy-Efficient STGNN Project
"""

import time
import json
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_preprocessing import get_base_dataloaders, get_cl_dataloaders
from model_stgnn import build_stgnn

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using: {DEVICE}")

# Base directory = src/
BASE_DIR = Path(__file__).resolve().parent

# Paths
BASE_MODEL_PATH = BASE_DIR / "models" / "stgnn_best_opt.pt"
CL_MODELS_DIR = BASE_DIR / "models" / "continual"
CL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Note: Results dir is inside src/ to match project structure
CL_RESULTS_DIR = BASE_DIR / "results" / "continual_learning"
CL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] BASE_MODEL_PATH = {BASE_MODEL_PATH}")
print(f"[paths] CL_MODELS_DIR   = {CL_MODELS_DIR}")
print(f"[paths] CL_RESULTS_DIR  = {CL_RESULTS_DIR}")

# Energy-Optimized CL Settings
CL_EPOCHS = 3
CL_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

# Enhanced Early Stopping Parameters
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 1e-4
EARLY_STOP_WARMUP = 1  # Minimal warmup for CL

# Pruning Parameters
PRUNING_AMOUNT = 0.3


# ============================================================================ #
# METRICS
# ============================================================================ #

def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict:
    """Compute evaluation metrics on already-batched tensors."""
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    mse = torch.mean((pred_flat - true_flat) ** 2).item()
    rmse = float(torch.sqrt(torch.tensor(mse)))
    mae = torch.mean(torch.abs(pred_flat - true_flat)).item()
    mape = torch.mean(torch.abs((pred_flat - true_flat) / (true_flat + 1e-8))).item() * 100

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
    }


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_trues = []

    for X, Y in loader:
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)
        preds = model(X)
        all_preds.append(preds.cpu())
        all_trues.append(Y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    return compute_metrics(all_preds, all_trues)


# ============================================================================ #
# ENERGY OPTIMIZATIONS
# ============================================================================ #

class EnhancedEarlyStopping:
    """Enhanced early stopping with min_delta and warmup."""
    
    def __init__(self, patience=3, min_delta=1e-4, warmup=1):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, epoch, val_loss):
        """Returns True if training should stop."""
        if epoch < self.warmup:
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            print(f"  🛑 Enhanced early stopping triggered at epoch {epoch + 1}")
            self.should_stop = True
            return True
        
        return False


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
        
        print(f"  ✂️  Applied pruning to {len(params_to_prune)} Linear layers ({amount*100:.0f}%)")
        return model
    except Exception as e:
        print(f"  ⚠️  Pruning failed: {e}")
        return model


# ============================================================================ #
# CONTINUAL UPDATE WITH OPTIMIZATIONS
# ============================================================================ #

def train_on_cl_window_optimized(
    model: torch.nn.Module,
    cl_loader,
    cl_name: str,
    epochs: int = CL_EPOCHS,
):
    """
    Fine-tune model on CL window with energy optimizations.
    
    Applies:
    - Enhanced early stopping
    - Structured pruning after training
    """
    print(f"\n{'=' * 70}")
    print(f"Energy-Optimized CL Update: {cl_name}")
    print(f"{'=' * 70}")
    print(f"Fine-tuning for up to {epochs} epochs with LR={CL_LEARNING_RATE}")
    print(f"Optimizations: Enhanced Early Stopping + Structured Pruning")
    
    # Clone model
    model = copy.deepcopy(model).to(DEVICE)
    model.train()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=CL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()
    
    # Enhanced early stopping
    early_stopping = EnhancedEarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        warmup=EARLY_STOP_WARMUP
    )
    
    start_time = time.time()
    epoch_losses = []
    actual_epochs = 0
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        progress = tqdm(
            cl_loader,
            desc=f"[{cl_name}] Epoch {epoch + 1}/{epochs}",
            leave=False,
        )
        
        for X, Y in progress:
            X = X.to(DEVICE, non_blocking=True)
            Y = Y.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, Y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        actual_epochs = epoch + 1
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.6f}")
        
        # Check early stopping
        if early_stopping(epoch, avg_loss):
            break
    
    duration = time.time() - start_time
    
    # Apply structured pruning
    print(f"\n  Applying structured pruning...")
    model = apply_structured_pruning(model, amount=PRUNING_AMOUNT)
    
    stats = {
        "cl_window": cl_name,
        "planned_epochs": epochs,
        "actual_epochs": actual_epochs,
        "early_stopped": early_stopping.should_stop,
        "duration_sec": duration,
        "duration_min": duration / 60,
        "final_loss": epoch_losses[-1],
        "loss_history": epoch_losses,
        "optimizations": ["enhanced_early_stopping", "structured_pruning"],
    }
    
    print(f"\n⏱️  Time: {duration:.1f}s ({duration / 60:.2f} min)")
    print(f"  Epochs run: {actual_epochs}/{epochs}")
    
    return model, stats


# ============================================================================ #
# FORGETTING MEASUREMENT
# ============================================================================ #

def compute_forgetting(current_metrics: dict, baseline_metrics: dict) -> dict:
    """Calculate forgetting (performance drop on old data)."""
    rmse_base = baseline_metrics["RMSE"]
    mae_base = baseline_metrics["MAE"]
    
    rmse_drop = current_metrics["RMSE"] - rmse_base
    mae_drop = current_metrics["MAE"] - mae_base
    
    rmse_pct = (rmse_drop / (rmse_base + 1e-8)) * 100
    mae_pct = (mae_drop / (mae_base + 1e-8)) * 100
    
    return {
        "RMSE_drop": rmse_drop,
        "MAE_drop": mae_drop,
        "RMSE_pct_change": rmse_pct,
        "MAE_pct_change": mae_pct,
        "baseline_RMSE": rmse_base,
        "current_RMSE": current_metrics["RMSE"],
    }


# ============================================================================ #
# MAIN PIPELINE
# ============================================================================ #

def run_energy_optimized_continual_learning():
    """
    Energy-optimized continual learning pipeline.
    
    Uses the energy-optimized base model and applies optimizations during CL.
    """
    print("\n" + "=" * 70)
    print("ENERGY-OPTIMIZED CONTINUAL LEARNING PIPELINE")
    print("=" * 70)
    
    # Load data
    print("\n📦 Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_base_dataloaders()
    cl_loaders = get_cl_dataloaders()
    
    if not cl_loaders:
        raise ValueError("No CL windows found! Check data/splits/continual/")
    
    cl_windows = sorted(cl_loaders.keys())
    print(f"  ✓ Found {len(cl_windows)} CL windows: {cl_windows}")
    
    # Load energy-optimized base model
    print(f"\n🧠 Loading energy-optimized base model from {BASE_MODEL_PATH}...")
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Energy-optimized model not found: {BASE_MODEL_PATH}\n"
            "Please run energy optimization training first."
        )
    
    base_model = build_stgnn().to(DEVICE)
    base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    print("  ✓ Energy-optimized base model loaded successfully")
    
    # Baseline evaluation
    print("\n📊 Evaluating base model on test set (pre-CL baseline)...")
    baseline_test_metrics = evaluate_model(base_model, test_loader)
    
    print("\n  BASELINE PERFORMANCE:")
    print(f"    Test RMSE: {baseline_test_metrics['RMSE']:.6f}")
    print(f"    Test MAE:  {baseline_test_metrics['MAE']:.6f}")
    print(f"    Test MAPE: {baseline_test_metrics['MAPE']:.2f}%")
    
    # Continual learning loop
    print(f"\n{'=' * 70}")
    print("STARTING ENERGY-OPTIMIZED CL UPDATES")
    print(f"{'=' * 70}")
    
    results = {
        "baseline_test_metrics": baseline_test_metrics,
        "cl_updates": [],
        "total_time_sec": 0.0,
        "model_type": "energy_optimized",
        "optimizations": ["enhanced_early_stopping", "structured_pruning"],
    }
    
    current_model = copy.deepcopy(base_model)
    
    for cl_name in cl_windows:
        print(f"\n{'#' * 70}")
        print(f"# Processing: {cl_name}")
        print(f"{'#' * 70}")
        
        cl_loader = cl_loaders[cl_name]
        
        # Fine-tune with optimizations
        updated_model, train_stats = train_on_cl_window_optimized(
            current_model,
            cl_loader,
            cl_name,
            epochs=CL_EPOCHS,
        )
        
        # Evaluate on new CL data
        print(f"\n  Evaluating on {cl_name} (new data)...")
        new_data_metrics = evaluate_model(updated_model, cl_loader)
        print(f"    New Data RMSE: {new_data_metrics['RMSE']:.6f}")
        print(f"    New Data MAE:  {new_data_metrics['MAE']:.6f}")
        
        # Evaluate on old test data (forgetting)
        print(f"\n  Evaluating on old test set (forgetting check)...")
        current_test_metrics = evaluate_model(updated_model, test_loader)
        forgetting = compute_forgetting(current_test_metrics, baseline_test_metrics)
        
        print(f"    Old Test RMSE: {current_test_metrics['RMSE']:.6f}")
        print(
            f"    Forgetting:    {forgetting['RMSE_drop']:+.6f} "
            f"({forgetting['RMSE_pct_change']:+.2f}%)"
        )
        
        if forgetting["RMSE_drop"] < 0:
            print("    ✅ Performance improved on old data!")
        elif abs(forgetting["RMSE_pct_change"]) < 5:
            print("    ✅ Minimal forgetting (<5%)")
        elif abs(forgetting["RMSE_pct_change"]) < 10:
            print("    ⚠️  Moderate forgetting (5–10%)")
        else:
            print("    ❌ Significant forgetting (>10%)")
        
        # Save updated model
        model_path = CL_MODELS_DIR / f"stgnn_energy_optimized_CL_{cl_name}.pt"
        torch.save(updated_model.state_dict(), model_path)
        print(f"\n  💾 Saved model: {model_path}")
        
        # Store results
        cl_result = {
            "window": cl_name,
            "train_stats": train_stats,
            "new_data_metrics": new_data_metrics,
            "old_test_metrics": current_test_metrics,
            "forgetting": forgetting,
        }
        
        results["cl_updates"].append(cl_result)
        results["total_time_sec"] += train_stats["duration_sec"]
        
        # Use updated model for next CL window
        current_model = updated_model
    
    # Save results
    results_path = CL_RESULTS_DIR / "energy_optimized_continual_learning_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("ENERGY-OPTIMIZED CONTINUAL LEARNING SUMMARY")
    print("=" * 70)
    
    print("\n📊 Baseline Performance (Before CL):")
    print(f"   Test RMSE: {baseline_test_metrics['RMSE']:.6f}")
    print(f"   Test MAE:  {baseline_test_metrics['MAE']:.6f}")
    
    print("\n📈 Per-Window Results:\n")
    print(f"{'Window':<10} {'New RMSE':<12} {'Old RMSE':<12} {'Forgetting':<15} {'Time (s)':<10} {'Epochs':<10}")
    print("-" * 80)
    
    for update in results["cl_updates"]:
        window = update["window"]
        new_rmse = update["new_data_metrics"]["RMSE"]
        old_rmse = update["old_test_metrics"]["RMSE"]
        forgetting_pct = update["forgetting"]["RMSE_pct_change"]
        duration = update["train_stats"]["duration_sec"]
        actual_epochs = update["train_stats"]["actual_epochs"]
        
        print(
            f"{window:<10} {new_rmse:<12.6f} {old_rmse:<12.6f} "
            f"{forgetting_pct:+6.2f}%        {duration:<10.1f} {actual_epochs:<10}"
        )
    
    print("-" * 80)
    
    final_update = results["cl_updates"][-1]
    final_rmse = final_update["old_test_metrics"]["RMSE"]
    final_forgetting = final_update["forgetting"]["RMSE_pct_change"]
    
    print("\n🏆 Final Model (after all CL updates):")
    print(f"   Test RMSE: {final_rmse:.6f}")
    print(f"   Total Forgetting: {final_forgetting:+.2f}%")
    print(
        f"   Total Time: {results['total_time_sec']:.1f}s "
        f"({results['total_time_sec'] / 60:.2f} min)"
    )
    
    avg_forgetting = float(
        np.mean([u["forgetting"]["RMSE_pct_change"] for u in results["cl_updates"]])
    )
    
    total_early_stops = sum(
        1 for u in results["cl_updates"] if u["train_stats"]["early_stopped"]
    )
    
    print("\n📊 Statistics:")
    print(f"   CL Windows Processed: {len(results['cl_updates'])}")
    print(f"   Average Forgetting: {avg_forgetting:+.2f}%")
    print(f"   Early Stops: {total_early_stops}/{len(results['cl_updates'])}")
    print(
        f"   Average Time per Update: "
        f"{results['total_time_sec'] / len(results['cl_updates']):.1f}s"
    )
    
    print(f"\n💾 Results saved to: {results_path}")
    print(f"💾 Updated models saved to: {CL_MODELS_DIR}/")
    
    print("\n" + "=" * 70)
    print("✅ ENERGY-OPTIMIZED CONTINUAL LEARNING COMPLETED!")
    print("=" * 70 + "\n")
    
    return results


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == "__main__":
    run_energy_optimized_continual_learning()
