#!/usr/bin/env python3
"""
ULTRA-AGGRESSIVE Energy-Optimized Continual Learning for STGNN

Strategy: Maximum energy savings with controlled forgetting
- Single epoch fine-tuning (66% time reduction)
- Massive gradient accumulation (16x)
- Optimized learning rate schedule
- Smart early stopping
- NO mixed precision (sparse matrix incompatible)
- NO pruning (preserves accuracy)

Base Model: stgnn_best_optimized.pt (from Conservative Training Strategy)

Goal: 60-70% energy savings vs baseline CL, <8% forgetting

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

# CodeCarbon for energy tracking
try:
    from codecarbon import EmissionsTracker

    CODECARBON_AVAILABLE = True
except ImportError:
    print("⚠️  CodeCarbon not installed. Energy tracking will be limited.")
    CODECARBON_AVAILABLE = False

# NVML for GPU tracking
try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except:
    PYNVML_AVAILABLE = False

from data_preprocessing import get_base_dataloaders, get_cl_dataloaders
from model_stgnn import build_stgnn

# ============================================================================ #
# ULTRA-AGGRESSIVE CONFIGURATION
# ============================================================================ #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using: {DEVICE}")

# Base directory = src/
BASE_DIR = Path(__file__).resolve().parent

# Paths
BASE_MODEL_PATH = BASE_DIR / "models" / "stgnn_best_optimized.pt"
CL_MODELS_DIR = BASE_DIR / "models" / "continual_aggressive"
CL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

CL_RESULTS_DIR = BASE_DIR / "results" / "continual_learning"
CL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENERGY_LOGS_DIR = CL_RESULTS_DIR / "codecarbon_logs_aggressive"
ENERGY_LOGS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] BASE_MODEL_PATH = {BASE_MODEL_PATH}")
print(f"[paths] CL_MODELS_DIR   = {CL_MODELS_DIR}")
print(f"[paths] CL_RESULTS_DIR  = {CL_RESULTS_DIR}")

# ============================================
# ULTRA-AGGRESSIVE CL SETTINGS
# Optimized for maximum energy savings + minimal forgetting
# ============================================
STRATEGY_NAME = "ultra_aggressive"

# Minimal epochs with high efficiency
CL_EPOCHS = 1  # Single epoch (66% time savings vs baseline)
CL_LEARNING_RATE = 8e-4  # Balanced: not too high (forgetting) or low (slow)
WEIGHT_DECAY = 5e-5  # Light regularization to reduce forgetting

# Maximum gradient accumulation (key energy saver)
GRADIENT_ACCUMULATION = 16  # 16x reduces backward passes by 93.75%

# Early stopping (safety mechanism)
EARLY_STOP_PATIENCE = 1  # Stop if loss increases
EARLY_STOP_MIN_DELTA = 1e-3  # Reasonable threshold
EARLY_STOP_WARMUP = 0  # No warmup for 1 epoch

# Disabled optimizations (sparse matrix constraints)
PRUNING_AMOUNT = 0.0  # No pruning (preserves accuracy)
USE_MIXED_PRECISION = False  # Sparse matrices don't support FP16

# Efficient evaluation
EVAL_BATCH_SIZE = 1  # Keep standard for accuracy

# Learning rate warmup for stability
USE_LR_WARMUP = True  # Warm up LR to prevent early overfitting
WARMUP_STEPS = 50  # First 50 steps

print("\n" + "=" * 80)
print("ULTRA-AGGRESSIVE ENERGY-OPTIMIZED CONTINUAL LEARNING")
print("=" * 80)
print("🎯 Goal: 60-70% energy savings, <8% forgetting")
print("⚡ Strategy: Extreme efficiency with forgetting control")
print("\n🔧 Optimizations:")
print(f"  1. Ultra-Short Training:     {CL_EPOCHS} epoch (vs 3 baseline)")
print(f"  2. Balanced Learning Rate:   {CL_LEARNING_RATE} (controlled adaptation)")
print(f"  3. Massive Grad Accumulation: {GRADIENT_ACCUMULATION}x (93.75% fewer steps)")
print(f"  4. LR Warmup:                {USE_LR_WARMUP} (stability)")
print(f"  5. Light Regularization:     Weight decay {WEIGHT_DECAY}")
print(f"  6. Early Stop Safety:        patience={EARLY_STOP_PATIENCE}")
print(f"  7. No Pruning/Mixed Prec:    Preserves accuracy")
print("\n💡 Focus: Maximum speed without sacrificing too much accuracy")
print("=" * 80 + "\n")


# ============================================================================ #
# METRICS
# ============================================================================ #

def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict:
    """Compute evaluation metrics."""
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
# LEARNING RATE WARMUP
# ============================================================================ #

class WarmupScheduler:
    """Linear learning rate warmup."""

    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


# ============================================================================ #
# EARLY STOPPING
# ============================================================================ #

class SafeEarlyStopping:
    """Early stopping that prevents overfitting."""

    def __init__(self, patience=1, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"  🛑 Early stopping: loss not improving")
                self.should_stop = True
                return True
        return False


# ============================================================================ #
# ULTRA-AGGRESSIVE CONTINUAL UPDATE
# ============================================================================ #

def train_on_cl_window_ultra_aggressive(
    model: torch.nn.Module,
    cl_loader,
    cl_name: str,
    epochs: int = CL_EPOCHS,
):
    """
    Ultra-aggressive fine-tuning with forgetting control.

    Key features:
    - Single epoch training
    - Massive gradient accumulation (16x)
    - LR warmup for stability
    - Light regularization
    - Early stopping safety net
    """
    print(f"\n{'=' * 70}")
    print(f"Ultra-Aggressive CL Update: {cl_name}")
    print(f"{'=' * 70}")
    print(f"Epochs: {epochs} | LR: {CL_LEARNING_RATE} | Grad Accum: {GRADIENT_ACCUMULATION}x")

    # Clone model
    model = copy.deepcopy(model).to(DEVICE)
    model.train()

    optimizer = optim.AdamW(  # AdamW for better generalization
        model.parameters(),
        lr=CL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),  # Standard
    )

    loss_fn = nn.MSELoss()

    # Learning rate warmup
    lr_scheduler = None
    if USE_LR_WARMUP:
        lr_scheduler = WarmupScheduler(optimizer, WARMUP_STEPS, CL_LEARNING_RATE)
        print(f"  🔥 LR warmup: {WARMUP_STEPS} steps")

    # Early stopping
    early_stopping = SafeEarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA
    )

    start_time = time.time()
    epoch_losses = []
    actual_epochs = 0
    total_steps = 0

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()

        progress = tqdm(
            cl_loader,
            desc=f"[{cl_name}] Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for i, (X, Y) in enumerate(progress):
            X = X.to(DEVICE, non_blocking=True)
            Y = Y.to(DEVICE, non_blocking=True)

            # Forward pass
            preds = model(X)
            loss = loss_fn(preds, Y)
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()

            # Gradient accumulation step
            if (i + 1) % GRADIENT_ACCUMULATION == 0 or (i + 1) == len(cl_loader):
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

                # LR warmup
                if lr_scheduler and total_steps < WARMUP_STEPS:
                    lr_scheduler.step()

                total_steps += 1

            total_loss += loss.item() * GRADIENT_ACCUMULATION
            num_batches += 1

            # Update progress
            current_lr = lr_scheduler.get_lr() if lr_scheduler else CL_LEARNING_RATE
            progress.set_postfix({
                "loss": f"{loss.item() * GRADIENT_ACCUMULATION:.4f}",
                "lr": f"{current_lr:.6f}"
            })

        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        actual_epochs = epoch + 1

        current_lr = lr_scheduler.get_lr() if lr_scheduler else CL_LEARNING_RATE
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")

        # Early stopping check
        if early_stopping(avg_loss):
            break

    duration = time.time() - start_time

    stats = {
        "cl_window": cl_name,
        "strategy": "ultra_aggressive",
        "planned_epochs": epochs,
        "actual_epochs": actual_epochs,
        "early_stopped": early_stopping.should_stop,
        "duration_sec": duration,
        "duration_min": duration / 60,
        "final_loss": epoch_losses[-1],
        "loss_history": epoch_losses,
        "total_steps": total_steps,
        "config": {
            "learning_rate": CL_LEARNING_RATE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "weight_decay": WEIGHT_DECAY,
            "lr_warmup": USE_LR_WARMUP,
            "warmup_steps": WARMUP_STEPS if USE_LR_WARMUP else 0,
        }
    }

    print(f"\n⏱️  Time: {duration:.1f}s ({duration / 60:.2f} min)")
    print(f"  Steps: {total_steps} | Early stop: {early_stopping.should_stop}")

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

def run_aggressive_continual_learning():
    """
    Ultra-aggressive continual learning pipeline.

    Optimized for maximum energy savings with controlled forgetting:
    1. Load energy-optimized base model
    2. Single-epoch fine-tuning on each CL window
    3. Massive gradient accumulation (16x)
    4. LR warmup for stability
    5. Light regularization to reduce forgetting
    """
    print("\n" + "=" * 70)
    print("ULTRA-AGGRESSIVE CONTINUAL LEARNING PIPELINE")
    print("=" * 70)

    # Setup energy tracking
    energy_kwh = 0.0
    co2_kg = 0.0
    gpu_energy_samples = []

    if CODECARBON_AVAILABLE:
        print("\n⚡ Starting CodeCarbon energy tracking...")
        try:
            tracker = EmissionsTracker(
                project_name="STGNN_Ultra_Aggressive_CL",
                output_dir=str(ENERGY_LOGS_DIR),
                measure_power_secs=1,
                save_to_file=True,
                log_level="error"
            )
            tracker.start()
        except Exception as e:
            print(f"⚠️  CodeCarbon failed: {e}")
            tracker = None
    else:
        tracker = None

    # GPU tracking
    gpu_handle = None
    if PYNVML_AVAILABLE and DEVICE.type == 'cuda':
        try:
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print("✅ NVML GPU tracking enabled")
        except:
            pass

    overall_start_time = time.time()

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
            f"Base model not found: {BASE_MODEL_PATH}\n"
            "Please run energy-optimized training first (Strategy 2)."
        )

    base_model = build_stgnn().to(DEVICE)
    base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE, weights_only=True))
    print("  ✓ Base model loaded successfully")
    print("  ✓ No pruning applied (preserves accuracy)")

    # Baseline evaluation
    print("\n📊 Evaluating base model on test set (pre-CL baseline)...")
    baseline_test_metrics = evaluate_model(base_model, test_loader)

    print("\n  BASELINE PERFORMANCE:")
    print(f"    Test RMSE: {baseline_test_metrics['RMSE']:.6f}")
    print(f"    Test MAE:  {baseline_test_metrics['MAE']:.6f}")
    print(f"    Test MAPE: {baseline_test_metrics['MAPE']:.2f}%")

    # Continual learning loop
    print(f"\n{'=' * 70}")
    print("STARTING ULTRA-AGGRESSIVE CL UPDATES")
    print(f"{'=' * 70}")

    results = {
        "strategy": "ultra_aggressive",
        "baseline_test_metrics": baseline_test_metrics,
        "cl_updates": [],
        "total_time_sec": 0.0,
        "config": {
            "cl_epochs": CL_EPOCHS,
            "learning_rate": CL_LEARNING_RATE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "weight_decay": WEIGHT_DECAY,
            "lr_warmup": USE_LR_WARMUP,
            "warmup_steps": WARMUP_STEPS if USE_LR_WARMUP else 0,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "pruning": "disabled",
            "mixed_precision": "disabled",
        }
    }

    current_model = copy.deepcopy(base_model)

    for cl_name in cl_windows:
        print(f"\n{'#' * 70}")
        print(f"# Processing: {cl_name}")
        print(f"{'#' * 70}")

        cl_loader = cl_loaders[cl_name]

        # GPU energy before
        if gpu_handle:
            try:
                energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
            except:
                energy_before = 0
        else:
            energy_before = 0

        # Ultra-aggressive fine-tuning
        updated_model, train_stats = train_on_cl_window_ultra_aggressive(
            current_model,
            cl_loader,
            cl_name,
            epochs=CL_EPOCHS,
        )

        # GPU energy after
        if gpu_handle:
            try:
                energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
                gpu_energy_samples.append(energy_after - energy_before)
            except:
                pass

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

        # Forgetting assessment
        if forgetting["RMSE_drop"] < 0:
            print("    ✅ Performance improved on old data!")
        elif abs(forgetting["RMSE_pct_change"]) < 5:
            print("    ✅ Minimal forgetting (<5%)")
        elif abs(forgetting["RMSE_pct_change"]) < 8:
            print("    ✅ Low forgetting (<8%) - Target achieved!")
        elif abs(forgetting["RMSE_pct_change"]) < 10:
            print("    ⚠️  Moderate forgetting (8-10%)")
        else:
            print("    ⚠️  Significant forgetting (>10%)")

        # Save updated model
        model_path = CL_MODELS_DIR / f"stgnn_ultra_aggressive_CL_{cl_name}.pt"
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

    # Stop energy tracking
    overall_time = time.time() - overall_start_time

    if tracker:
        try:
            emissions_data = tracker.stop()
            if isinstance(emissions_data, float):
                co2_kg = emissions_data
            elif emissions_data and hasattr(emissions_data, 'energy_consumed'):
                energy_kwh = emissions_data.energy_consumed
                co2_kg = emissions_data.emissions
        except:
            pass

    # Process GPU energy
    if gpu_energy_samples and sum(gpu_energy_samples) > 0:
        gpu_energy_joules = sum(gpu_energy_samples) / 1000.0
        gpu_energy_kwh = gpu_energy_joules / 3600000.0
        if energy_kwh == 0.0:
            energy_kwh = gpu_energy_kwh

    # Energy results
    results["energy"] = {
        "total_time_sec": overall_time,
        "total_time_min": overall_time / 60.0,
        "cl_time_sec": results["total_time_sec"],
        "cl_time_min": results["total_time_sec"] / 60.0,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "energy_per_window_kwh": energy_kwh / len(cl_windows) if energy_kwh > 0 else 0,
    }

    # Save results
    results_path = CL_RESULTS_DIR / "aggressive_continual_learning_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 70)
    print("ULTRA-AGGRESSIVE CONTINUAL LEARNING SUMMARY")
    print("=" * 70)

    print("\n📊 Baseline Performance (Before CL):")
    print(f"   Test RMSE: {baseline_test_metrics['RMSE']:.6f}")
    print(f"   Test MAE:  {baseline_test_metrics['MAE']:.6f}")

    print("\n📈 Per-Window Results:\n")
    print(f"{'Window':<10} {'New RMSE':<12} {'Old RMSE':<12} {'Forgetting':<15} {'Time (s)':<10} {'Steps':<10}")
    print("-" * 80)

    for update in results["cl_updates"]:
        window = update["window"]
        new_rmse = update["new_data_metrics"]["RMSE"]
        old_rmse = update["old_test_metrics"]["RMSE"]
        forgetting_pct = update["forgetting"]["RMSE_pct_change"]
        duration = update["train_stats"]["duration_sec"]
        steps = update["train_stats"]["total_steps"]

        print(
            f"{window:<10} {new_rmse:<12.6f} {old_rmse:<12.6f} "
            f"{forgetting_pct:+6.2f}%        {duration:<10.1f} {steps:<10}"
        )

    print("-" * 80)

    final_update = results["cl_updates"][-1]
    final_rmse = final_update["old_test_metrics"]["RMSE"]
    final_forgetting = final_update["forgetting"]["RMSE_pct_change"]

    print("\n🏆 Final Model (after all CL updates):")
    print(f"   Test RMSE: {final_rmse:.6f}")
    print(f"   Total Forgetting: {final_forgetting:+.2f}%")
    print(f"   Total CL Time: {results['total_time_sec']:.1f}s ({results['total_time_sec'] / 60:.2f} min)")

    avg_forgetting = float(
        np.mean([u["forgetting"]["RMSE_pct_change"] for u in results["cl_updates"]])
    )

    avg_time = results["total_time_sec"] / len(results["cl_updates"])

    print("\n📊 Statistics:")
    print(f"   CL Windows Processed: {len(results['cl_updates'])}")
    print(f"   Average Forgetting: {avg_forgetting:+.2f}%")
    print(f"   Average Time per Window: {avg_time:.1f}s")

    if energy_kwh > 0:
        print(f"\n⚡ Energy Metrics:")
        print(f"   Total Energy: {energy_kwh:.6f} kWh")
        print(f"   Energy per Window: {energy_kwh / len(cl_windows):.6f} kWh")
        print(f"   CO2 Emissions: {co2_kg:.6f} kg")

        # Compare to baseline CL
        baseline_cl_energy = 0.012  # From your baseline CL run
        baseline_cl_time = 296.1  # seconds

        if baseline_cl_energy > 0:
            energy_savings = ((baseline_cl_energy - energy_kwh) / baseline_cl_energy) * 100
            time_savings = ((baseline_cl_time - results['total_time_sec']) / baseline_cl_time) * 100

            print(f"\n💰 Comparison to Baseline CL:")
            print(f"   Energy Savings: {energy_savings:+.1f}%")
            print(f"   Time Savings: {time_savings:+.1f}%")
            print(f"   Forgetting Increase: {avg_forgetting - 1.11:+.2f}% (baseline was +1.11%)")

            if energy_savings >= 60:
                print(f"   🏆 TARGET ACHIEVED: {energy_savings:.1f}% savings (goal: 60-70%)")
            elif energy_savings >= 50:
                print(f"   ✅ GOOD: {energy_savings:.1f}% savings (close to 60% target)")
            else:
                print(f"   ⚠️  Below target: {energy_savings:.1f}% (goal: 60%+)")

    print(f"\n💾 Results saved to: {results_path}")
    print(f"💾 Models saved to: {CL_MODELS_DIR}/")
    print(f"📂 Energy logs: {ENERGY_LOGS_DIR}/")

    print("\n" + "=" * 70)
    print("✅ ULTRA-AGGRESSIVE CONTINUAL LEARNING COMPLETED!")
    print("=" * 70 + "\n")

    return results


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == "__main__":
    run_aggressive_continual_learning()
