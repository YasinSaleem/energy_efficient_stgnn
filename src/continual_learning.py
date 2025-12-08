#!/usr/bin/env python3
"""
Continual Learning for STGNN (Clean Implementation)

Simple incremental fine-tuning pipeline:
✅ Update model on CL_1, CL_2, CL_3, CL_4 windows
✅ Track performance on new and old data
✅ Measure forgetting
✅ Save updated models

(Energy tracking can be added later as a separate layer)

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

# Paths (all inside src/)
BASE_MODEL_PATH = BASE_DIR / "models" / "stgnn_best.pt"
CL_MODELS_DIR = BASE_DIR / "models" / "continual"
CL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

CL_RESULTS_DIR = BASE_DIR / "results" / "continual_learning"
CL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] BASE_MODEL_PATH = {BASE_MODEL_PATH}")
print(f"[paths] CL_MODELS_DIR   = {CL_MODELS_DIR}")
print(f"[paths] CL_RESULTS_DIR  = {CL_RESULTS_DIR}")

# Continual Learning Settings
CL_EPOCHS = 3            # Number of epochs per CL update
CL_LEARNING_RATE = 5e-4  # Lower than base training (fine-tuning)
WEIGHT_DECAY = 1e-4


# ============================================================================ #
# METRICS
# ============================================================================ #

def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict:
    """
    Compute evaluation metrics on already-batched tensors.

    Args:
        pred: [B, H, N]
        true: [B, H, N]

    Returns:
        dict with MSE, RMSE, MAE, MAPE
    """
    # Flatten across batch, horizon, nodes
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
    """
    Evaluate model on a dataset.

    Args:
        model: STGNN model
        loader: DataLoader yielding (X, Y)

    Returns:
        dict of metrics
    """
    model.eval()
    all_preds = []
    all_trues = []

    for X, Y in loader:
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)
        preds = model(X)
        all_preds.append(preds.cpu())
        all_trues.append(Y.cpu())

    all_preds = torch.cat(all_preds, dim=0)  # [Total, H, N]
    all_trues = torch.cat(all_trues, dim=0)

    return compute_metrics(all_preds, all_trues)


# ============================================================================ #
# CONTINUAL UPDATE
# ============================================================================ #

def train_on_cl_window(
    model: torch.nn.Module,
    cl_loader,
    cl_name: str,
    epochs: int = CL_EPOCHS,
):
    """
    Fine-tune model on a continual learning window.

    Args:
        model: Current model (will be deep-copied)
        cl_loader: DataLoader for CL window
        cl_name: Name of CL window (e.g., "CL_1")
        epochs: Number of fine-tuning epochs

    Returns:
        updated_model, training_stats
    """
    print(f"\n{'=' * 70}")
    print(f"Continual Learning Update: {cl_name}")
    print(f"{'=' * 70}")
    print(f"Fine-tuning for {epochs} epochs with LR={CL_LEARNING_RATE}")

    # Clone model to avoid modifying the input reference
    model = copy.deepcopy(model).to(DEVICE)
    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=CL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    start_time = time.time()
    epoch_losses = []

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

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.6f}")

    duration = time.time() - start_time

    stats = {
        "cl_window": cl_name,
        "epochs": epochs,
        "duration_sec": duration,
        "duration_min": duration / 60,
        "final_loss": epoch_losses[-1],
        "loss_history": epoch_losses,
    }

    print(f"\n⏱️  Time: {duration:.1f}s ({duration / 60:.2f} min)")

    return model, stats


# ============================================================================ #
# FORGETTING MEASUREMENT
# ============================================================================ #

def compute_forgetting(current_metrics: dict, baseline_metrics: dict) -> dict:
    """
    Calculate forgetting (performance drop on old data).

    Args:
        current_metrics: Metrics after CL update
        baseline_metrics: Original metrics before any CL

    Returns:
        dict: Forgetting statistics
    """
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

def run_continual_learning():
    """
    Main continual learning pipeline.

    Process:
    1. Load base model (trained on pre-COVID data)
    2. Evaluate on test set (baseline)
    3. For each CL window (CL_1 → CL_2 → CL_3 → CL_4):
       - Fine-tune on new data
       - Evaluate on new data
       - Evaluate on old test set (measure forgetting)
       - Save updated model
    4. Export results
    """
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------------------- #
    # 1. LOAD DATA
    # --------------------------------------------------------------------- #

    print("\n📦 Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_base_dataloaders()
    cl_loaders = get_cl_dataloaders()

    if not cl_loaders:
        raise ValueError("No CL windows found! Check data/splits/continual/")

    cl_windows = sorted(cl_loaders.keys())
    print(f"  ✓ Found {len(cl_windows)} CL windows: {cl_windows}")

    # --------------------------------------------------------------------- #
    # 2. LOAD BASE MODEL
    # --------------------------------------------------------------------- #

    print(f"\n🧠 Loading base model from {BASE_MODEL_PATH}...")
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")

    base_model = build_stgnn().to(DEVICE)
    base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    print("  ✓ Base model loaded successfully")

    # --------------------------------------------------------------------- #
    # 3. BASELINE EVALUATION (Before any CL)
    # --------------------------------------------------------------------- #

    print("\n📊 Evaluating base model on test set (pre-CL baseline)...")
    baseline_test_metrics = evaluate_model(base_model, test_loader)

    print("\n  BASELINE PERFORMANCE:")
    print(f"    Test RMSE: {baseline_test_metrics['RMSE']:.6f}")
    print(f"    Test MAE:  {baseline_test_metrics['MAE']:.6f}")
    print(f"    Test MAPE: {baseline_test_metrics['MAPE']:.2f}%")

    # --------------------------------------------------------------------- #
    # 4. CONTINUAL LEARNING LOOP
    # --------------------------------------------------------------------- #

    print(f"\n{'=' * 70}")
    print("STARTING CONTINUAL UPDATES")
    print(f"{'=' * 70}")

    results = {
        "baseline_test_metrics": baseline_test_metrics,
        "cl_updates": [],
        "total_time_sec": 0.0,
    }

    current_model = copy.deepcopy(base_model)

    for cl_name in cl_windows:
        print(f"\n{'#' * 70}")
        print(f"# Processing: {cl_name}")
        print(f"{'#' * 70}")

        cl_loader = cl_loaders[cl_name]

        # Fine-tune on CL window
        updated_model, train_stats = train_on_cl_window(
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
        model_path = CL_MODELS_DIR / f"stgnn_{cl_name}.pt"
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

    # --------------------------------------------------------------------- #
    # 5. SAVE RESULTS
    # --------------------------------------------------------------------- #

    results_path = CL_RESULTS_DIR / "continual_learning_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    # --------------------------------------------------------------------- #
    # 6. SUMMARY
    # --------------------------------------------------------------------- #

    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING SUMMARY")
    print("=" * 70)

    print("\n📊 Baseline Performance (Before CL):")
    print(f"   Test RMSE: {baseline_test_metrics['RMSE']:.6f}")
    print(f"   Test MAE:  {baseline_test_metrics['MAE']:.6f}")

    print("\n📈 Per-Window Results:\n")
    print(f"{'Window':<10} {'New RMSE':<12} {'Old RMSE':<12} {'Forgetting':<15} {'Time (s)':<10}")
    print("-" * 70)

    for update in results["cl_updates"]:
        window = update["window"]
        new_rmse = update["new_data_metrics"]["RMSE"]
        old_rmse = update["old_test_metrics"]["RMSE"]
        forgetting_pct = update["forgetting"]["RMSE_pct_change"]
        duration = update["train_stats"]["duration_sec"]

        print(
            f"{window:<10} {new_rmse:<12.6f} {old_rmse:<12.6f} "
            f"{forgetting_pct:+6.2f}%        {duration:<10.1f}"
        )

    print("-" * 70)

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

    print("\n📊 Statistics:")
    print(f"   CL Windows Processed: {len(results['cl_updates'])}")
    print(f"   Average Forgetting: {avg_forgetting:+.2f}%")
    print(
        f"   Average Time per Update: "
        f"{results['total_time_sec'] / len(results['cl_updates']):.1f}s"
    )

    print(f"\n💾 Results saved to: {results_path}")
    print(f"💾 Updated models saved to: {CL_MODELS_DIR}/")

    print("\n" + "=" * 70)
    print("✅ CONTINUAL LEARNING COMPLETED!")
    print("=" * 70 + "\n")

    return results


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == "__main__":
    run_continual_learning()
