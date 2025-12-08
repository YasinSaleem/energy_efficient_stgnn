#!/usr/bin/env python3
"""
STGNN Model Evaluation Script

Comprehensive metrics for trained ST-GNN model:
- MSE, RMSE, MAE, MAPE
- Per-horizon breakdown (6-hour forecast)
- Visualization of predictions vs actuals
- Statistical analysis

Author: Energy-Efficient STGNN Project
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

from data_preprocessing import get_base_dataloaders
from model_stgnn import build_stgnn

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/stgnn_best.pt")
RESULTS_DIR = Path("results/evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(pred, true):
    """
    Compute comprehensive metrics.

    Args:
        pred: [B, H, N] predictions
        true: [B, H, N] ground truth

    Returns:
        dict: All metrics
    """
    # Flatten for overall metrics
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    # Basic metrics
    mse = torch.mean((pred_flat - true_flat) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    mae = torch.mean(torch.abs(pred_flat - true_flat)).item()

    # MAPE (avoid division by zero)
    mape = torch.mean(
        torch.abs((pred_flat - true_flat) / (true_flat + 1e-8))
    ).item() * 100

    # R² score
    ss_res = torch.sum((true_flat - pred_flat) ** 2).item()
    ss_tot = torch.sum((true_flat - torch.mean(true_flat)) ** 2).item()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def compute_per_horizon_metrics(pred, true):
    """
    Compute metrics for each forecast horizon.

    Args:
        pred: [B, H, N] predictions
        true: [B, H, N] ground truth

    Returns:
        dict: Metrics per horizon
    """
    B, H, N = pred.shape

    horizon_metrics = {}

    for h in range(H):
        pred_h = pred[:, h, :].reshape(-1)
        true_h = true[:, h, :].reshape(-1)

        mse = torch.mean((pred_h - true_h) ** 2).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        mae = torch.mean(torch.abs(pred_h - true_h)).item()

        horizon_metrics[f'h{h + 1}'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }

    return horizon_metrics


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_model(model, loader, split_name='test'):
    """
    Evaluate model on a dataset split.

    Args:
        model: Trained STGNN model
        loader: DataLoader
        split_name: Name of split (for logging)

    Returns:
        dict: Comprehensive evaluation results
    """
    model.eval()

    all_preds = []
    all_trues = []

    print(f"\n📊 Evaluating on {split_name} set...")

    for X, Y in tqdm(loader, desc=f"Evaluating {split_name}"):
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)

        preds = model(X)

        all_preds.append(preds.cpu())
        all_trues.append(Y.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)  # [Total, H, N]
    all_trues = torch.cat(all_trues, dim=0)  # [Total, H, N]

    # Overall metrics
    print(f"\n{'=' * 60}")
    print(f"OVERALL METRICS ({split_name.upper()})")
    print(f"{'=' * 60}")

    overall_metrics = compute_metrics(all_preds, all_trues)
    for metric, value in overall_metrics.items():
        print(f"{metric:10s}: {value:.6f}")

    # Per-horizon metrics
    print(f"\n{'=' * 60}")
    print(f"PER-HORIZON METRICS ({split_name.upper()})")
    print(f"{'=' * 60}")

    horizon_metrics = compute_per_horizon_metrics(all_preds, all_trues)

    print(f"{'Horizon':<10} {'RMSE':<12} {'MAE':<12}")
    print(f"{'-' * 34}")
    for h_name, metrics in horizon_metrics.items():
        print(f"{h_name:<10} {metrics['RMSE']:<12.6f} {metrics['MAE']:<12.6f}")

    # Statistical summary
    print(f"\n{'=' * 60}")
    print(f"PREDICTION STATISTICS ({split_name.upper()})")
    print(f"{'=' * 60}")

    pred_mean = all_preds.mean().item()
    pred_std = all_preds.std().item()
    pred_min = all_preds.min().item()
    pred_max = all_preds.max().item()

    true_mean = all_trues.mean().item()
    true_std = all_trues.std().item()
    true_min = all_trues.min().item()
    true_max = all_trues.max().item()

    print(f"{'Predictions:':<15} mean={pred_mean:.4f}, std={pred_std:.4f}, "
          f"min={pred_min:.4f}, max={pred_max:.4f}")
    print(f"{'Ground Truth:':<15} mean={true_mean:.4f}, std={true_std:.4f}, "
          f"min={true_min:.4f}, max={true_max:.4f}")

    # Save results
    results = {
        'split': split_name,
        'overall_metrics': overall_metrics,
        'horizon_metrics': horizon_metrics,
        'statistics': {
            'predictions': {
                'mean': pred_mean,
                'std': pred_std,
                'min': pred_min,
                'max': pred_max
            },
            'ground_truth': {
                'mean': true_mean,
                'std': true_std,
                'min': true_min,
                'max': true_max
            }
        }
    }

    return results, all_preds, all_trues


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_predictions(preds, trues, split_name, num_samples=5, num_nodes=3):
    """
    Plot sample predictions vs ground truth.

    Args:
        preds: [B, H, N] predictions
        trues: [B, H, N] ground truth
        split_name: Name of split
        num_samples: Number of time samples to plot
        num_nodes: Number of nodes to plot
    """
    fig, axes = plt.subplots(num_nodes, num_samples,
                             figsize=(num_samples * 3, num_nodes * 2))

    if num_nodes == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    B, H, N = preds.shape

    # Select random samples and nodes
    sample_indices = np.random.choice(B, num_samples, replace=False)
    node_indices = np.random.choice(N, num_nodes, replace=False)

    for i, node_idx in enumerate(node_indices):
        for j, sample_idx in enumerate(sample_indices):
            ax = axes[i, j]

            pred = preds[sample_idx, :, node_idx].numpy()
            true = trues[sample_idx, :, node_idx].numpy()

            hours = np.arange(1, H + 1)

            ax.plot(hours, true, 'o-', label='True', color='blue', linewidth=2)
            ax.plot(hours, pred, 's--', label='Pred', color='red', linewidth=2)

            ax.set_xlabel('Horizon (hours)')
            ax.set_ylabel('Normalized kWh')
            ax.set_title(f'Node {node_idx}, Sample {sample_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / f"{split_name}_predictions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved plot: {plot_path}")
    plt.close()


def plot_error_distribution(preds, trues, split_name):
    """
    Plot error distribution histogram.
    """
    errors = (preds - trues).reshape(-1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(errors, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Error Distribution ({split_name})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot (quantile-quantile)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot ({split_name})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / f"{split_name}_error_dist.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved plot: {plot_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("STGNN MODEL EVALUATION")
    print("=" * 60)

    # Check model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Load data
    print("\n📦 Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_base_dataloaders()

    # Load model
    print(f"\n🧠 Loading model from {MODEL_PATH}...")
    model = build_stgnn().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Evaluate on all splits
    all_results = {}

    # Test set (primary)
    test_results, test_preds, test_trues = evaluate_model(
        model, test_loader, 'test'
    )
    all_results['test'] = test_results

    # Validation set
    val_results, val_preds, val_trues = evaluate_model(
        model, val_loader, 'validation'
    )
    all_results['validation'] = val_results

    # Visualizations
    print("\n📊 Generating visualizations...")
    plot_predictions(test_preds, test_trues, 'test')
    plot_error_distribution(test_preds, test_trues, 'test')

    # Save all results
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Saved results: {results_path}")

    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETED!")
    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print()


if __name__ == "__main__":
    main()
