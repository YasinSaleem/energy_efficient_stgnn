#!/usr/bin/env python3
"""
Comprehensive Energy Optimization Test Suite
Tests individual optimizations and all combinations systematically
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd

# Suppress codecarbon warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['CODECARBON_VERBOSE'] = '0'

# CodeCarbon for energy tracking
try:
    from codecarbon import EmissionsTracker

    CODECARBON_AVAILABLE = True
except ImportError:
    print("⚠️  CodeCarbon not installed. Energy tracking will be disabled.")
    CODECARBON_AVAILABLE = False

# Try to import pynvml for better GPU tracking
try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except:
    PYNVML_AVAILABLE = False

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import config as cfg
from src.data_preprocessing import get_base_dataloaders
from src.model_stgnn import build_stgnn


# ============================================================================
# OPTIMIZATION TECHNIQUES
# ============================================================================

class OptimizationMethod:
    """Base class for optimization methods"""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.enabled = False

    def setup(self, model, optimizer, device):
        """Setup method-specific configurations"""
        pass

    def before_epoch(self, epoch):
        """Called before each epoch"""
        pass

    def training_step(self, model, X, Y, optimizer, loss_fn):
        """Execute one training step with optimization"""
        pass

    def after_epoch(self, epoch, metrics):
        """Called after each epoch"""
        pass


# ----------------------------------------------------------------------------
# Method 1: Mixed Precision Training (AMP) - DISABLED FOR SPARSE OPS
# ----------------------------------------------------------------------------

class MixedPrecisionAMP(OptimizationMethod):
    def __init__(self):
        super().__init__(
            name="mixed_precision",
            description="Automatic Mixed Precision (FP16) training - DISABLED (sparse ops incompatible)"
        )
        self.scaler = None

    def setup(self, model, optimizer, device):
        # DISABLED: Sparse matrix operations don't support FP16
        print(f"  ⚠️  {self.name}: DISABLED - Sparse operations don't support FP16")
        self.enabled = False


# ----------------------------------------------------------------------------
# Method 2: Increased Batch Size with Gradient Accumulation
# ----------------------------------------------------------------------------

class IncreasedBatchSize(OptimizationMethod):
    def __init__(self, batch_size=8, accumulation_steps=4):
        super().__init__(
            name="increased_batch",
            description=f"Batch size {batch_size} with {accumulation_steps} accumulation steps"
        )
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def setup(self, model, optimizer, device):
        print(f"  ✅ {self.name}: Batch={self.batch_size}, Accum={self.accumulation_steps}")
        print(f"     Effective batch size: {self.batch_size * self.accumulation_steps}")

    def before_epoch(self, epoch):
        self.current_step = 0

    def training_step(self, model, X, Y, optimizer, loss_fn):
        if not self.enabled:
            return None

        if self.current_step == 0:
            optimizer.zero_grad()

        preds = model(X)
        loss = loss_fn(preds, Y) / self.accumulation_steps
        loss.backward()

        self.current_step += 1
        grad_norm = 0.0

        if self.current_step == self.accumulation_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.GRADIENT_CLIP_NORM
            )
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            optimizer.step()
            self.current_step = 0

        return loss.item() * self.accumulation_steps, grad_norm


# ----------------------------------------------------------------------------
# Method 3: Enhanced Early Stopping
# ----------------------------------------------------------------------------

class EnhancedEarlyStopping(OptimizationMethod):
    def __init__(self, patience=3, min_delta=1e-4, warmup=5):
        super().__init__(
            name="enhanced_early_stop",
            description=f"Early stopping (patience={patience}, min_delta={min_delta}, warmup={warmup})"
        )
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def setup(self, model, optimizer, device):
        print(f"  ✅ {self.name}: Patience={self.patience}, MinDelta={self.min_delta}")
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def after_epoch(self, epoch, metrics):
        if not self.enabled:
            return False

        val_loss = metrics.get('val_rmse', float('inf'))

        if epoch < self.warmup:
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"  🛑 {self.name}: Triggered at epoch {epoch}")
            self.should_stop = True
            return True

        return False


# ----------------------------------------------------------------------------
# Method 4: Dynamic Quantization (Post-Training)
# ----------------------------------------------------------------------------

class DynamicQuantization(OptimizationMethod):
    def __init__(self):
        super().__init__(
            name="dynamic_quantization",
            description="Dynamic INT8 quantization for Linear and GRU layers"
        )
        self.quantized_model = None

    def setup(self, model, optimizer, device):
        print(f"  ✅ {self.name}: Will quantize after training")

    def quantize_model(self, model):
        """Apply dynamic quantization to trained model"""
        try:
            quantized = torch.quantization.quantize_dynamic(
                model.cpu(),
                {nn.Linear, nn.GRU},
                dtype=torch.qint8
            )
            print(f"  ✅ {self.name}: Model quantized to INT8")
            return quantized
        except Exception as e:
            print(f"  ⚠️  {self.name}: Quantization failed - {e}")
            return model


# ----------------------------------------------------------------------------
# Method 5: Structured Pruning
# ----------------------------------------------------------------------------

class StructuredPruning(OptimizationMethod):
    def __init__(self, amount=0.3):
        super().__init__(
            name="structured_pruning",
            description=f"L1 unstructured pruning ({amount * 100:.0f}% of weights)"
        )
        self.amount = amount

    def setup(self, model, optimizer, device):
        print(f"  ✅ {self.name}: Will prune {self.amount * 100:.0f}% of weights")

    def prune_model(self, model):
        """Apply pruning to trained model"""
        try:
            import torch.nn.utils.prune as prune

            params_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    params_to_prune.append((module, 'weight'))

            for module, param_name in params_to_prune:
                prune.l1_unstructured(module, name=param_name, amount=self.amount)
                prune.remove(module, param_name)

            print(f"  ✅ {self.name}: Pruned {len(params_to_prune)} layers")
            return model
        except Exception as e:
            print(f"  ⚠️  {self.name}: Pruning failed - {e}")
            return model


# ----------------------------------------------------------------------------
# Method 6: Reduced Model Complexity
# ----------------------------------------------------------------------------

class ReducedModelComplexity(OptimizationMethod):
    def __init__(self, hidden_reduction=0.5):
        super().__init__(
            name="reduced_complexity",
            description=f"Reduce hidden dimensions by {hidden_reduction * 100:.0f}%"
        )
        self.hidden_reduction = hidden_reduction

    def setup(self, model, optimizer, device):
        print(f"  ✅ {self.name}: Model will be rebuilt with smaller dims")
        # This would require rebuilding the model - handled in experiment runner


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_device():
    """Select best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch_baseline(model, loader, optimizer, loss_fn, device):
    """Baseline training without optimizations"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X, Y in tqdm(loader, desc="Training", leave=False):
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, Y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def train_epoch_optimized(model, loader, optimizer, loss_fn, device, methods):
    """Training with active optimization methods"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Check which methods are active
    use_batch = any(m.name == "increased_batch" and m.enabled for m in methods)

    for X, Y in tqdm(loader, desc="Training", leave=False):
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        # Try optimized training steps
        loss_val = None

        if use_batch:
            for method in methods:
                if method.name == "increased_batch" and method.enabled:
                    result = method.training_step(model, X, Y, optimizer, loss_fn)
                    if result:
                        loss_val = result[0]
                        break

        # Fallback to baseline
        if loss_val is None:
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
            optimizer.step()
            loss_val = loss.item()

        total_loss += loss_val
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on validation/test set"""
    model.eval()

    total_mse = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    num_batches = 0

    for X, Y in tqdm(loader, desc="Evaluating", leave=False):
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        preds = model(X)

        mse = torch.nn.functional.mse_loss(preds, Y)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(preds - Y))

        total_mse += mse.item()
        total_rmse += rmse.item()
        total_mae += mae.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_mse / num_batches,
        total_rmse / num_batches,
        total_mae / num_batches
    )


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(method_combo, epochs, train_loader, val_loader, test_loader, device, results_dir):
    """Run single experiment with given method combination"""

    combo_name = "+".join([m.name for m in method_combo]) if method_combo else "baseline"
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT: {combo_name}")
    print(f"{'=' * 80}")

    # Initialize model and optimizer
    model = build_stgnn().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    loss_fn = nn.SmoothL1Loss(beta=cfg.HUBER_DELTA)

    # Setup optimization methods
    for method in method_combo:
        method.enabled = True
        method.setup(model, optimizer, device)

    # Start energy tracking with multiple methods
    energy_kwh = 0.0
    co2_kg = 0.0
    gpu_energy_joules = 0.0

    # Method 1: CodeCarbon
    if CODECARBON_AVAILABLE:
        try:
            tracker = EmissionsTracker(
                project_name=f"STGNN_{combo_name}",
                output_dir=str(results_dir / "energy_logs"),
                measure_power_secs=1,
                save_to_file=True,
                log_level="error"
            )
            tracker.start()
        except Exception as e:
            print(f"  ⚠️  Energy tracking failed to start: {e}")
            tracker = None
    else:
        tracker = None

    # Method 2: NVML for GPU
    gpu_handle = None
    if PYNVML_AVAILABLE and device.type == 'cuda':
        try:
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print(f"  ✅ NVML GPU tracking enabled")
        except:
            pass

    start_time = time.time()
    energy_samples = []

    # Training loop
    best_val_rmse = float('inf')
    history = []

    for epoch in range(1, epochs + 1):
        # Before epoch hooks
        for method in method_combo:
            if method.enabled:
                method.before_epoch(epoch)

        # Sample GPU energy before training
        if gpu_handle is not None:
            try:
                energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
            except:
                energy_before = 0
        else:
            energy_before = 0

        # Training
        if method_combo:
            train_loss = train_epoch_optimized(model, train_loader, optimizer, loss_fn, device, method_combo)
        else:
            train_loss = train_epoch_baseline(model, train_loader, optimizer, loss_fn, device)

        # Sample GPU energy after training
        if gpu_handle is not None:
            try:
                energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
                energy_samples.append(energy_after - energy_before)
            except:
                pass

        # Validation
        val_mse, val_rmse, val_mae = evaluate(model, val_loader, device)

        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae
        }
        history.append(metrics)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val RMSE: {val_rmse:.6f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

        # After epoch hooks
        should_stop = False
        for method in method_combo:
            if method.enabled:
                if method.after_epoch(epoch, metrics):
                    should_stop = True

        if should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Post-training optimizations
    for method in method_combo:
        if method.name == "dynamic_quantization" and method.enabled:
            model = method.quantize_model(model)
            model = model.to(device)
        elif method.name == "structured_pruning" and method.enabled:
            model = method.prune_model(model)

    # Final test evaluation
    test_mse, test_rmse, test_mae = evaluate(model, test_loader, device)

    # Stop energy tracking
    training_time = time.time() - start_time

    # Process CodeCarbon results
    if tracker is not None:
        try:
            emissions_data = tracker.stop()

            # Handle different return types from CodeCarbon
            if isinstance(emissions_data, float):
                co2_kg = emissions_data
            elif emissions_data is None:
                pass
            elif hasattr(emissions_data, 'energy_consumed'):
                energy_kwh = emissions_data.energy_consumed
                co2_kg = emissions_data.emissions
            else:
                energy_kwh = getattr(emissions_data, 'energy_consumed', 0.0)
                co2_kg = getattr(emissions_data, 'emissions', 0.0)
        except Exception as e:
            print(f"  ⚠️  Error stopping tracker: {e}")

    # Process NVML GPU energy
    if energy_samples and sum(energy_samples) > 0:
        gpu_energy_joules = sum(energy_samples) / 1000.0  # Convert mJ to J
        gpu_energy_kwh = gpu_energy_joules / 3600000.0  # Convert J to kWh
        print(f"  ✅ GPU energy (NVML): {gpu_energy_kwh:.6f} kWh")

        # Use GPU measurement if CodeCarbon failed
        if energy_kwh == 0.0:
            energy_kwh = gpu_energy_kwh

    # Fallback energy estimation if both failed
    if energy_kwh == 0.0 and co2_kg > 0.0:
        # Estimate from CO2 using average grid intensity
        energy_kwh = co2_kg / 0.475
        print(f"  ℹ️  Energy estimated from CO2: {energy_kwh:.6f} kWh")
    elif energy_kwh == 0.0 and device.type == 'cuda':
        # Final fallback: time-based estimation
        # i7-13650HX typical power: ~45W base, GPU ~150W under load
        estimated_watts = 195.0  # CPU + GPU
        energy_kwh = (estimated_watts * training_time) / 3600000.0
        print(f"  ℹ️  Energy estimated from time: {energy_kwh:.6f} kWh ({training_time:.1f}s @ {estimated_watts}W)")

    # Estimate CO2 if not available
    if co2_kg == 0.0 and energy_kwh > 0.0:
        co2_kg = energy_kwh * 0.475  # US average grid intensity
        print(f"  ℹ️  CO2 estimated from energy: {co2_kg:.6f} kg")

    # Calculate model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Compile results
    results = {
        'experiment': combo_name,
        'methods': [m.name for m in method_combo],
        'num_methods': len(method_combo),
        'epochs_run': len(history),
        'best_val_rmse': best_val_rmse,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'energy_kwh': energy_kwh,
        'co2_kg': co2_kg,
        'training_time_sec': training_time,
        'model_size_mb': model_size_mb,
        'param_count': param_count,
        'history': history
    }

    print(f"\n✅ Experiment Complete: {combo_name}")
    print(f"   Test RMSE: {test_rmse:.6f}")
    print(f"   Energy: {energy_kwh:.6f} kWh")
    print(f"   CO2: {co2_kg:.6f} kg")
    print(f"   Time: {training_time:.2f} sec")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("ENERGY OPTIMIZATION TEST SUITE")
    print("=" * 80)

    # Setup
    device = get_device()
    print(f"\n📱 Device: {device}")

    # Check energy tracking capabilities
    print(f"\n🔌 Energy Tracking Methods:")
    print(f"   CodeCarbon: {'✅ Available' if CODECARBON_AVAILABLE else '❌ Not installed'}")
    print(f"   NVML (GPU): {'✅ Available' if PYNVML_AVAILABLE else '❌ Not available'}")
    if not CODECARBON_AVAILABLE and not PYNVML_AVAILABLE:
        print(f"   Fallback: ✅ Time-based estimation (~195W for i7-13650HX + GPU)")
    print(f"\n   ℹ️  Note: Install Intel Power Gadget for accurate CPU tracking on Windows")

    results_dir = Path("results/energy_optimization")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "energy_logs").mkdir(exist_ok=True)

    # Load data once
    print("\n📦 Loading data loaders...")
    train_loader, val_loader, test_loader = get_base_dataloaders()
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Define optimization methods (REMOVED MIXED PRECISION)
    methods = [
        IncreasedBatchSize(batch_size=8, accumulation_steps=4),
        EnhancedEarlyStopping(patience=3, min_delta=1e-4, warmup=5),
        DynamicQuantization(),
        StructuredPruning(amount=0.3)
    ]

    print(f"\n🔧 Optimization Methods:")
    for i, method in enumerate(methods, 1):
        print(f"   {i}. {method.name}: {method.description}")

    print("\n⚠️  Note: Mixed Precision (AMP) removed - incompatible with sparse operations")

    # Generate all combinations
    all_experiments = [[]]  # Baseline (no methods)

    # Individual methods
    for method in methods:
        all_experiments.append([method])

    # Combinations of 2
    for combo in combinations(methods, 2):
        all_experiments.append(list(combo))

    # Combinations of 3
    for combo in combinations(methods, 3):
        all_experiments.append(list(combo))

    # All 4 methods
    all_experiments.append(methods)

    print(f"\n📊 Total experiments to run: {len(all_experiments)}")

    # Run experiments
    TEST_EPOCHS = 5
    all_results = []

    for i, combo in enumerate(all_experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"RUNNING EXPERIMENT {i}/{len(all_experiments)}")
        print(f"{'=' * 80}")

        try:
            result = run_experiment(
                combo,
                TEST_EPOCHS,
                train_loader,
                val_loader,
                test_loader,
                device,
                results_dir
            )
            all_results.append(result)

            # Save intermediate results
            with open(results_dir / "intermediate_results.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

        except Exception as e:
            print(f"\n❌ Experiment {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate final report
    print("\n" + "=" * 80)
    print("GENERATING FINAL REPORT")
    print("=" * 80)

    if not all_results:
        print("❌ No successful experiments to report!")
        return

    # Create DataFrame
    df = pd.DataFrame([
        {
            'Experiment': r['experiment'],
            'Methods': ', '.join(r['methods']) if r['methods'] else 'None',
            'Num_Methods': r['num_methods'],
            'Epochs': r['epochs_run'],
            'Test_RMSE': r['test_rmse'],
            'Test_MAE': r['test_mae'],
            'Energy_kWh': r['energy_kwh'],
            'CO2_kg': r['co2_kg'],
            'Time_sec': r['training_time_sec'],
            'Model_Size_MB': r['model_size_mb'],
            'Params': r['param_count']
        }
        for r in all_results
    ])

    # Sort by energy consumption
    df = df.sort_values('Energy_kWh')

    # Calculate improvements vs baseline
    baseline_row = df[df['Experiment'] == 'baseline']
    if not baseline_row.empty:
        baseline = baseline_row.iloc[0]
        df['Energy_Reduction_%'] = (
                (baseline['Energy_kWh'] - df['Energy_kWh']) / (baseline['Energy_kWh'] + 1e-10) * 100)
        df['Time_Reduction_%'] = ((baseline['Time_sec'] - df['Time_sec']) / baseline['Time_sec'] * 100)
        df['RMSE_Change_%'] = ((df['Test_RMSE'] - baseline['Test_RMSE']) / baseline['Test_RMSE'] * 100)
    else:
        df['Energy_Reduction_%'] = 0.0
        df['Time_Reduction_%'] = 0.0
        df['RMSE_Change_%'] = 0.0

    # Save results
    df.to_csv(results_dir / "optimization_results.csv", index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS BY TRAINING TIME")
    print("=" * 80)
    df_time = df.sort_values('Time_sec')
    print(
        df_time[['Experiment', 'Time_sec', 'Time_Reduction_%', 'Test_RMSE', 'CO2_kg']].head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("BEST ACCURACY CONFIGURATIONS")
    print("=" * 80)
    df_sorted_acc = df.sort_values('Test_RMSE')
    print(df_sorted_acc[['Experiment', 'Test_RMSE', 'RMSE_Change_%', 'Time_sec']].head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("LOWEST CO2 EMISSIONS")
    print("=" * 80)
    df_co2 = df.sort_values('CO2_kg')
    print(df_co2[['Experiment', 'CO2_kg', 'Test_RMSE', 'Time_sec']].head(10).to_string(index=False))

    print(f"\n✅ Full results saved to: {results_dir / 'optimization_results.csv'}")
    print(f"✅ Energy logs saved to: {results_dir / 'energy_logs'}")

    # Save JSON
    with open(results_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"📊 Total successful experiments: {len(all_results)}/{len(all_experiments)}")


if __name__ == "__main__":
    main()
