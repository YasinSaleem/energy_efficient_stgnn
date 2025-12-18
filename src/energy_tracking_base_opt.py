#!/usr/bin/env python3
"""
Energy Tracking Wrapper for Optimized STGNN Training
This script ONLY tracks energy - all training logic is in train_optimized.py
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# CodeCarbon for energy tracking
try:
    from codecarbon import EmissionsTracker

    CODECARBON_AVAILABLE = True
except ImportError:
    print("⚠️  CodeCarbon not installed")
    CODECARBON_AVAILABLE = False

# NVML for GPU tracking
try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except:
    PYNVML_AVAILABLE = False

# Import the actual training script
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.train_optimized import main as train_optimized_main


# ============================================================================
# ENERGY TRACKING ONLY
# ============================================================================

def track_energy_for_training():
    """Wrapper that tracks energy while running train_optimized.py"""

    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    ENERGY_RESULTS_DIR = PROJECT_ROOT / "results" / "energy_tracking"
    ENERGY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ENERGY_LOGS_DIR = ENERGY_RESULTS_DIR / "codecarbon_logs"
    ENERGY_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("ENERGY TRACKING FOR OPTIMIZED TRAINING")
    print("=" * 80)
    print("📊 Training configuration from train_optimized.py")
    print("⚡ This script ONLY tracks energy consumption")
    print("=" * 80 + "\n")

    # Start energy tracking
    energy_kwh = 0.0
    co2_kg = 0.0
    gpu_energy_samples = []

    # CodeCarbon tracker
    if CODECARBON_AVAILABLE:
        try:
            tracker = EmissionsTracker(
                project_name="STGNN_Optimized_Training",
                output_dir=str(ENERGY_LOGS_DIR),
                measure_power_secs=1,
                save_to_file=True,
                log_level="error"
            )
            tracker.start()
            print("⚡ CodeCarbon tracking started\n")
        except Exception as e:
            print(f"⚠️  CodeCarbon failed: {e}\n")
            tracker = None
    else:
        tracker = None

    # GPU tracking
    gpu_handle = None
    if PYNVML_AVAILABLE:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                print(f"✅ NVML GPU tracking enabled\n")
        except:
            pass

    # Track GPU energy before
    if gpu_handle:
        try:
            energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
        except:
            energy_before = 0
    else:
        energy_before = 0

    start_time = time.time()

    # RUN THE ACTUAL TRAINING (from train_optimized.py)
    print("🚀 Starting training from train_optimized.py...\n")
    training_results = train_optimized_main()

    # Track GPU energy after
    if gpu_handle:
        try:
            energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu_handle)
            gpu_energy_joules = (energy_after - energy_before) / 1000.0
            gpu_energy_kwh = gpu_energy_joules / 3600000.0
            print(f"\n✅ GPU energy (NVML): {gpu_energy_kwh:.6f} kWh")
        except:
            gpu_energy_kwh = 0.0
    else:
        gpu_energy_kwh = 0.0

    training_time = time.time() - start_time

    # Stop CodeCarbon
    if tracker:
        try:
            emissions_data = tracker.stop()
            if hasattr(emissions_data, 'energy_consumed'):
                energy_kwh = emissions_data.energy_consumed
                co2_kg = emissions_data.emissions
            print(f"✅ CodeCarbon energy: {energy_kwh:.6f} kWh")
        except:
            pass

    # Use GPU energy if CodeCarbon failed
    if energy_kwh == 0.0 and gpu_energy_kwh > 0.0:
        energy_kwh = gpu_energy_kwh

    # Fallback estimation
    if energy_kwh == 0.0:
        energy_kwh = (195.0 * training_time) / 3600000.0
        print(f"ℹ️  Energy estimated from time: {energy_kwh:.6f} kWh")

    if co2_kg == 0.0 and energy_kwh > 0.0:
        co2_kg = energy_kwh * 0.475

    # Combine training results with energy data
    baseline_test_rmse = 0.713454
    baseline_energy = 0.114038
    baseline_time = 5312.5

    test_rmse = training_results.get('test_rmse', 0.0)
    rmse_change = ((test_rmse - baseline_test_rmse) / baseline_test_rmse) * 100
    energy_reduction = ((baseline_energy - energy_kwh) / baseline_energy) * 100
    time_reduction = ((baseline_time - training_time) / baseline_time) * 100

    # Add energy data to results
    full_results = {
        **training_results,
        'energy_tracking': {
            'total_time_sec': training_time,
            'total_time_min': training_time / 60.0,
            'energy_kwh': energy_kwh,
            'co2_kg': co2_kg,
            'tracking_method': 'NVML' if gpu_energy_kwh > 0 else 'CodeCarbon' if tracker else 'Estimated'
        },
        'comparison_to_baseline': {
            'baseline_test_rmse': baseline_test_rmse,
            'baseline_energy_kwh': baseline_energy,
            'baseline_time_sec': baseline_time,
            'test_rmse_change_pct': rmse_change,
            'energy_reduction_pct': energy_reduction,
            'time_reduction_pct': time_reduction
        }
    }

    # Print summary
    print("\n" + "=" * 80)
    print("ENERGY TRACKING SUMMARY")
    print("=" * 80)
    print(f"Training Time: {training_time:.1f}s ({training_time / 60:.2f} min)")
    print(f"Energy Used: {energy_kwh:.6f} kWh")
    print(f"CO2 Emitted: {co2_kg:.6f} kg")
    print()
    print("📊 Comparison to Baseline:")
    print(f"  Test RMSE: {test_rmse:.6f} vs {baseline_test_rmse:.6f} ({rmse_change:+.2f}%)")
    print(f"  Energy: {energy_kwh:.6f} vs {baseline_energy:.6f} kWh ({energy_reduction:+.1f}%)")
    print(f"  Time: {training_time / 60:.1f} vs {baseline_time / 60:.1f} min ({time_reduction:+.1f}%)")
    print()

    # Success checks
    if abs(rmse_change) < 3.0:
        print("✅ SUCCESS: Test RMSE degradation < 3%!")
    else:
        print(f"⚠️  Test RMSE degradation {abs(rmse_change):.1f}% (target: < 3%)")

    if energy_reduction > 40:
        print("✅ SUCCESS: Energy savings > 40%!")
    else:
        print(f"⚠️  Energy savings {energy_reduction:.1f}% (target: > 40%)")

    print("=" * 80)

    # Save results
    output_path = ENERGY_RESULTS_DIR / "optimized_training_with_energy.json"
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_path}")
    print(f"📂 CodeCarbon logs: {ENERGY_LOGS_DIR}")

    return full_results


if __name__ == "__main__":
    try:
        results = track_energy_for_training()
        print("\n✅ Energy tracking completed!")
    except Exception as e:
        print(f"\n❌ Energy tracking failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
