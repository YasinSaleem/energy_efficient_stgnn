#!/usr/bin/env python3
"""
Energy Tracking Script for STGNN Base Training

This script:
- Automatically runs the base training script (train.py)
- Tracks energy consumption & CO₂ using CodeCarbon
- Saves a JSON summary with:
    * Total energy (kWh)
    * Total emissions (kg CO₂)
    * Total runtime (sec/min)
    * (Optionally) any training metrics returned by train.main()

Author: Energy-Efficient STGNN Project
"""

import time
import json
from pathlib import Path

import torch

# Try to import CodeCarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# Import your base training entrypoint
# Make sure train.py has a `main()` function.
# If your function has a different name, change the import accordingly.
from train import main as run_base_training


# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using: {DEVICE}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENERGY_RESULTS_DIR = PROJECT_ROOT / "results" / "energy_tracking"
ENERGY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENERGY_LOGS_DIR = ENERGY_RESULTS_DIR / "codecarbon_logs"
ENERGY_LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================ #
# MAIN ENERGY TRACKING LOGIC FOR BASE TRAINING
# ============================================================================ #

def run_base_training_with_energy():
    """
    Run the base STGNN training (train.py) with energy tracking.

    Steps:
    1. Start CodeCarbon tracker
    2. Call run_base_training() (from train.py)
    3. Stop tracker and collect energy/emissions
    4. Save everything to JSON
    """
    if not CODECARBON_AVAILABLE:
        raise RuntimeError(
            "CodeCarbon is not installed.\n"
            "Install it with:  pip install codecarbon"
        )

    print("\n" + "=" * 70)
    print("ENERGY TRACKING: BASE TRAINING")
    print("=" * 70)

    # Set up CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="STGNN_BaseTraining",
        output_dir=str(ENERGY_LOGS_DIR),
        log_level="error",   # quiet in console, logs go to file
        save_to_file=True,
    )

    # Start tracking
    print("\nStarting energy tracking with CodeCarbon for base training...")
    start_time = time.time()
    tracker.start()

    # ------------------------------------------------------------------ #
    # Run your base training
    # ------------------------------------------------------------------ #
    # If `train.main()` returns something (e.g., metrics dict), we capture it.
    # If it returns None, that's also fine.
    training_results = run_base_training()

    # Stop tracking
    emissions_data = tracker.stop()
    end_time = time.time()
    duration_sec = end_time - start_time

    # Extract energy & CO₂ from CodeCarbon output
    energy_kwh = None
    co2_kg = None

    try:
        # Newer CodeCarbon: EmissionsData object
        energy_kwh = getattr(emissions_data, "energy_consumed", None)
        co2_kg = getattr(emissions_data, "emissions", None)
    except Exception:
        pass

    # Older behaviour: sometimes a float is returned
    if energy_kwh is None and isinstance(emissions_data, (int, float)):
        energy_kwh = float(emissions_data)

    # Build energy summary
    energy_summary = {
        "device": str(DEVICE),
        "total_runtime_sec": duration_sec,
        "total_runtime_min": duration_sec / 60.0,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "codecarbon_raw": repr(emissions_data),
    }

    # Pretty print summary
    print("\n" + "=" * 70)
    print("BASE TRAINING ENERGY SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {duration_sec:.1f} s ({duration_sec / 60.0:.2f} min)")
    if energy_kwh is not None:
        print(f"Energy used : {energy_kwh:.6f} kWh")
    else:
        print("Energy used : [could not parse from CodeCarbon output]")
    if co2_kg is not None:
        print(f"CO₂ emitted : {co2_kg:.6f} kg")
    else:
        print("CO₂ emitted : [could not parse from CodeCarbon output]")

    # Combine with (optional) training results in one JSON
    combined_results = {
        "energy_summary": energy_summary,
        "training_results": training_results,  # may be None if train.main() returns nothing
    }

    output_path = ENERGY_RESULTS_DIR / "base_training_energy_results.json"
    with output_path.open("w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\nSaved base training energy results to: {output_path}")
    print(f"CodeCarbon logs directory: {ENERGY_LOGS_DIR}")
    print("\nEnergy tracking for base training completed.\n")

    return combined_results


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == "__main__":
    run_base_training_with_energy()
