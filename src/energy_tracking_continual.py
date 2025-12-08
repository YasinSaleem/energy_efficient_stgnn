#!/usr/bin/env python3
"""
Energy Tracking Script for STGNN Continual Learning

This script:
- Wraps the continual learning pipeline
- Tracks energy consumption & CO₂ using CodeCarbon
- Saves a JSON summary with:
    * Total energy (kWh)
    * Total emissions (kg CO₂)
    * Total runtime (sec/min)
    * CL metrics summary

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

# Import your continual learning pipeline
# Make sure this matches your file name:
#   src/continual_learning.py  ->  from continual_learning import run_continual_learning
from continual_learning import run_continual_learning

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
# MAIN ENERGY TRACKING LOGIC
# ============================================================================ #

def run_continual_learning_with_energy():
    """
    Run the continual learning pipeline with energy tracking.

    Steps:
    1. Start CodeCarbon tracker
    2. Call run_continual_learning() (from continual_learning.py)
    3. Stop tracker and collect energy/emissions
    4. Save everything to JSON
    """
    if not CODECARBON_AVAILABLE:
        raise RuntimeError(
            "CodeCarbon is not installed.\n"
            "Install it with:  pip install codecarbon"
        )

    print("\n" + "=" * 70)
    print("ENERGY TRACKING: CONTINUAL LEARNING")
    print("=" * 70)

    # Set up CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="STGNN_ContinualLearning",
        output_dir=str(ENERGY_LOGS_DIR),
        log_level="error",     # keep it quiet, logs go to file
        save_to_file=True,
    )

    # Start tracking
    print("\n⚡ Starting energy tracking with CodeCarbon...")
    start_time = time.time()
    tracker.start()

    # Run your continual learning experiment (this returns the results dict)
    cl_results = run_continual_learning()

    # Stop tracking
    emissions_data = tracker.stop()
    end_time = time.time()
    duration_sec = end_time - start_time

    # CodeCarbon's tracker returns an Emissions object or float depending on version.
    # We'll be robust and extract what we can.
    energy_kwh = None
    co2_kg = None

    # Newer CodeCarbon versions return an EmissionsData object with attributes
    try:
        # If emissions_data is an EmissionsData object
        energy_kwh = getattr(emissions_data, "energy_consumed", None)
        co2_kg = getattr(emissions_data, "emissions", None)
    except Exception:
        pass

    # If still None, try if it's just a float (older behaviour)
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

    print("\n" + "=" * 70)
    print("ENERGY TRACKING SUMMARY")
    print("=" * 70)
    print(f"⏱  Total runtime: {duration_sec:.1f} s ({duration_sec / 60.0:.2f} min)")
    if energy_kwh is not None:
        print(f"⚡ Total energy: {energy_kwh:.6f} kWh")
    else:
        print("⚡ Total energy: [could not parse from CodeCarbon output]")
    if co2_kg is not None:
        print(f"🌍 Total CO₂:    {co2_kg:.6f} kg")
    else:
        print("🌍 Total CO₂:    [could not parse from CodeCarbon output]")

    # Combine with CL results in one JSON
    combined_results = {
        "energy_summary": energy_summary,
        "continual_learning_results": cl_results,
    }

    output_path = ENERGY_RESULTS_DIR / "cl_energy_results.json"
    with output_path.open("w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n💾 Saved combined energy + CL results to: {output_path}")
    print(f"💾 CodeCarbon logs directory: {ENERGY_LOGS_DIR}")
    print("\n✅ ENERGY TRACKING COMPLETED!\n")

    return combined_results


# ============================================================================ #
# ENTRY POINT
# ============================================================================ #

if __name__ == "__main__":
    run_continual_learning_with_energy()
