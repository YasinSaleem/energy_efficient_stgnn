#!/usr/bin/env python3
"""
Energy Tracking for Energy-Optimized Continual Learning

Wraps the energy-optimized continual learning pipeline with CodeCarbon
to measure energy consumption and carbon emissions.

Author: Energy-Efficient STGNN Project
"""

import json
from pathlib import Path
from codecarbon import EmissionsTracker

from energy_optimized_continual_learning import run_energy_optimized_continual_learning


def main():
    """Run energy-optimized CL with energy tracking."""
    
    print("\n" + "=" * 80)
    print("ENERGY-OPTIMIZED CONTINUAL LEARNING WITH ENERGY TRACKING")
    print("=" * 80)
    print("\nThis will measure energy consumption during optimized CL updates.")
    print("Optimizations: Enhanced Early Stopping + Structured Pruning")
    print("=" * 80 + "\n")
    
    # Setup tracking directory
    BASE_DIR = Path(__file__).resolve().parent
    CL_RESULTS_DIR = BASE_DIR / "results" / "continual_learning"
    CL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="STGNN_Energy_Optimized_Continual_Learning",
        output_dir=str(CL_RESULTS_DIR),
        output_file="emissions_energy_optimized_cl.csv",
        log_level="error",
        save_to_file=True,
        tracking_mode="process"
    )
    
    # Start tracking
    print("🔋 Starting energy tracking...\n")
    tracker.start()
    
    try:
        # Run energy-optimized continual learning
        results = run_energy_optimized_continual_learning()
        
        # Stop tracking
        emissions = tracker.stop()
        
        print("\n" + "=" * 80)
        print("ENERGY TRACKING RESULTS")
        print("=" * 80)
        
        if emissions is not None:
            print(f"\n⚡ Energy Consumed: {emissions:.6f} kWh")
            print(f"🌍 CO2 Emissions: {emissions * 1000:.6f} gCO2eq")
            print(f"⏱️  Total Time: {results['total_time_sec']:.1f}s ({results['total_time_sec'] / 60:.2f} min)")
            
            # Add energy metrics to results
            results["energy_metrics"] = {
                "energy_kwh": emissions,
                "co2_grams": emissions * 1000,
                "tracking_mode": "process",
                "project_name": "STGNN_Energy_Optimized_Continual_Learning",
            }
            
            # Save updated results with energy metrics
            results_path = CL_RESULTS_DIR / "energy_optimized_continual_learning_results_with_energy.json"
            with results_path.open("w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results with energy metrics saved to: {results_path}")
        else:
            print("\n⚠️  No energy data captured (tracker returned None)")
        
        print(f"\n📊 Detailed emissions log: {CL_RESULTS_DIR}/emissions_energy_optimized_cl.csv")
        
        print("\n" + "=" * 80)
        print("✅ ENERGY-OPTIMIZED CL WITH TRACKING COMPLETED!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        # Ensure tracker stops even if error occurs
        tracker.stop()
        print(f"\n❌ Error during energy-optimized CL: {e}")
        raise


if __name__ == "__main__":
    main()
