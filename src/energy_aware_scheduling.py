#!/usr/bin/env python3
"""
Energy-Aware Scheduling Simulation for STGNN Training

This script DOES NOT run actual training.
Instead, it simulates how different scheduling strategies
(when to run training / continual learning jobs over 24h)
affect:

- Total energy consumption (kWh)
- Total "cost" (synthetic price curve)
- Total CO2 emissions (synthetic carbon-intensity curve)

Strategies:
1) immediate      – run jobs as soon as possible
2) night_only     – run only during 22:00–06:00
3) carbon_aware   – place each job in the lowest-emission slot window

Author: Energy-Efficient STGNN Project
"""

from pathlib import Path
import json
import numpy as np

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

BASE_DIR = Path(__file__).resolve().parent           # src/
RESULTS_DIR = BASE_DIR / "results" / "energy_scheduling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SLOT_HOURS = 0.5         # 30 minutes per slot
HORIZON_HOURS = 24.0     # 1 day
NUM_SLOTS = int(HORIZON_HOURS / SLOT_HOURS)


# ============================================================================ #
# GRID PROFILE (Synthetic but structured)
# ============================================================================ #

def build_grid_profile():
    """
    Synthetic 24-hour grid profile with:
    - price_per_kwh       : relative units
    - carbon_kg_per_kwh   : kg CO2 per kWh

    Pattern:
    - 00:00–06:00  : low price, low carbon
    - 06:00–10:00  : mid price, mid carbon
    - 10:00–18:00  : high price, high carbon
    - 18:00–22:00  : mid-high
    - 22:00–24:00  : mid price, lower carbon
    """
    hours = np.arange(0, HORIZON_HOURS, SLOT_HOURS)
    price = np.zeros_like(hours, dtype=np.float32)
    carbon = np.zeros_like(hours, dtype=np.float32)

    for i, h in enumerate(hours):
        if 0 <= h < 6:
            price[i] = 4.0
            carbon[i] = 0.35
        elif 6 <= h < 10:
            price[i] = 6.0
            carbon[i] = 0.45
        elif 10 <= h < 18:
            price[i] = 8.0
            carbon[i] = 0.60
        elif 18 <= h < 22:
            price[i] = 7.0
            carbon[i] = 0.55
        else:  # 22–24
            price[i] = 5.0
            carbon[i] = 0.40

    return {
        "hours": hours,
        "price_per_kwh": price,
        "carbon_kg_per_kwh": carbon,
    }


# ============================================================================ #
# JOB DEFINITIONS
# ============================================================================ #

def define_jobs():
    """
    Define training / CL jobs with approximate durations and power.

    power_kw is average electrical power of the machine (GPU + CPU + others).
    These are illustrative values, not measured.

    Returns: list of job dicts:
        - name
        - duration_hours
        - power_kw
        - slots (computed)
    """
    jobs = [
        {
            "name": "base_training",
            "duration_hours": 3.0,
            "power_kw": 0.30,   # 300 W
        },
        {
            "name": "cl_update_CL1",
            "duration_hours": 1.0,
            "power_kw": 0.25,
        },
        {
            "name": "cl_update_CL2",
            "duration_hours": 1.0,
            "power_kw": 0.25,
        },
        {
            "name": "cl_update_CL3",
            "duration_hours": 1.0,
            "power_kw": 0.25,
        },
        {
            "name": "cl_update_CL4",
            "duration_hours": 1.0,
            "power_kw": 0.25,
        },
    ]

    for job in jobs:
        job["slots"] = int(np.ceil(job["duration_hours"] / SLOT_HOURS))

    return jobs


# ============================================================================ #
# SCHEDULING HELPERS
# ============================================================================ #

def find_earliest_fit(availability, required_slots, allowed_mask=None):
    """
    Find earliest index where 'required_slots' consecutive slots are free
    and (if provided) allowed_mask is True.

    availability: boolean [NUM_SLOTS]
    allowed_mask: boolean [NUM_SLOTS] or None

    Returns: start_slot (int) or None
    """
    N = len(availability)
    for start in range(0, N - required_slots + 1):
        end = start + required_slots
        segment = availability[start:end]
        if not segment.all():
            continue
        if allowed_mask is not None and not allowed_mask[start:end].all():
            continue
        return start
    return None


def compute_job_energy_cost_emissions(job, start_slot, grid_profile):
    """
    Compute total energy, cost, and emissions for a job placed at start_slot.

    Energy per slot = power_kw * SLOT_HOURS
    """
    slots = job["slots"]
    power_kw = job["power_kw"]

    end_slot = start_slot + slots
    price = grid_profile["price_per_kwh"][start_slot:end_slot]
    carbon = grid_profile["carbon_kg_per_kwh"][start_slot:end_slot]

    energy_per_slot = power_kw * SLOT_HOURS
    total_energy = float(energy_per_slot * slots)  # kWh

    total_cost = float(np.sum(price * energy_per_slot))
    total_emissions = float(np.sum(carbon * energy_per_slot))

    return total_energy, total_cost, total_emissions


# ============================================================================ #
# STRATEGIES
# ============================================================================ #

def schedule_immediate(jobs, grid_profile):
    """
    Strategy 1: Immediate scheduling
    - Jobs start as soon as there is enough free time, ignoring price/carbon.
    """
    availability = np.ones(NUM_SLOTS, dtype=bool)
    schedules = {}
    total_energy = total_cost = total_emissions = 0.0

    for job in jobs:
        start_slot = find_earliest_fit(availability, job["slots"])
        if start_slot is None:
            continue

        end_slot = start_slot + job["slots"]
        availability[start_slot:end_slot] = False

        e_kwh, cost, co2 = compute_job_energy_cost_emissions(job, start_slot, grid_profile)

        schedules[job["name"]] = {
            "start_slot": int(start_slot),
            "end_slot": int(end_slot),
            "start_hour": float(start_slot * SLOT_HOURS),
            "end_hour": float(end_slot * SLOT_HOURS),
            "energy_kwh": e_kwh,
            "cost": cost,
            "emissions_kg": co2,
        }

        total_energy += e_kwh
        total_cost += cost
        total_emissions += co2

    return {
        "strategy": "immediate",
        "total_energy_kwh": total_energy,
        "total_cost": total_cost,
        "total_emissions_kg": total_emissions,
        "job_schedules": schedules,
    }


def schedule_night_only(jobs, grid_profile):
    """
    Strategy 2: Night-only scheduling
    - Jobs can only run in [22:00–24:00) U [00:00–06:00).
    """
    availability = np.ones(NUM_SLOTS, dtype=bool)
    hours = grid_profile["hours"]

    allowed_mask = np.zeros(NUM_SLOTS, dtype=bool)
    for i, h in enumerate(hours):
        if (22 <= h < 24) or (0 <= h < 6):
            allowed_mask[i] = True

    schedules = {}
    total_energy = total_cost = total_emissions = 0.0

    for job in jobs:
        start_slot = find_earliest_fit(availability, job["slots"], allowed_mask)
        if start_slot is None:
            continue

        end_slot = start_slot + job["slots"]
        availability[start_slot:end_slot] = False

        e_kwh, cost, co2 = compute_job_energy_cost_emissions(job, start_slot, grid_profile)

        schedules[job["name"]] = {
            "start_slot": int(start_slot),
            "end_slot": int(end_slot),
            "start_hour": float(start_slot * SLOT_HOURS),
            "end_hour": float(end_slot * SLOT_HOURS),
            "energy_kwh": e_kwh,
            "cost": cost,
            "emissions_kg": co2,
        }

        total_energy += e_kwh
        total_cost += cost
        total_emissions += co2

    return {
        "strategy": "night_only",
        "total_energy_kwh": total_energy,
        "total_cost": total_cost,
        "total_emissions_kg": total_emissions,
        "job_schedules": schedules,
    }


def schedule_carbon_aware(jobs, grid_profile):
    """
    Strategy 3: Carbon-aware scheduling
    - For each job, try every feasible start slot and choose the one
      with minimum total emissions for that job.
    - Jobs are scheduled sequentially, so they don't overlap.
    """
    availability = np.ones(NUM_SLOTS, dtype=bool)
    schedules = {}
    total_energy = total_cost = total_emissions = 0.0

    for job in jobs:
        best_start = None
        best_emissions = None

        max_start = NUM_SLOTS - job["slots"]

        for start_slot in range(0, max_start + 1):
            end_slot = start_slot + job["slots"]
            if not availability[start_slot:end_slot].all():
                continue

            _, _, co2 = compute_job_energy_cost_emissions(job, start_slot, grid_profile)

            if best_emissions is None or co2 < best_emissions:
                best_emissions = co2
                best_start = start_slot

        if best_start is None:
            continue

        end_slot = best_start + job["slots"]
        availability[best_start:end_slot] = False

        e_kwh, cost, co2 = compute_job_energy_cost_emissions(job, best_start, grid_profile)

        schedules[job["name"]] = {
            "start_slot": int(best_start),
            "end_slot": int(end_slot),
            "start_hour": float(best_start * SLOT_HOURS),
            "end_hour": float(end_slot * SLOT_HOURS),
            "energy_kwh": e_kwh,
            "cost": cost,
            "emissions_kg": co2,
        }

        total_energy += e_kwh
        total_cost += cost
        total_emissions += co2

    return {
        "strategy": "carbon_aware",
        "total_energy_kwh": total_energy,
        "total_cost": total_cost,
        "total_emissions_kg": total_emissions,
        "job_schedules": schedules,
    }


# ============================================================================ #
# REPORTING / TERMINAL OUTPUT
# ============================================================================ #

def print_strategy_summary(result, baseline=None):
    """
    Print a clean summary of one strategy.
    If baseline is provided, also print relative differences.
    """
    name = result["strategy"]
    E = result["total_energy_kwh"]
    C = result["total_cost"]
    CO2 = result["total_emissions_kg"]

    avg_price = C / E if E > 0 else 0.0
    avg_carbon = CO2 / E if E > 0 else 0.0

    print("\n" + "-" * 70)
    print(f"STRATEGY: {name}")
    print("-" * 70)
    print(f"Total energy       : {E:.3f} kWh")
    print(f"Total cost         : {C:.3f} (relative units)")
    print(f"Total CO2          : {CO2:.3f} kg")
    print(f"Average price      : {avg_price:.3f} per kWh")
    print(f"Average carbon     : {avg_carbon:.3f} kg CO2 / kWh")

    if baseline is not None:
        E0 = baseline["total_energy_kwh"]
        C0 = baseline["total_cost"]
        CO20 = baseline["total_emissions_kg"]

        dE = (E - E0) / E0 * 100 if E0 > 0 else 0.0
        dC = (C - C0) / C0 * 100 if C0 > 0 else 0.0
        dCO2 = (CO2 - CO20) / CO20 * 100 if CO20 > 0 else 0.0

        print("\nRelative to baseline (immediate):")
        print(f"  Energy change    : {dE:+.2f}%")
        print(f"  Cost change      : {dC:+.2f}%")
        print(f"  CO2 change       : {dCO2:+.2f}%")

    print("\nJob schedule (start-end hours):")
    print(f"{'Job':<20} {'Start (h)':>10} {'End (h)':>10} {'Energy (kWh)':>14} {'Cost':>10} {'CO2 (kg)':>10}")
    print("-" * 70)
    for job_name, sched in result["job_schedules"].items():
        print(
            f"{job_name:<20} "
            f"{sched['start_hour']:>10.1f} "
            f"{sched['end_hour']:>10.1f} "
            f"{sched['energy_kwh']:>14.3f} "
            f"{sched['cost']:>10.3f} "
            f"{sched['emissions_kg']:>10.3f}"
        )


def print_overall_comparison(immediate, night, carbon):
    """
    Print a compact comparison table for all strategies.
    """
    strategies = [immediate, night, carbon]

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (ALL STRATEGIES)")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Energy (kWh)':>14} {'Cost':>10} {'CO2 (kg)':>10} "
          f"{'Avg price':>12} {'Avg CO2/kWh':>14}")
    print("-" * 70)

    for r in strategies:
        E = r["total_energy_kwh"]
        C = r["total_cost"]
        CO2 = r["total_emissions_kg"]
        avg_price = C / E if E > 0 else 0.0
        avg_carbon = CO2 / E if E > 0 else 0.0
        print(
            f"{r['strategy']:<15} "
            f"{E:>14.3f} "
            f"{C:>10.3f} "
            f"{CO2:>10.3f} "
            f"{avg_price:>12.3f} "
            f"{avg_carbon:>14.3f}"
        )


def print_recommendation(immediate, night, carbon):
    """
    Print a final recommendation based on minimum total CO2 emissions.
    """
    candidates = [immediate, night, carbon]
    best = min(candidates, key=lambda x: x["total_emissions_kg"])

    print("\n" + "=" * 70)
    print("RECOMMENDED SCHEDULING FOR STGNN TRAINING")
    print("=" * 70)
    print(f"Recommended strategy: {best['strategy']}")
    print(f"Total energy        : {best['total_energy_kwh']:.3f} kWh")
    print(f"Total cost          : {best['total_cost']:.3f}")
    print(f"Total CO2           : {best['total_emissions_kg']:.3f} kg")

    print("\nSuggested execution times:")
    print(f"{'Job':<20} {'Start (h)':>10} {'End (h)':>10}")
    print("-" * 46)
    for job_name, sched in best["job_schedules"].items():
        print(
            f"{job_name:<20} "
            f"{sched['start_hour']:>10.1f} "
            f"{sched['end_hour']:>10.1f}"
        )

    print("\nInterpretation:")
    print(" - These are the hours at which you should ideally schedule:")
    print("     • Base model training")
    print("     • Continual learning updates (CL_1 … CL_4)")
    print(" - The recommendation aims to minimize total CO2 emissions,")
    print("   while keeping total energy and cost comparable.")


# ============================================================================ #
# MAIN
# ============================================================================ #

def main():
    print("\n" + "=" * 70)
    print("ENERGY-AWARE SCHEDULING SIMULATION")
    print("=" * 70)

    grid_profile = build_grid_profile()
    jobs = define_jobs()

    print("\nSimulation settings:")
    print(f"  Time resolution : {SLOT_HOURS} hours/slot")
    print(f"  Horizon         : {HORIZON_HOURS} hours ({NUM_SLOTS} slots)")
    print("\nJobs considered:")
    for j in jobs:
        print(f"  - {j['name']:<15} duration={j['duration_hours']} h, "
              f"power={j['power_kw']:.2f} kW, slots={j['slots']}")

    # Run strategies
    result_immediate = schedule_immediate(jobs, grid_profile)
    result_night = schedule_night_only(jobs, grid_profile)
    result_carbon = schedule_carbon_aware(jobs, grid_profile)

    # Detailed per-strategy summaries
    print_strategy_summary(result_immediate)
    print_strategy_summary(result_night, baseline=result_immediate)
    print_strategy_summary(result_carbon, baseline=result_immediate)

    # Compact comparison table
    print_overall_comparison(result_immediate, result_night, result_carbon)

    # Final recommendation section
    print_recommendation(result_immediate, result_night, result_carbon)

    # Save results to JSON (for plots / paper)
    all_results = {
        "grid_profile": {
            "slot_hours": SLOT_HOURS,
            "horizon_hours": HORIZON_HOURS,
            "hours": grid_profile["hours"].tolist(),
            "price_per_kwh": grid_profile["price_per_kwh"].tolist(),
            "carbon_kg_per_kwh": grid_profile["carbon_kg_per_kwh"].tolist(),
        },
        "jobs": jobs,
        "strategies": {
            "immediate": result_immediate,
            "night_only": result_night,
            "carbon_aware": result_carbon,
        },
    }

    out_path = RESULTS_DIR / "energy_scheduling_results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to:", out_path)
    print("\nSimulation completed.\n")


if __name__ == "__main__":
    main()
