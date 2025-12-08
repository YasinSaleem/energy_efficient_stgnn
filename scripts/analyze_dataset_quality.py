#!/usr/bin/env python3
"""
Dataset Quality Analysis Tool

This script analyzes the GoiEner Smart Meter Dataset to compute quality metrics
for each household across different time periods (Pre-COVID, During-COVID, Post-COVID).

For each household, it computes:
- Total rows
- Missing/imputed rows
- First and last timestamp
- Data completeness metrics

Author: Energy-Efficient STGNN Project
Date: December 2025
"""

import os
import tarfile
import pandas as pd
import zstandard as zstd
from pathlib import Path
from tqdm import tqdm
import warnings
from datetime import datetime
from io import BytesIO

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "7362094"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "quality_reports"

# Time windows for each dataset
# Note: Set to None to analyze full dataset without filtering
# Or specify {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'} to filter
TIME_WINDOWS = {
    'imp-pre.tzst': {
        'name': 'Pre-COVID',
        'start': None,  # Will analyze full dataset
        'end': None
    },
    'imp-in.tzst': {
        'name': 'During-COVID',
        'start': None,
        'end': None
    },
    'imp-post.tzst': {
        'name': 'Post-COVID',
        'start': None,
        'end': None
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def list_available_datasets():
    """
    List all available .tzst datasets in the raw data directory.

    Returns:
        list: List of available .tzst files
    """
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: Directory not found: {RAW_DATA_DIR}")
        return []

    tzst_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.tzst')]
    tzst_files.sort()

    return tzst_files


def display_dataset_menu(datasets):
    """
    Display available datasets and let user choose.

    Args:
        datasets (list): List of available dataset files

    Returns:
        str: Selected dataset filename
    """
    print("\n" + "="*70)
    print("  AVAILABLE DATASETS")
    print("="*70)

    for idx, dataset in enumerate(datasets, 1):
        # Get metadata if available
        if dataset in TIME_WINDOWS:
            info = TIME_WINDOWS[dataset]
            print(f"  {idx}. {dataset:20s} - {info['name']:15s} ({info['start']} to {info['end']})")
        else:
            print(f"  {idx}. {dataset}")

    print("="*70)

    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(datasets)}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                return None

            choice_num = int(choice)
            if 1 <= choice_num <= len(datasets):
                selected = datasets[choice_num - 1]
                print(f"\n✓ Selected: {selected}")
                return selected
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number or 'q' to quit")


def get_time_window(dataset_file):
    """
    Get the time window for the selected dataset.

    Args:
        dataset_file (str): Dataset filename

    Returns:
        dict: Dictionary with 'start' and 'end' timestamps, or None for no filtering
    """
    if dataset_file in TIME_WINDOWS:
        window = TIME_WINDOWS[dataset_file]

        # Check if actual dates are specified
        if window.get('start') is not None and window.get('end') is not None:
            return {
                'name': window['name'],
                'start': pd.to_datetime(window['start']),
                'end': pd.to_datetime(window['end'])
            }
        else:
            # Return name but no filtering
            return {
                'name': window['name'],
                'start': None,
                'end': None
            }
    else:
        # No time filtering for raw.tzst or unknown datasets
        return None


def extract_and_analyze_households(dataset_path, time_window):
    """
    Extract and analyze all household CSVs from a compressed archive.

    Args:
        dataset_path (str): Path to the .tzst file
        time_window (dict): Time window with 'start' and 'end' timestamps, or None

    Returns:
        pd.DataFrame: Quality metrics for all households
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING DATASET: {os.path.basename(dataset_path)}")
    print(f"{'='*70}")

    if time_window:
        print(f"Period: {time_window['name']}")
        if time_window['start'] is not None and time_window['end'] is not None:
            print(f"  Start: {time_window['start']}")
            print(f"  End:   {time_window['end']}")
        else:
            print(f"  Analyzing full dataset (no time filtering)")
    else:
        print("Analyzing full dataset (no time filtering)")

    # Storage for household metrics
    household_metrics = []

    print(f"\nOpening compressed archive...")

    with open(dataset_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode='r|') as tar:

                print("Analyzing households...")
                csv_count = 0

                for member in tqdm(tar, desc="Processing"):
                    if member.name.endswith('.csv'):
                        csv_count += 1

                        # Extract user hash from filename
                        user_hash = os.path.basename(member.name).replace('.csv', '')

                        # Extract and read CSV content
                        f = tar.extractfile(member)
                        if f is None:
                            continue

                        try:
                            # Read into BytesIO to make it seekable
                            content = BytesIO(f.read())

                            # Read CSV
                            df = pd.read_csv(content)

                            # Ensure required columns exist
                            if 'timestamp' not in df.columns or 'kWh' not in df.columns:
                                continue

                            # Convert timestamp to datetime
                            df['timestamp'] = pd.to_datetime(df['timestamp'])

                            # Apply time window filter if specified
                            if time_window and time_window.get('start') is not None and time_window.get('end') is not None:
                                df = df[
                                    (df['timestamp'] >= time_window['start']) &
                                    (df['timestamp'] < time_window['end'])
                                ]

                            # Skip if no data in time window
                            if len(df) == 0:
                                continue

                            # Compute metrics
                            total_rows = len(df)

                            # Count imputed rows
                            if 'imputed' in df.columns:
                                imputed_rows = df['imputed'].sum()
                            else:
                                imputed_rows = 0

                            first_timestamp = df['timestamp'].min()
                            last_timestamp = df['timestamp'].max()

                            # Calculate expected rows (hourly data)
                            time_span = last_timestamp - first_timestamp
                            expected_hours = int(time_span.total_seconds() / 3600) + 1
                            completeness = (total_rows / expected_hours * 100) if expected_hours > 0 else 0

                            # Store metrics
                            household_metrics.append({
                                'user_hash': user_hash,
                                'total_rows': total_rows,
                                'imputed_rows': imputed_rows,
                                'imputed_percentage': (imputed_rows / total_rows * 100) if total_rows > 0 else 0,
                                'first_timestamp': first_timestamp,
                                'last_timestamp': last_timestamp,
                                'time_span_days': time_span.days,
                                'expected_hours': expected_hours,
                                'completeness_percentage': completeness
                            })

                        except Exception as e:
                            # Skip problematic files
                            continue

    print(f"\n✓ Analyzed {len(household_metrics):,} households")

    # Convert to DataFrame
    metrics_df = pd.DataFrame(household_metrics)

    return metrics_df


def generate_summary_statistics(metrics_df, dataset_name, time_window):
    """
    Generate and display summary statistics for the dataset.

    Args:
        metrics_df (pd.DataFrame): DataFrame with household metrics
        dataset_name (str): Name of the dataset
        time_window (dict): Time window information
    """
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*70}")

    if len(metrics_df) == 0:
        print("❌ No households found in the specified time window")
        return

    print(f"\nDataset: {dataset_name}")
    if time_window:
        print(f"Period: {time_window['name']}")

    print(f"\nHousehold Count: {len(metrics_df):,}")

    print(f"\nTotal Rows Statistics:")
    print(f"  Min:      {metrics_df['total_rows'].min():,}")
    print(f"  Max:      {metrics_df['total_rows'].max():,}")
    print(f"  Mean:     {metrics_df['total_rows'].mean():,.0f}")
    print(f"  Median:   {metrics_df['total_rows'].median():,.0f}")
    print(f"  Total:    {metrics_df['total_rows'].sum():,}")

    print(f"\nImputed Rows Statistics:")
    print(f"  Total imputed:     {metrics_df['imputed_rows'].sum():,}")
    print(f"  Mean % imputed:    {metrics_df['imputed_percentage'].mean():.2f}%")
    print(f"  Median % imputed:  {metrics_df['imputed_percentage'].median():.2f}%")
    print(f"  Max % imputed:     {metrics_df['imputed_percentage'].max():.2f}%")

    print(f"\nTime Coverage:")
    print(f"  Earliest start: {metrics_df['first_timestamp'].min()}")
    print(f"  Latest end:     {metrics_df['last_timestamp'].max()}")
    print(f"  Mean span:      {metrics_df['time_span_days'].mean():.1f} days")

    print(f"\nData Completeness:")
    print(f"  Mean completeness:   {metrics_df['completeness_percentage'].mean():.2f}%")
    print(f"  Median completeness: {metrics_df['completeness_percentage'].median():.2f}%")

    # Quality tiers
    high_quality = len(metrics_df[metrics_df['completeness_percentage'] >= 95])
    medium_quality = len(metrics_df[(metrics_df['completeness_percentage'] >= 80) &
                                     (metrics_df['completeness_percentage'] < 95)])
    low_quality = len(metrics_df[metrics_df['completeness_percentage'] < 80])

    print(f"\nQuality Tiers:")
    print(f"  High (≥95%):   {high_quality:,} households ({high_quality/len(metrics_df)*100:.1f}%)")
    print(f"  Medium (80-95%): {medium_quality:,} households ({medium_quality/len(metrics_df)*100:.1f}%)")
    print(f"  Low (<80%):    {low_quality:,} households ({low_quality/len(metrics_df)*100:.1f}%)")


def save_metrics(metrics_df, dataset_name, time_window):
    """
    Save metrics to CSV file.

    Args:
        metrics_df (pd.DataFrame): DataFrame with household metrics
        dataset_name (str): Name of the dataset
        time_window (dict): Time window information
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate filename
    base_name = dataset_name.replace('.tzst', '')
    if time_window:
        period_name = time_window['name'].lower().replace('-', '_').replace(' ', '_')
        filename = f"{base_name}_{period_name}_quality_metrics.csv"
    else:
        filename = f"{base_name}_quality_metrics.csv"

    output_path = os.path.join(OUTPUT_DIR, filename)

    # Save to CSV
    metrics_df.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED")
    print(f"{'='*70}")
    print(f"File: {output_path}")
    print(f"Rows: {len(metrics_df):,}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.2f} KB")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("  DATASET QUALITY ANALYSIS TOOL")
    print("="*70)

    try:
        # Step 1: List available datasets
        available_datasets = list_available_datasets()

        if not available_datasets:
            print("\n❌ No .tzst datasets found in", RAW_DATA_DIR)
            return

        # Step 2: Let user select dataset
        selected_dataset = display_dataset_menu(available_datasets)

        if selected_dataset is None:
            print("\n👋 Exiting...")
            return

        # Step 3: Get time window for selected dataset
        time_window = get_time_window(selected_dataset)

        # Step 4: Construct full path
        dataset_path = os.path.join(RAW_DATA_DIR, selected_dataset)

        # Step 5: Extract and analyze households
        metrics_df = extract_and_analyze_households(dataset_path, time_window)

        # Step 6: Generate summary statistics
        generate_summary_statistics(metrics_df, selected_dataset, time_window)

        # Step 7: Save metrics to CSV
        save_metrics(metrics_df, selected_dataset, time_window)

        print(f"\n{'='*70}")
        print(f"✓ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")

    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {str(e)}")
        print(f"{'='*70}")
        raise


if __name__ == "__main__":
    main()
