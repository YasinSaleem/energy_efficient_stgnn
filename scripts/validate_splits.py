#!/usr/bin/env python3
"""
Split Dataset Validation Script

This script validates the split datasets created for the Energy-Efficient STGNN project.
It performs comprehensive checks on:
- Global validation (folder structure, household consistency)
- Chronological integrity (no backward jumps, no overlaps)
- Date window compliance
- Row count validation
- Column schema consistency

Author: Energy-Efficient STGNN Project
Date: December 2025
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

SPLITS_DIR = "data/splits"
OUTPUT_DIR = "data/validation"

# Split folders
SPLIT_FOLDERS = {
    'train': os.path.join(SPLITS_DIR, "train"),
    'val': os.path.join(SPLITS_DIR, "val"),
    'test': os.path.join(SPLITS_DIR, "test"),
    'cl_1': os.path.join(SPLITS_DIR, "continual", "CL_1"),
    'cl_2': os.path.join(SPLITS_DIR, "continual", "CL_2"),
    'cl_3': os.path.join(SPLITS_DIR, "continual", "CL_3"),
    'cl_4': os.path.join(SPLITS_DIR, "continual", "CL_4"),
}

# Expected date windows
BASE_START = pd.Timestamp("2014-11-02 01:00:00")
BASE_END = pd.Timestamp("2018-12-01 00:00:00")
CL_START = pd.Timestamp("2018-12-01 00:00:00")
CL_END = pd.Timestamp("2020-03-01 00:00:00")

# Minimum row count thresholds
MIN_ROWS = {
    'train': 5000,
    'val': 1000,
    'test': 1000,
    'cl_1': 1000,
    'cl_2': 1000,
    'cl_3': 1000,
    'cl_4': 1000,
}

# Expected row count ratios for base dataset
EXPECTED_RATIOS = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15,
}

# Tolerance for ratio validation
RATIO_TOLERANCE = 0.05

console = Console()


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_folder_structure():
    """
    Validate that all required folders exist and are not empty.
    
    Returns:
        tuple: (bool, list of errors)
    """
    console.print("\n[bold blue]1. Validating Folder Structure[/bold blue]")
    
    errors = []
    
    # Check if base splits directory exists
    if not os.path.exists(SPLITS_DIR):
        errors.append(f"Base splits directory not found: {SPLITS_DIR}")
        return False, errors
    
    # Check each split folder
    for split_name, folder_path in SPLIT_FOLDERS.items():
        if not os.path.exists(folder_path):
            errors.append(f"Missing folder: {split_name} ({folder_path})")
        else:
            # Check if folder is empty
            csv_files = list(Path(folder_path).glob("*.csv"))
            if len(csv_files) == 0:
                errors.append(f"Empty folder: {split_name}")
            else:
                console.print(f"  ✓ {split_name:8s} - {len(csv_files):,} files")
    
    if errors:
        return False, errors
    
    console.print("  [green]✓ All folders exist and contain files[/green]")
    return True, []


def get_household_ids():
    """
    Get household IDs from all split folders and validate consistency.
    
    Returns:
        tuple: (bool, dict of household sets, list of errors)
    """
    console.print("\n[bold blue]2. Validating Household Consistency[/bold blue]")
    
    household_sets = {}
    errors = []
    
    # Collect household IDs from each split
    for split_name, folder_path in SPLIT_FOLDERS.items():
        csv_files = list(Path(folder_path).glob("*.csv"))
        household_ids = {f.stem for f in csv_files}
        household_sets[split_name] = household_ids
        console.print(f"  {split_name:8s} - {len(household_ids):,} households")
    
    # Check if all splits have the same households
    reference_set = household_sets['train']
    
    for split_name, household_set in household_sets.items():
        if split_name == 'train':
            continue
        
        missing = reference_set - household_set
        extra = household_set - reference_set
        
        if missing:
            errors.append(f"{split_name}: Missing {len(missing)} households from train")
        if extra:
            errors.append(f"{split_name}: Has {len(extra)} extra households not in train")
    
    if errors:
        return False, household_sets, errors
    
    console.print(f"  [green]✓ All splits have consistent household IDs ({len(reference_set):,} households)[/green]")
    return True, household_sets, []


def validate_schema(household_id):
    """
    Validate that column schema is identical across all splits for a household.
    
    Args:
        household_id (str): Household identifier
    
    Returns:
        tuple: (bool, dict of column info, list of errors)
    """
    errors = []
    schemas = {}
    
    # Load and check schema for each split
    for split_name, folder_path in SPLIT_FOLDERS.items():
        csv_path = Path(folder_path) / f"{household_id}.csv"
        
        try:
            df = pd.read_csv(csv_path, nrows=1)
            schemas[split_name] = list(df.columns)
        except Exception as e:
            errors.append(f"{split_name}: Failed to read schema - {e}")
            return False, {}, errors
    
    # Compare all schemas to train
    reference_schema = schemas['train']
    
    for split_name, schema in schemas.items():
        if split_name == 'train':
            continue
        
        if schema != reference_schema:
            errors.append(
                f"{split_name}: Schema mismatch. "
                f"Expected {reference_schema}, got {schema}"
            )
    
    if errors:
        return False, schemas, errors
    
    return True, {'schema': reference_schema}, []


def validate_chronological(household_id):
    """
    Validate that timestamps are strictly chronological within each split.
    
    Args:
        household_id (str): Household identifier
    
    Returns:
        tuple: (bool, dict of timestamp info, list of errors)
    """
    errors = []
    timestamp_info = {}
    
    for split_name, folder_path in SPLIT_FOLDERS.items():
        csv_path = Path(folder_path) / f"{household_id}.csv"
        
        try:
            df = pd.read_csv(csv_path)
            
            if 'timestamp' not in df.columns:
                errors.append(f"{split_name}: Missing 'timestamp' column")
                continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Check for duplicates
            duplicates = df['timestamp'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"{split_name}: Found {duplicates} duplicate timestamps")
            
            # Check if chronological
            if not df['timestamp'].is_monotonic_increasing:
                errors.append(f"{split_name}: Timestamps not in chronological order")
            
            # Store first and last timestamp
            if len(df) > 0:
                timestamp_info[split_name] = {
                    'first': df['timestamp'].iloc[0],
                    'last': df['timestamp'].iloc[-1],
                    'count': len(df)
                }
            
        except Exception as e:
            errors.append(f"{split_name}: Failed to validate chronology - {e}")
    
    if errors:
        return False, timestamp_info, errors
    
    return True, timestamp_info, []


def validate_no_overlap(timestamp_info):
    """
    Validate that there is no data leakage between splits.
    
    Args:
        timestamp_info (dict): Dictionary containing first/last timestamps per split
    
    Returns:
        tuple: (bool, list of errors)
    """
    errors = []
    
    # Define the order of splits
    split_order = ['train', 'val', 'test', 'cl_1', 'cl_2', 'cl_3', 'cl_4']
    
    # Check sequential non-overlap
    for i in range(len(split_order) - 1):
        current_split = split_order[i]
        next_split = split_order[i + 1]
        
        if current_split not in timestamp_info or next_split not in timestamp_info:
            continue
        
        current_last = timestamp_info[current_split]['last']
        next_first = timestamp_info[next_split]['first']
        
        if current_last >= next_first:
            errors.append(
                f"Overlap between {current_split} and {next_split}: "
                f"{current_split} ends at {current_last}, "
                f"{next_split} starts at {next_first}"
            )
    
    if errors:
        return False, errors
    
    return True, []


def validate_date_windows(timestamp_info):
    """
    Validate that timestamps fall within expected date windows.
    
    Args:
        timestamp_info (dict): Dictionary containing first/last timestamps per split
    
    Returns:
        tuple: (bool, list of errors)
    """
    errors = []
    
    # Define expected windows for each split
    windows = {
        'train': (BASE_START, BASE_END),
        'val': (BASE_START, BASE_END),
        'test': (BASE_START, BASE_END),
        'cl_1': (CL_START, CL_END),
        'cl_2': (CL_START, CL_END),
        'cl_3': (CL_START, CL_END),
        'cl_4': (CL_START, CL_END),
    }
    
    for split_name, (expected_start, expected_end) in windows.items():
        if split_name not in timestamp_info:
            continue
        
        actual_first = timestamp_info[split_name]['first']
        actual_last = timestamp_info[split_name]['last']
        
        if actual_first < expected_start:
            errors.append(
                f"{split_name}: First timestamp {actual_first} is before "
                f"expected window start {expected_start}"
            )
        
        if actual_last >= expected_end:
            errors.append(
                f"{split_name}: Last timestamp {actual_last} is at or after "
                f"expected window end {expected_end}"
            )
    
    if errors:
        return False, errors
    
    return True, []


def validate_row_counts(timestamp_info):
    """
    Validate row counts meet minimum thresholds and expected ratios.
    
    Args:
        timestamp_info (dict): Dictionary containing row counts per split
    
    Returns:
        tuple: (bool, dict of row count info, list of errors)
    """
    errors = []
    row_counts = {split: info['count'] for split, info in timestamp_info.items()}
    
    # Check minimum thresholds
    for split_name, min_count in MIN_ROWS.items():
        if split_name in row_counts:
            actual_count = row_counts[split_name]
            if actual_count < min_count:
                errors.append(
                    f"{split_name}: Row count {actual_count:,} below "
                    f"minimum threshold {min_count:,}"
                )
    
    # Check base dataset ratios
    base_total = sum(row_counts.get(s, 0) for s in ['train', 'val', 'test'])
    
    if base_total > 0:
        for split_name, expected_ratio in EXPECTED_RATIOS.items():
            if split_name in row_counts:
                actual_ratio = row_counts[split_name] / base_total
                ratio_diff = abs(actual_ratio - expected_ratio)
                
                if ratio_diff > RATIO_TOLERANCE:
                    errors.append(
                        f"{split_name}: Ratio {actual_ratio:.2%} differs from "
                        f"expected {expected_ratio:.2%} by {ratio_diff:.2%}"
                    )
    
    # Check CL window equality
    cl_counts = [row_counts.get(f'cl_{i}', 0) for i in range(1, 5)]
    if cl_counts:
        max_cl = max(cl_counts)
        min_cl = min(cl_counts)
        if max_cl - min_cl > 1:
            errors.append(
                f"CL windows have unequal row counts: {cl_counts}"
            )
    
    if errors:
        return False, row_counts, errors
    
    return True, row_counts, []


def validate_household(household_id):
    """
    Perform all validation checks for a single household.
    
    Args:
        household_id (str): Household identifier
    
    Returns:
        dict: Validation results
    """
    result = {
        'household_id': household_id,
        'passed': True,
        'errors': {
            'schema': [],
            'chronological': [],
            'overlap': [],
            'date_windows': [],
            'row_counts': [],
        },
        'info': {}
    }
    
    # 1. Validate schema
    schema_ok, schema_info, schema_errors = validate_schema(household_id)
    if not schema_ok:
        result['passed'] = False
        result['errors']['schema'] = schema_errors
    result['info']['schema'] = schema_info
    
    # 2. Validate chronological order
    chrono_ok, timestamp_info, chrono_errors = validate_chronological(household_id)
    if not chrono_ok:
        result['passed'] = False
        result['errors']['chronological'] = chrono_errors
    result['info']['timestamps'] = timestamp_info
    
    # Only proceed with further checks if we have timestamp info
    if timestamp_info:
        # 3. Validate no overlap
        overlap_ok, overlap_errors = validate_no_overlap(timestamp_info)
        if not overlap_ok:
            result['passed'] = False
            result['errors']['overlap'] = overlap_errors
        
        # 4. Validate date windows
        window_ok, window_errors = validate_date_windows(timestamp_info)
        if not window_ok:
            result['passed'] = False
            result['errors']['date_windows'] = window_errors
        
        # 5. Validate row counts
        rows_ok, row_counts, row_errors = validate_row_counts(timestamp_info)
        if not rows_ok:
            result['passed'] = False
            result['errors']['row_counts'] = row_errors
        result['info']['row_counts'] = row_counts
    
    return result


# ============================================================================
# REPORTING FUNCTIONS
# ============================================================================

def generate_summary_report(results):
    """
    Generate and display a summary report.
    
    Args:
        results (list): List of validation results
    """
    console.print("\n" + "="*70)
    console.print("[bold cyan]SPLIT VALIDATION SUMMARY[/bold cyan]", justify="center")
    console.print("="*70)
    
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed
    
    # Overall statistics
    table = Table(title="Overall Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Households Processed", f"{total:,}")
    table.add_row("Passed", f"{passed:,}")
    table.add_row("Failed", f"{failed:,}")
    table.add_row("Pass Rate", f"{(passed/total*100) if total > 0 else 0:.2f}%")
    
    console.print(table)
    
    # Failure breakdown
    if failed > 0:
        console.print("\n[bold red]Failures by Type:[/bold red]")
        
        failure_counts = defaultdict(int)
        for result in results:
            if not result['passed']:
                for error_type, errors in result['errors'].items():
                    if errors:
                        failure_counts[error_type] += 1
        
        failure_table = Table()
        failure_table.add_column("Error Type", style="yellow")
        failure_table.add_column("Count", style="red")
        
        for error_type, count in sorted(failure_counts.items()):
            failure_table.add_row(error_type.replace('_', ' ').title(), str(count))
        
        console.print(failure_table)
    
    # Row count statistics (for passed households)
    if passed > 0:
        console.print("\n[bold green]Row Count Statistics (Passed Households):[/bold green]")
        
        row_table = Table()
        row_table.add_column("Split", style="cyan")
        row_table.add_column("Total Rows", style="green")
        row_table.add_column("Avg per Household", style="green")
        
        for split_name in ['train', 'val', 'test', 'cl_1', 'cl_2', 'cl_3', 'cl_4']:
            total_rows = sum(
                r['info'].get('row_counts', {}).get(split_name, 0)
                for r in results if r['passed']
            )
            avg_rows = total_rows / passed if passed > 0 else 0
            row_table.add_row(split_name, f"{total_rows:,}", f"{avg_rows:,.0f}")
        
        console.print(row_table)


def save_detailed_report(results):
    """
    Save detailed validation report to JSON file.
    
    Args:
        results (list): List of validation results
    
    Returns:
        str: Path to saved report
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert timestamps to strings for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        
        # Convert timestamp info
        if 'timestamps' in serializable_result.get('info', {}):
            timestamp_info = {}
            for split, info in serializable_result['info']['timestamps'].items():
                timestamp_info[split] = {
                    'first': info['first'].isoformat() if pd.notna(info['first']) else None,
                    'last': info['last'].isoformat() if pd.notna(info['last']) else None,
                    'count': info['count']
                }
            serializable_result['info']['timestamps'] = timestamp_info
        
        serializable_results.append(serializable_result)
    
    report_path = os.path.join(OUTPUT_DIR, "validation_report.json")
    
    with open(report_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return report_path


def save_failed_households(results):
    """
    Save list of failed households to text file.
    
    Args:
        results (list): List of validation results
    
    Returns:
        str: Path to saved file
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    failed_path = os.path.join(OUTPUT_DIR, "failed_households.txt")
    
    with open(failed_path, 'w') as f:
        f.write("# Failed Households\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        
        for result in results:
            if not result['passed']:
                f.write(f"{result['household_id']}\n")
                
                # Write error details
                for error_type, errors in result['errors'].items():
                    if errors:
                        f.write(f"  [{error_type}]\n")
                        for error in errors:
                            f.write(f"    - {error}\n")
                f.write("\n")
    
    return failed_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    console.print("\n[bold green]SPLIT DATASET VALIDATION[/bold green]", justify="center")
    console.print("="*70 + "\n")
    
    try:
        # Step 1: Validate folder structure
        structure_ok, structure_errors = validate_folder_structure()
        if not structure_ok:
            console.print("[bold red]Folder structure validation failed:[/bold red]")
            for error in structure_errors:
                console.print(f"  ❌ {error}")
            return
        
        # Step 2: Validate household consistency
        consistency_ok, household_sets, consistency_errors = get_household_ids()
        if not consistency_ok:
            console.print("[bold red]Household consistency validation failed:[/bold red]")
            for error in consistency_errors:
                console.print(f"  ❌ {error}")
            return
        
        # Get list of households to validate
        household_ids = sorted(household_sets['train'])
        
        console.print(f"\n[bold blue]3. Validating {len(household_ids):,} Households[/bold blue]")
        
        # Step 3: Validate each household
        results = []
        for household_id in tqdm(household_ids, desc="Validating households"):
            result = validate_household(household_id)
            results.append(result)
        
        # Step 4: Generate summary report
        generate_summary_report(results)
        
        # Step 5: Save detailed reports
        console.print("\n[bold blue]Saving Reports[/bold blue]")
        
        report_path = save_detailed_report(results)
        console.print(f"  ✓ Detailed report: {report_path}")
        
        failed_path = save_failed_households(results)
        console.print(f"  ✓ Failed households: {failed_path}")
        
        console.print("\n" + "="*70)
        console.print("[bold green]✓ VALIDATION COMPLETED[/bold green]", justify="center")
        console.print("="*70 + "\n")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Validation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]❌ ERROR: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    main()
