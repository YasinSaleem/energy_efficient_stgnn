#!/usr/bin/env python3
"""
Split Pre-COVID Dataset for Energy-Efficient STGNN

This script splits the GoiEner Smart Meter Dataset (Pre-COVID) into:
- Base dataset: Train (70%), Validation (15%), Test (15%)
- Continual Learning: 4 equal windows (CL_1, CL_2, CL_3, CL_4)

Key constraints:
- Fixed date boundaries: Base (2014-11-02 to 2018-12-01), CL (2018-12-01 to 2020-03-01)
- Row-based splits within each range (not time-based percentages)
- Preserves chronological order
- No data leakage between splits

Author: Energy-Efficient STGNN Project
Date: December 2025
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input directory containing extracted household CSVs
INPUT_DIR = "data/extracted/imp-pre"

# Output directory for splits
OUTPUT_DIR = "data/splits"

# Date boundaries
BASE_START = pd.Timestamp("2014-11-02 01:00:00")
BASE_END = pd.Timestamp("2018-12-01 00:00:00")
CL_START = pd.Timestamp("2018-12-01 00:00:00")
CL_END = pd.Timestamp("2020-03-01 00:00:00")

# Split ratios for base dataset (row-based)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Number of continual learning windows
NUM_CL_WINDOWS = 4

# Minimum rows requirement per household
MIN_ROWS_REQUIRED = 10000


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_directories():
    """Create all necessary output directories."""
    directories = [
        os.path.join(OUTPUT_DIR, "train"),
        os.path.join(OUTPUT_DIR, "val"),
        os.path.join(OUTPUT_DIR, "test"),
    ]
    
    # Add continual learning directories
    for i in range(1, NUM_CL_WINDOWS + 1):
        directories.append(os.path.join(OUTPUT_DIR, "continual", f"CL_{i}"))
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"✓ Created output directories under: {OUTPUT_DIR}")


def validate_chronological_order(df, household_name):
    """
    Validate that timestamps are in chronological order.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        household_name (str): Name of household for error reporting
    
    Raises:
        ValueError: If timestamps are not in chronological order
    """
    if not df['timestamp'].is_monotonic_increasing:
        raise ValueError(f"Household {household_name}: Timestamps are not in chronological order!")


def validate_date_boundaries(df, start, end, split_name, household_name):
    """
    Validate that all rows fall within expected date boundaries.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        start (pd.Timestamp): Expected start timestamp
        end (pd.Timestamp): Expected end timestamp
        split_name (str): Name of the split
        household_name (str): Name of household for error reporting
    
    Raises:
        ValueError: If any rows fall outside the boundaries
    """
    if len(df) == 0:
        return
    
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    
    if min_ts < start or max_ts >= end:
        raise ValueError(
            f"Household {household_name}, Split {split_name}: "
            f"Data leakage detected! Expected [{start}, {end}), "
            f"but got [{min_ts}, {max_ts}]"
        )


def process_household(csv_path):
    """
    Process a single household CSV and create all splits.
    
    Args:
        csv_path (Path): Path to household CSV file
    
    Returns:
        dict: Statistics about the splits created
    """
    household_name = csv_path.stem
    
    try:
        # Load household data
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'kWh']
        if not all(col in df.columns for col in required_cols):
            return {
                'household': household_name,
                'status': 'error',
                'message': f'Missing required columns. Found: {list(df.columns)}'
            }
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate chronological order
        validate_chronological_order(df, household_name)
        
        # Check minimum rows requirement
        if len(df) < MIN_ROWS_REQUIRED:
            raise ValueError(
                f"Insufficient data: {len(df)} rows (minimum required: {MIN_ROWS_REQUIRED})"
            )
        
        # =====================================================================
        # SPLIT 1: BASE DATASET (Train, Val, Test)
        # =====================================================================
        
        # Filter to base period
        base_df = df[(df['timestamp'] >= BASE_START) & (df['timestamp'] < BASE_END)].copy()
        base_df = base_df.reset_index(drop=True)
        
        if len(base_df) == 0:
            return {
                'household': household_name,
                'status': 'skipped',
                'message': 'No data in base period'
            }
        
        # Calculate split indices (row-based)
        total_base_rows = len(base_df)
        train_end_idx = int(total_base_rows * TRAIN_RATIO)
        val_end_idx = int(total_base_rows * (TRAIN_RATIO + VAL_RATIO))
        
        # Create base splits
        train_df = base_df.iloc[:train_end_idx].copy()
        val_df = base_df.iloc[train_end_idx:val_end_idx].copy()
        test_df = base_df.iloc[val_end_idx:].copy()
        
        # Validate boundaries
        validate_date_boundaries(train_df, BASE_START, BASE_END, "Train", household_name)
        validate_date_boundaries(val_df, BASE_START, BASE_END, "Val", household_name)
        validate_date_boundaries(test_df, BASE_START, BASE_END, "Test", household_name)
        
        # =====================================================================
        # SPLIT 2: CONTINUAL LEARNING WINDOWS
        # =====================================================================
        
        # Filter to continual learning period
        cl_df = df[(df['timestamp'] >= CL_START) & (df['timestamp'] < CL_END)].copy()
        cl_df = cl_df.reset_index(drop=True)
        
        cl_windows = []
        if len(cl_df) > 0:
            # Calculate window size (row-based)
            total_cl_rows = len(cl_df)
            window_size = total_cl_rows // NUM_CL_WINDOWS
            
            # Create CL windows
            for i in range(NUM_CL_WINDOWS):
                start_idx = i * window_size
                
                # Last window gets any remaining rows
                if i == NUM_CL_WINDOWS - 1:
                    end_idx = total_cl_rows
                else:
                    end_idx = (i + 1) * window_size
                
                window_df = cl_df.iloc[start_idx:end_idx].copy()
                
                # Validate boundaries
                validate_date_boundaries(window_df, CL_START, CL_END, f"CL_{i+1}", household_name)
                
                cl_windows.append(window_df)
        else:
            # Create empty windows if no CL data
            for i in range(NUM_CL_WINDOWS):
                cl_windows.append(pd.DataFrame(columns=df.columns))
        
        # =====================================================================
        # SAVE ALL SPLITS
        # =====================================================================
        
        # Save base splits
        train_df.to_csv(os.path.join(OUTPUT_DIR, "train", f"{household_name}.csv"), index=False)
        val_df.to_csv(os.path.join(OUTPUT_DIR, "val", f"{household_name}.csv"), index=False)
        test_df.to_csv(os.path.join(OUTPUT_DIR, "test", f"{household_name}.csv"), index=False)
        
        # Save CL windows
        for i, window_df in enumerate(cl_windows, 1):
            window_path = os.path.join(OUTPUT_DIR, "continual", f"CL_{i}", f"{household_name}.csv")
            window_df.to_csv(window_path, index=False)
        
        # Return statistics
        return {
            'household': household_name,
            'status': 'success',
            'total_rows': len(df),
            'base_rows': len(base_df),
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'test_rows': len(test_df),
            'cl_rows': len(cl_df),
            'cl_window_sizes': [len(w) for w in cl_windows],
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        }
        
    except Exception as e:
        return {
            'household': household_name,
            'status': 'error',
            'message': str(e)
        }


def print_summary(results):
    """
    Print summary statistics of the splitting process.
    
    Args:
        results (list): List of result dictionaries from process_household
    """
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    # Count outcomes
    successful = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    print(f"\nProcessing Results:")
    print(f"  Total households:  {len(results):,}")
    print(f"  Successful:        {len(successful):,}")
    print(f"  Errors:            {len(errors):,}")
    print(f"  Skipped:           {len(skipped):,}")
    
    if len(successful) > 0:
        # Calculate statistics for successful households
        total_rows = sum(r['total_rows'] for r in successful)
        base_rows = sum(r['base_rows'] for r in successful)
        train_rows = sum(r['train_rows'] for r in successful)
        val_rows = sum(r['val_rows'] for r in successful)
        test_rows = sum(r['test_rows'] for r in successful)
        cl_rows = sum(r['cl_rows'] for r in successful)
        
        print(f"\nData Distribution:")
        print(f"  Total rows:        {total_rows:,}")
        print(f"  Base period:       {base_rows:,} rows")
        print(f"    - Train:         {train_rows:,} rows ({train_rows/base_rows*100:.1f}%)")
        print(f"    - Validation:    {val_rows:,} rows ({val_rows/base_rows*100:.1f}%)")
        print(f"    - Test:          {test_rows:,} rows ({test_rows/base_rows*100:.1f}%)")
        print(f"  CL period:         {cl_rows:,} rows")
        
        # CL window statistics
        print(f"\nContinual Learning Windows:")
        for i in range(NUM_CL_WINDOWS):
            window_rows = sum(r['cl_window_sizes'][i] for r in successful)
            print(f"    - CL_{i+1}:          {window_rows:,} rows")
        
        # Average per household
        print(f"\nAverage per Household:")
        print(f"  Total rows:        {total_rows/len(successful):,.0f}")
        print(f"  Train rows:        {train_rows/len(successful):,.0f}")
        print(f"  Val rows:          {val_rows/len(successful):,.0f}")
        print(f"  Test rows:         {test_rows/len(successful):,.0f}")
        print(f"  CL rows:           {cl_rows/len(successful):,.0f}")
    
    # Print errors if any
    if len(errors) > 0:
        print(f"\n{'='*70}")
        print("ERRORS")
        print(f"{'='*70}")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  {err['household']}: {err['message']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Print skipped if any
    if len(skipped) > 0:
        print(f"\n{'='*70}")
        print("SKIPPED HOUSEHOLDS")
        print(f"{'='*70}")
        for skip in skipped[:10]:  # Show first 10
            print(f"  {skip['household']}: {skip['message']}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more skipped")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("  PRE-COVID DATASET SPLITTING")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Input directory:     {INPUT_DIR}")
    print(f"  Output directory:    {OUTPUT_DIR}")
    print(f"  Base period:         {BASE_START} to {BASE_END}")
    print(f"  CL period:           {CL_START} to {CL_END}")
    print(f"  Base splits:         Train {TRAIN_RATIO*100:.0f}%, Val {VAL_RATIO*100:.0f}%, Test {TEST_RATIO*100:.0f}%")
    print(f"  CL windows:          {NUM_CL_WINDOWS}")
    print(f"  Min rows required:   {MIN_ROWS_REQUIRED:,}")
    
    try:
        # Step 1: Create output directories
        create_output_directories()
        
        # Step 2: Find all household CSVs
        input_path = Path(INPUT_DIR)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")
        
        csv_files = list(input_path.rglob("*.csv"))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")
        
        print(f"\n✓ Found {len(csv_files):,} household CSV files")
        
        # Step 3: Process all households
        print(f"\nProcessing households...")
        results = []
        
        for csv_path in tqdm(csv_files, desc="Splitting households"):
            result = process_household(csv_path)
            results.append(result)
        
        # Step 4: Print summary
        print_summary(results)
        
        print(f"\n{'='*70}")
        print("✓ SPLITTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"\nOutput location: {OUTPUT_DIR}/")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {str(e)}")
        print(f"{'='*70}")
        raise


if __name__ == "__main__":
    main()
