#!/usr/bin/env python3
"""
Dataset Integrity Validation Script for STGNN Energy Forecasting

Validates dataset integrity before training to catch:
- Shape inconsistencies
- Time leakage between input/target windows
- Overlap between train/val/test splits
- Data leakage from global scaling
- Distribution anomalies

Author: Energy-Efficient STGNN Project
"""

import sys
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import config as cfg

# ============================================================================
# DEVICE SELECTION
# ============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"\n✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"\n✅ Using Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print(f"\n⚠️  Using CPU (no GPU available)")

# For validation, use CPU to avoid memory issues
DEVICE = torch.device("cpu")
print(f"[Validation] Using CPU to avoid memory constraints")

# ============================================================================
# CONSTANTS
# ============================================================================

WINDOW_SIZE = cfg.WINDOW_SIZE
HORIZON = cfg.HORIZON
SPLITS_DIR = cfg.SPLITS_DIR
TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR = SPLITS_DIR / "val"
TEST_DIR = SPLITS_DIR / "test"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_split_data(split_dir):
    """Load all CSV files from a split directory and extract timestamps."""
    import pandas as pd
    
    all_timestamps = []
    all_values = []
    file_count = 0
    
    csv_files = sorted(split_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Use (household_id, timestamp) tuples for proper uniqueness checking
        if {'timestamp', 'LCLid'}.issubset(df.columns):
            # LCLid is the household identifier in this dataset
            all_timestamps.extend(
                list(zip(df['LCLid'], df['timestamp']))
            )
        elif 'timestamp' in df.columns:
            # Fallback: use filename as household identifier
            household_id = csv_file.stem  # filename without extension
            all_timestamps.extend(
                [(household_id, ts) for ts in df['timestamp'].tolist()]
            )
        
        if 'energy(kWh/hh)' in df.columns:
            all_values.extend(df['energy(kWh/hh)'].tolist())
        file_count += 1
    
    return all_timestamps, all_values, file_count


def check_shape_consistency(X, y, split_name):
    """Check if X and y have consistent shapes."""
    print(f"\n{'='*80}")
    print(f"[{split_name}] SHAPE CONSISTENCY CHECK")
    print(f"{'='*80}")
    
    issues = []
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Expected shapes
    expected_X = (X.shape[0], WINDOW_SIZE, X.shape[2])  # (samples, window, nodes)
    expected_y = (y.shape[0], HORIZON, y.shape[2])       # (samples, horizon, nodes)
    
    if X.shape[1] != WINDOW_SIZE:
        issues.append(f"❌ X window size is {X.shape[1]}, expected {WINDOW_SIZE}")
    else:
        print(f"✅ X window size correct: {WINDOW_SIZE}")
    
    if y.shape[1] != HORIZON:
        issues.append(f"❌ y horizon is {y.shape[1]}, expected {HORIZON}")
    else:
        print(f"✅ y horizon correct: {HORIZON}")
    
    if X.shape[0] != y.shape[0]:
        issues.append(f"❌ Sample mismatch: X has {X.shape[0]}, y has {y.shape[0]}")
    else:
        print(f"✅ Sample count matches: {X.shape[0]}")
    
    if X.shape[2] != y.shape[2]:
        issues.append(f"❌ Node mismatch: X has {X.shape[2]}, y has {y.shape[2]}")
    else:
        print(f"✅ Node count matches: {X.shape[2]}")
    
    return issues


def check_time_leakage_within_sample(X, y, split_name, num_samples=5):
    """Check if target window overlaps with input window (time leakage)."""
    print(f"\n{'='*80}")
    print(f"[{split_name}] TIME LEAKAGE CHECK (Input vs Target)")
    print(f"{'='*80}")
    
    issues = []
    
    # Check if last timestep of X matches first timestep of y
    # This would indicate overlap (time leakage)
    
    samples_to_check = min(num_samples, X.shape[0])
    
    for i in range(samples_to_check):
        X_sample = X[i]  # shape: (window, nodes)
        y_sample = y[i]  # shape: (horizon, nodes)
        
        # Last input timestep
        X_last = X_sample[-1]
        # First target timestep
        y_first = y_sample[0]
        
        # Use correlation-based similarity check (more robust than strict equality)
        try:
            sim = torch.corrcoef(torch.stack([X_last.flatten(), y_first.flatten()]))[0, 1]
            if sim > 0.999:
                issues.append(f"⚠️  Sample {i}: Extremely high similarity (corr={sim:.4f}) between X_last and y_first")
        except:
            # Fallback to exact match if correlation fails (e.g., zero variance)
            if torch.allclose(X_last, y_first, atol=1e-6):
                issues.append(f"⚠️  Sample {i}: Last input timestep == First target timestep (exact match)")
        
        # Check if target values appear in input (optimized: check only first 2 target timesteps)
        for t in range(min(2, y_sample.shape[0])):
            try:
                sim = torch.corrcoef(
                    torch.stack([X_last.flatten(), y_sample[t].flatten()])
                )[0, 1]
                if sim > 0.999:
                    issues.append(f"⚠️  Sample {i}: Suspicious similarity (corr={sim:.4f}) in target timestep {t}")
            except:
                pass  # Skip if correlation computation fails
    
    if not issues:
        print(f"✅ No time leakage detected (checked {samples_to_check} samples)")
    else:
        print(f"❌ Found {len(issues)} potential time leakage issues:")
        for issue in issues[:10]:  # Print first 10
            print(f"   {issue}")
    
    return issues


def check_split_disjoint(train_timestamps, val_timestamps, test_timestamps):
    """Verify that train/val/test are time-disjoint."""
    print(f"\n{'='*80}")
    print(f"TIME-DISJOINT VALIDATION (Train/Val/Test)")
    print(f"{'='*80}")
    
    issues = []
    
    train_set = set(train_timestamps)
    val_set = set(val_timestamps)
    test_set = set(test_timestamps)
    
    # Check overlaps
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    print(f"Train (household, timestamp) pairs: {len(train_set)} unique")
    print(f"Val (household, timestamp) pairs: {len(val_set)} unique")
    print(f"Test (household, timestamp) pairs: {len(test_set)} unique")
    
    if train_val_overlap:
        issues.append(f"❌ Train/Val overlap: {len(train_val_overlap)} (household, timestamp) pairs")
        print(f"❌ Train/Val overlap: {len(train_val_overlap)} (household, timestamp) pairs")
    else:
        print("✅ Train/Val are disjoint")
    
    if train_test_overlap:
        issues.append(f"❌ Train/Test overlap: {len(train_test_overlap)} (household, timestamp) pairs")
        print(f"❌ Train/Test overlap: {len(train_test_overlap)} (household, timestamp) pairs")
    else:
        print("✅ Train/Test are disjoint")
    
    if val_test_overlap:
        issues.append(f"❌ Val/Test overlap: {len(val_test_overlap)} (household, timestamp) pairs")
        print(f"❌ Val/Test overlap: {len(val_test_overlap)} (household, timestamp) pairs")
    else:
        print("✅ Val/Test are disjoint")
    
    return issues


def print_timestamp_ranges(train_timestamps, val_timestamps, test_timestamps):
    """Print timestamp ranges for each split."""
    print(f"\n{'='*80}")
    print(f"TIMESTAMP RANGES")
    print(f"{'='*80}")
    
    if train_timestamps:
        print(f"Train: {min(train_timestamps)} → {max(train_timestamps)}")
    else:
        print("Train: No timestamps found")
    
    if val_timestamps:
        print(f"Val:   {min(val_timestamps)} → {max(val_timestamps)}")
    else:
        print("Val: No timestamps found")
    
    if test_timestamps:
        print(f"Test:  {min(test_timestamps)} → {max(test_timestamps)}")
    else:
        print("Test: No timestamps found")


def compare_distributions(X_train, y_train, X_val, y_val, X_test, y_test):
    """Compare statistical distributions across splits."""
    print(f"\n{'='*80}")
    print(f"DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    issues = []
    
    # Flatten for statistics
    train_X_flat = X_train.flatten()
    train_y_flat = y_train.flatten()
    val_X_flat = X_val.flatten()
    val_y_flat = y_val.flatten()
    test_X_flat = X_test.flatten()
    test_y_flat = y_test.flatten()
    
    print(f"\n{'Split':<10} {'Type':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*70}")
    
    # Train
    print(f"{'Train':<10} {'Input':<8} {train_X_flat.mean():<12.6f} {train_X_flat.std():<12.6f} {train_X_flat.min():<12.6f} {train_X_flat.max():<12.6f}")
    print(f"{'Train':<10} {'Target':<8} {train_y_flat.mean():<12.6f} {train_y_flat.std():<12.6f} {train_y_flat.min():<12.6f} {train_y_flat.max():<12.6f}")
    
    # Val
    print(f"{'Val':<10} {'Input':<8} {val_X_flat.mean():<12.6f} {val_X_flat.std():<12.6f} {val_X_flat.min():<12.6f} {val_X_flat.max():<12.6f}")
    print(f"{'Val':<10} {'Target':<8} {val_y_flat.mean():<12.6f} {val_y_flat.std():<12.6f} {val_y_flat.min():<12.6f} {val_y_flat.max():<12.6f}")
    
    # Test
    print(f"{'Test':<10} {'Input':<8} {test_X_flat.mean():<12.6f} {test_X_flat.std():<12.6f} {test_X_flat.min():<12.6f} {test_X_flat.max():<12.6f}")
    print(f"{'Test':<10} {'Target':<8} {test_y_flat.mean():<12.6f} {test_y_flat.std():<12.6f} {test_y_flat.min():<12.6f} {test_y_flat.max():<12.6f}")
    
    # Check for suspicious patterns
    train_mean = train_X_flat.mean().item()
    val_mean = val_X_flat.mean().item()
    test_mean = test_X_flat.mean().item()
    
    train_std = train_X_flat.std().item()
    val_std = val_X_flat.std().item()
    test_std = test_X_flat.std().item()
    
    # Flag if distributions are too different
    mean_diff_val = abs(train_mean - val_mean) / (train_mean + 1e-8)
    mean_diff_test = abs(train_mean - test_mean) / (train_mean + 1e-8)
    
    print(f"\n{'='*80}")
    print(f"DISTRIBUTION SHIFT DETECTION")
    print(f"{'='*80}")
    print(f"Train vs Val mean difference:  {mean_diff_val*100:.2f}%")
    print(f"Train vs Test mean difference: {mean_diff_test*100:.2f}%")
    
    if mean_diff_val > 0.3:  # 30% difference
        issues.append(f"⚠️  Large mean shift between Train and Val: {mean_diff_val*100:.2f}%")
        print(f"⚠️  Large mean shift between Train and Val: {mean_diff_val*100:.2f}%")
    
    if mean_diff_test > 0.3:
        issues.append(f"⚠️  Large mean shift between Train and Test: {mean_diff_test*100:.2f}%")
        print(f"⚠️  Large mean shift between Train and Test: {mean_diff_test*100:.2f}%")
    
    # Check for identical statistics (sign of data leakage)
    if abs(train_mean - val_mean) < 1e-6 and abs(train_std - val_std) < 1e-6:
        issues.append("❌ Train and Val have IDENTICAL statistics (data leakage suspected)")
        print("❌ Train and Val have IDENTICAL statistics (data leakage suspected)")
    
    if abs(train_mean - test_mean) < 1e-6 and abs(train_std - test_std) < 1e-6:
        issues.append("❌ Train and Test have IDENTICAL statistics (data leakage suspected)")
        print("❌ Train and Test have IDENTICAL statistics (data leakage suspected)")
    
    return issues


def detect_global_scaling_leakage(X_train, X_val, X_test):
    """Detect if global scaling was applied (uses test data statistics)."""
    print(f"\n{'='*80}")
    print(f"GLOBAL SCALING LEAKAGE DETECTION")
    print(f"{'='*80}")
    
    issues = []
    
    # Calculate combined statistics
    all_data = torch.cat([X_train.flatten(), X_val.flatten(), X_test.flatten()])
    global_mean = all_data.mean().item()
    global_std = all_data.std().item()
    
    # Check if train data was normalized with global stats
    train_mean = X_train.flatten().mean().item()
    train_std = X_train.flatten().std().item()
    
    # Additional check: train-only vs train+val combined
    train_only_mean = X_train.flatten().mean().item()
    combined_train_val_mean = torch.cat([X_train.flatten(), X_val.flatten()]).mean().item()
    
    print(f"Global mean (train+val+test): {global_mean:.6f}")
    print(f"Global std (train+val+test):  {global_std:.6f}")
    print(f"Train-only mean:              {train_only_mean:.6f}")
    print(f"Train-only std:               {train_std:.6f}")
    print(f"Combined (train+val) mean:    {combined_train_val_mean:.6f}")
    
    # Check for global fitting of scaler (train == combined mean)
    if abs(train_only_mean - combined_train_val_mean) < 1e-6:
        issues.append("⚠️  Possible global fitting of scaler (train mean == combined mean)")
        print("⚠️  Possible global fitting of scaler (train mean == combined mean)")
        print("    This suggests normalization was done on entire dataset before splitting")
    
    # If train is centered around 0 with std ~1, likely normalized
    if abs(train_mean) < 0.1 and abs(train_std - 1.0) < 0.1:
        # Check if this matches global stats
        if abs(global_mean) < 0.1 and abs(global_std - 1.0) < 0.1:
            issues.append("⚠️  Data appears globally normalized (potential test leakage)")
            print("⚠️  Data appears globally normalized (potential test leakage)")
            print("    Recommendation: Use train-only statistics for normalization")
        else:
            print("✅ Data normalized using train-only statistics")
    else:
        print("ℹ️  Data does not appear to be normalized")
    
    return issues


def check_sample_overlap(X_train, X_val, X_test):
    """Check for identical samples across splits."""
    print(f"\n{'='*80}")
    print(f"SAMPLE OVERLAP DETECTION")
    print(f"{'='*80}")
    
    issues = []
    
    # Hash samples for efficient comparison
    def hash_sample(x):
        return hash(x.flatten().cpu().numpy().tobytes())
    
    print("Hashing samples (this may take a moment)...")
    
    # Sample a subset for computational efficiency
    max_samples = 1000
    train_hashes = {hash_sample(X_train[i]) for i in range(min(len(X_train), max_samples))}
    val_hashes = {hash_sample(X_val[i]) for i in range(min(len(X_val), max_samples))}
    test_hashes = {hash_sample(X_test[i]) for i in range(min(len(X_test), max_samples))}
    
    # Check overlaps
    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes
    
    if train_val_overlap:
        issues.append(f"❌ Found {len(train_val_overlap)} identical samples between Train and Val")
        print(f"❌ Found {len(train_val_overlap)} identical samples between Train and Val")
    else:
        print("✅ No identical samples found between Train and Val")
    
    if train_test_overlap:
        issues.append(f"❌ Found {len(train_test_overlap)} identical samples between Train and Test")
        print(f"❌ Found {len(train_test_overlap)} identical samples between Train and Test")
    else:
        print("✅ No identical samples found between Train and Test")
    
    if val_test_overlap:
        issues.append(f"❌ Found {len(val_test_overlap)} identical samples between Val and Test")
        print(f"❌ Found {len(val_test_overlap)} identical samples between Val and Test")
    else:
        print("✅ No identical samples found between Val and Test")
    
    return issues


def check_nan_inf(X, y, split_name):
    """Check for NaN or Inf values."""
    print(f"\n{'='*80}")
    print(f"[{split_name}] NaN/Inf CHECK")
    print(f"{'='*80}")
    
    issues = []
    
    X_nan = torch.isnan(X).sum().item()
    X_inf = torch.isinf(X).sum().item()
    y_nan = torch.isnan(y).sum().item()
    y_inf = torch.isinf(y).sum().item()
    
    if X_nan > 0:
        issues.append(f"❌ X contains {X_nan} NaN values")
        print(f"❌ X contains {X_nan} NaN values")
    else:
        print("✅ X has no NaN values")
    
    if X_inf > 0:
        issues.append(f"❌ X contains {X_inf} Inf values")
        print(f"❌ X contains {X_inf} Inf values")
    else:
        print("✅ X has no Inf values")
    
    if y_nan > 0:
        issues.append(f"❌ y contains {y_nan} NaN values")
        print(f"❌ y contains {y_nan} NaN values")
    else:
        print("✅ y has no NaN values")
    
    if y_inf > 0:
        issues.append(f"❌ y contains {y_inf} Inf values")
        print(f"❌ y contains {y_inf} Inf values")
    else:
        print("✅ y has no Inf values")
    
    return issues


# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def main():
    print("\n" + "="*80)
    print("DATASET INTEGRITY VALIDATION")
    print("="*80)
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Forecast horizon: {HORIZON}")
    print(f"Splits directory: {SPLITS_DIR}")
    
    all_issues = []
    
    # Check if directories exist
    if not TRAIN_DIR.exists():
        print(f"\n❌ ERROR: Train directory not found: {TRAIN_DIR}")
        return
    if not VAL_DIR.exists():
        print(f"\n❌ ERROR: Val directory not found: {VAL_DIR}")
        return
    if not TEST_DIR.exists():
        print(f"\n❌ ERROR: Test directory not found: {TEST_DIR}")
        return
    
    # Load data from preprocessing
    print(f"\n{'='*80}")
    print("LOADING DATASET")
    print(f"{'='*80}")
    
    try:
        from src.data_preprocessing import get_base_dataloaders
        train_loader, val_loader, test_loader = get_base_dataloaders()
        
        # Extract tensors from dataloaders
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_test_list, y_test_list = [], []
        
        print("Extracting training data...")
        for X, y in train_loader:
            X_train_list.append(X)
            y_train_list.append(y)
        
        print("Extracting validation data...")
        for X, y in val_loader:
            X_val_list.append(X)
            y_val_list.append(y)
        
        print("Extracting test data...")
        for X, y in test_loader:
            X_test_list.append(X)
            y_test_list.append(y)
        
        X_train = torch.cat(X_train_list, dim=0)
        y_train = torch.cat(y_train_list, dim=0)
        X_val = torch.cat(X_val_list, dim=0)
        y_val = torch.cat(y_val_list, dim=0)
        X_test = torch.cat(X_test_list, dim=0)
        y_test = torch.cat(y_test_list, dim=0)
        
        print(f"\n✅ Loaded tensors successfully")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        print(f"   Test:  {len(X_test)} samples")
        
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        print("Make sure data_preprocessing.py is working correctly")
        return
    
    # Run validation checks
    all_issues.extend(check_shape_consistency(X_train, y_train, "TRAIN"))
    all_issues.extend(check_shape_consistency(X_val, y_val, "VAL"))
    all_issues.extend(check_shape_consistency(X_test, y_test, "TEST"))
    
    all_issues.extend(check_nan_inf(X_train, y_train, "TRAIN"))
    all_issues.extend(check_nan_inf(X_val, y_val, "VAL"))
    all_issues.extend(check_nan_inf(X_test, y_test, "TEST"))
    
    all_issues.extend(check_time_leakage_within_sample(X_train, y_train, "TRAIN"))
    all_issues.extend(check_time_leakage_within_sample(X_val, y_val, "VAL"))
    all_issues.extend(check_time_leakage_within_sample(X_test, y_test, "TEST"))
    
    # Load timestamps for temporal checks
    print(f"\n{'='*80}")
    print("LOADING TIMESTAMPS FROM CSV FILES")
    print(f"{'='*80}")
    
    train_timestamps, train_values, train_files = load_split_data(TRAIN_DIR)
    val_timestamps, val_values, val_files = load_split_data(VAL_DIR)
    test_timestamps, test_values, test_files = load_split_data(TEST_DIR)
    
    print(f"Loaded {train_files} train files")
    print(f"Loaded {val_files} val files")
    print(f"Loaded {test_files} test files")
    
    if train_timestamps and val_timestamps and test_timestamps:
        all_issues.extend(check_split_disjoint(train_timestamps, val_timestamps, test_timestamps))
        print_timestamp_ranges(train_timestamps, val_timestamps, test_timestamps)
    else:
        print("⚠️  Could not load timestamps for temporal validation")
    
    all_issues.extend(compare_distributions(X_train, y_train, X_val, y_val, X_test, y_test))
    all_issues.extend(detect_global_scaling_leakage(X_train, X_val, X_test))
    all_issues.extend(check_sample_overlap(X_train, X_val, X_test))
    
    # Final summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    if all_issues:
        print(f"\n❌ FOUND {len(all_issues)} ISSUES:\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        print(f"\n⚠️  WARNING: Dataset has integrity issues. Review and fix before training.")
    else:
        print("\n✅ ALL CHECKS PASSED")
        print("Dataset appears to be correctly prepared for training.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
