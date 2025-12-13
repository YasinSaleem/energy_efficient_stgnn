#!/usr/bin/env python3
"""
Data loading and SmartScaler-based normalization for STGNN.

Key design:
- Work entirely on CPU here.
- Only individual batches are moved to GPU inside train.py.
- Windowing is done on-the-fly in Dataset to avoid huge memory usage.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from utils import config as cfg

SPLITS_DIR = cfg.SPLITS_DIR
TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR   = SPLITS_DIR / "val"
TEST_DIR  = SPLITS_DIR / "test"

SCALER_PATH = cfg.PROCESSED_DIR / "scaler_stgnn.pkl"

TIMESTAMP_COL = "timestamp"
TARGET_COL    = "kWh"


# ============================================================================
# SmartScaler (same behavior as your previous logs)
# ============================================================================

class SmartScaler:
    """
    Simple global mean/std scaler for all entries (matching your previous logs).
    """

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, x: np.ndarray):
        """
        x: 2D array [T, N] of kWh values.
        """
        flat = x.reshape(-1)
        non_zero = flat[flat > 0]

        total_samples = flat.shape[0]
        zeros = np.sum(flat == 0)
        print("\n[scaler] Analyzing data...")
        print(f"  Total samples: {total_samples}")
        print(f"  Min: {flat.min():.6f}")
        print(f"  Max: {flat.max():.6f}")
        print(f"  Zeros: {zeros} ({zeros / total_samples * 100:.1f}%)")
        if non_zero.size > 0:
            print(f"  Non-zero min: {non_zero.min():.6f}")
            print(f"  Non-zero max: {non_zero.max():.6f}")
            print(f"  Non-zero median: {np.median(non_zero):.6f}")
            print(f"  Non-zero mean: {np.mean(non_zero):.6f}")
            print(f"  Non-zero std: {np.std(non_zero):.6f}")

        self.mean_ = float(np.mean(flat))
        self.std_  = float(np.std(flat))

        print("\n[scaler] Chosen parameters:")
        print(f"  Center (global mean): {self.mean_:.6f}")
        print(f"  Scale  (global std):  {self.std_:.6f}")

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / (self.std_ + 1e-8)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def save(self, path: Path):
        import pickle
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_}, f)

    def load(self, path: Path):
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.mean_ = obj["mean"]
        self.std_  = obj["std"]


# ============================================================================
# Helper: load full (T,N) matrix using a reference timeline
# ============================================================================

def load_split_matrix(split_dir: Path):
    """
    Load all CSVs in split_dir and stack them into a matrix [T, N_nodes].

    Strategy:
    - Use the timestamps from the FIRST CSV as the reference time index.
    - For every other household, align (reindex) to that index.
    - Fill missing timestamps with 0.0 for that household.

    This mirrors your original behavior where you had:
      Train: (9222, 8442)
      Val:   (1976, 8442)
      Test:  (1977, 8442)
    even though individual files had slightly different lengths.
    """

    csv_files = sorted(list(split_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {split_dir}")

    print(f"[{split_dir.name}] Found {len(csv_files)} files")

    # Reference index from the first file
    first_df = pd.read_csv(csv_files[0], usecols=[TIMESTAMP_COL, TARGET_COL])
    base_ts = pd.to_datetime(first_df[TIMESTAMP_COL])
    base_index = base_ts.sort_values()
    T = len(base_index)
    N = len(csv_files)

    print(f"  Example file: {csv_files[0].name}, rows={len(first_df)}")
    print(f"[{split_dir.name}] Using {T} reference timestamps across {N} nodes")

    mat = np.zeros((T, N), dtype=np.float32)

    # Fill first column directly from first file (aligned to base_index)
    s0 = pd.Series(first_df[TARGET_COL].astype(np.float32).values, index=base_ts)
    aligned0 = s0.reindex(base_index)
    if aligned0.isna().any():
        aligned0 = aligned0.fillna(0.0)
    mat[:, 0] = aligned0.values.astype(np.float32)

    # Process remaining files
    for j, f in enumerate(csv_files[1:], start=1):
        df = pd.read_csv(f, usecols=[TIMESTAMP_COL, TARGET_COL])
        ts = pd.to_datetime(df[TIMESTAMP_COL])
        s = pd.Series(df[TARGET_COL].astype(np.float32).values, index=ts)

        aligned = s.reindex(base_index)
        if aligned.isna().any():
            aligned = aligned.fillna(0.0)

        mat[:, j] = aligned.values.astype(np.float32)

        if j % 1000 == 0:
            print(f"  Processed {j} / {len(csv_files)} files for alignment...")

    print(f"[{split_dir.name}] Final matrix shape: {mat.shape[0]} timestamps × {mat.shape[1]} nodes")
    return base_index, mat


# ============================================================================
# Dataset: windowing on-the-fly (CPU only)
# ============================================================================

class STGNNDataset(Dataset):
    """
    Dataset for STGNN.

    data_matrix: np.ndarray [T, N]
    Returns X: [W, N, 1], Y: [H, N, 1]
    DataLoader will stack -> X: [B, W, N, 1]
    """

    def __init__(self, data_matrix: np.ndarray, window_size: int, horizon: int):
        super().__init__()
        self.data = torch.from_numpy(data_matrix.astype(np.float32))  # [T, N] on CPU
        self.window_size = window_size
        self.horizon = horizon
        self.num_samples = self.data.shape[0] - window_size - horizon + 1

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        w = self.window_size
        h = self.horizon
        x = self.data[idx:idx + w]                    # [W, N]
        y = self.data[idx + w: idx + w + h]          # [H, N]

        x = x.unsqueeze(-1)  # [W, N, 1]
        y = y.unsqueeze(-1)  # [H, N, 1]
        return x, y


# ============================================================================
# Main function: build dataloaders
# ============================================================================

def get_base_dataloaders():
    """
    Returns train_loader, val_loader, test_loader.

    All heavy tensors remain on CPU. Batches are moved to GPU in train.py.
    """
    print("\n" + "="*60)
    print("📦 BUILDING PROPERLY NORMALIZED DATALOADERS")
    print("="*60)

    # 1) Load raw matrices for each split
    train_ts, train_mat = load_split_matrix(TRAIN_DIR)
    val_ts,   val_mat   = load_split_matrix(VAL_DIR)
    test_ts,  test_mat  = load_split_matrix(TEST_DIR)

    print("\n📊 RAW DATA STATISTICS:")
    def stats(name, mat):
        flat = mat.reshape(-1)
        non_zero = flat[flat > 0]
        print(f"  {name}:")
        print(f"    Shape: {mat.shape}")
        print(f"    All - Min: {flat.min():.3f}, Mean: {flat.mean():.3f}, Max: {flat.max():.3f}")
        if non_zero.size > 0:
            print(f"    Non-zero - Min: {non_zero.min():.3f}, Median: {np.median(non_zero):.3f}, Mean: {non_zero.mean():.3f}")

    stats("Train", train_mat)
    stats("Val",   val_mat)
    stats("Test",  test_mat)

    # 2) Fit SmartScaler on TRAIN only
    print("\n=== Fitting smart scaler ===")
    scaler = SmartScaler()
    train_norm = scaler.fit_transform(train_mat)
    print("\n[scaler] Expected transformations (approx):")
    for v in [0.0, 0.05, 0.117, 0.2, 0.5, 1.0, 10.0]:
        print(f"  {v:6.3f} kWh → {((v - scaler.mean_) / (scaler.std_ + 1e-8)):7.2f}")
    scaler.save(SCALER_PATH)
    print(f"[scaler] Saved to {SCALER_PATH}")

    # 3) Apply same scaler to val/test
    val_norm  = scaler.transform(val_mat)
    test_norm = scaler.transform(test_mat)

    # 4) Check normalization stats
    print("\n=== Applying smart normalization ===")
    def norm_stats(name, mat):
        flat = mat.reshape(-1)
        print(f"  {name}: mean={flat.mean():.4f}, std={flat.std():.4f}")
    norm_stats("Train", train_norm)
    norm_stats("Val",   val_norm)
    norm_stats("Test",  test_norm)
    print("✅ Normalization OK (train~N(0,1); val/test may differ in std/mean due to distribution shift)")

    # 5) Create Datasets (windowing on-the-fly)
    W = cfg.WINDOW_SIZE
    H = cfg.HORIZON

    train_ds = STGNNDataset(train_norm, window_size=W, horizon=H)
    val_ds   = STGNNDataset(val_norm,   window_size=W, horizon=H)
    test_ds  = STGNNDataset(test_norm,  window_size=W, horizon=H)

    print("\n=== Creating datasets ===")
    print(f"✅ DATALOADERS READY:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print("="*60)

    # 6) Build DataLoaders (still CPU-only)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


def get_cl_dataloaders():
    """
    Returns dictionary of continual learning DataLoaders.
    
    Uses the SAME scaler fitted on training data (loaded from disk).
    This ensures consistent normalization for continual learning windows.
    
    Returns:
        dict: {"CL_1": loader1, "CL_2": loader2, "CL_3": loader3, "CL_4": loader4}
    """
    print("\n" + "="*60)
    print("📦 BUILDING CONTINUAL LEARNING DATALOADERS")
    print("="*60)
    
    # Define CL directories
    CL_DIRS = {
        "CL_1": SPLITS_DIR / "continual" / "CL_1",
        "CL_2": SPLITS_DIR / "continual" / "CL_2",
        "CL_3": SPLITS_DIR / "continual" / "CL_3",
        "CL_4": SPLITS_DIR / "continual" / "CL_4",
    }
    
    # Load the scaler fitted on training data
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}\n"
            "Please run base training first to generate the scaler."
        )
    
    print(f"\n[scaler] Loading existing scaler from {SCALER_PATH}")
    scaler = SmartScaler()
    scaler.load(SCALER_PATH)
    print(f"[scaler] Loaded: mean={scaler.mean_:.6f}, std={scaler.std_:.6f}")
    
    cl_loaders = {}
    
    for cl_name, cl_dir in CL_DIRS.items():
        if not cl_dir.exists():
            print(f"\n⚠️  Warning: {cl_name} directory not found: {cl_dir}")
            continue
        
        print(f"\n[{cl_name}] Loading data...")
        cl_ts, cl_mat = load_split_matrix(cl_dir)
        
        # Apply the EXISTING scaler (no refitting)
        cl_norm = scaler.transform(cl_mat)
        print(f"[{cl_name}] Normalized: mean={cl_norm.mean():.4f}, std={cl_norm.std():.4f}")
        
        # Create dataset with same window/horizon
        W = cfg.WINDOW_SIZE
        H = cfg.HORIZON
        cl_ds = STGNNDataset(cl_norm, window_size=W, horizon=H)
        print(f"[{cl_name}] Dataset: {len(cl_ds)} samples")
        
        # Create DataLoader
        cl_loader = DataLoader(
            cl_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,  # Keep temporal order for CL
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )
        
        cl_loaders[cl_name] = cl_loader
    
    print("\n" + "="*60)
    print(f"✅ CL DATALOADERS READY: {len(cl_loaders)} windows")
    print("="*60)
    
    return cl_loaders


if __name__ == "__main__":
    # Simple test run
    train_loader, val_loader, test_loader = get_base_dataloaders()
    batch = next(iter(train_loader))
    X, Y = batch
    print("\nSample batch shapes:")
    print("  X:", X.shape)  # [B, W, N, 1]
    print("  Y:", Y.shape)  # [B, H, N, 1]
    
    # Test CL loaders
    print("\n" + "="*60)
    print("Testing CL DataLoaders...")
    cl_loaders = get_cl_dataloaders()
    for cl_name, cl_loader in cl_loaders.items():
        print(f"\n{cl_name}: {len(cl_loader)} batches")
        if len(cl_loader) > 0:
            X_cl, Y_cl = next(iter(cl_loader))
            print(f"  Batch shapes - X: {X_cl.shape}, Y: {Y_cl.shape}")
