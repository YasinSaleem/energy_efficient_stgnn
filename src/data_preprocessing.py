#!/usr/bin/env python3
"""
Temporal Data Pipeline for Energy-Efficient ST-GNN (GPU + RAM SAFE)

- Loads split CSVs (train/val/test + CL windows)
- Aligns all households on a common UNION timeline
- Applies global StandardScaler (fit on train only)
- Uses LAZY sliding windows (NO RAM overflow)
- Returns true ST-GNN tensors:
      X: [B, 24, N, 1]
      Y: [B, 6, N]
- GPU efficient: pin_memory + lazy loading

Author: Energy-Efficient STGNN Project
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_ROOT = PROJECT_ROOT / "data" / "splits"
TRAIN_DIR = SPLITS_ROOT / "train"
VAL_DIR = SPLITS_ROOT / "val"
TEST_DIR = SPLITS_ROOT / "test"
CL_DIR = SPLITS_ROOT / "continual"

NODE_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "node_map.json"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "scaler_stgnn.pkl"

# Temporal setup
WINDOW_SIZE = 24      # past 24 hours
HORIZON = 6           # predict next 6 hours
STEP_SIZE = 1

# Loader setup (RTX SAFE)
BATCH_SIZE = 4        # ✅ RTX 4050 safe
NUM_WORKERS = 0       # ✅ Windows safe


# ============================================================================
# CORE UTILITIES
# ============================================================================

def load_node_map():
    with NODE_MAP_PATH.open("r") as f:
        node_map = json.load(f)
    print(f"[node_map] Loaded {len(node_map):,} nodes")
    return node_map


def load_split_timeseries(split_dir: Path, node_map: Dict[str, int]):
    if not split_dir.exists():
        raise FileNotFoundError(f"Split not found: {split_dir}")

    csv_files = sorted(split_dir.glob("*.csv"))
    print(f"[{split_dir.name}] Using {len(csv_files):,} households")

    # ✅ TIMELINE UNION
    global_timestamps = set()
    print(f"[{split_dir.name}] Scanning timestamps...")
    for f in tqdm(csv_files):
        df = pd.read_csv(f, usecols=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(inplace=True)
        global_timestamps.update(df["timestamp"].values)

    timestamps = sorted(global_timestamps)
    T, N = len(timestamps), len(node_map)
    panel = np.zeros((T, N), dtype=np.float32)
    ts_index = pd.Index(timestamps)

    print(f"[{split_dir.name}] Aligning households...")
    for f in tqdm(csv_files):
        hh_id = f.stem
        if hh_id not in node_map:
            continue

        idx = node_map[hh_id]
        df = pd.read_csv(f, usecols=["timestamp", "kWh"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(inplace=True)
        df.sort_values("timestamp", inplace=True)

        s = df.set_index("timestamp")["kWh"]
        s = s.reindex(ts_index).ffill().bfill().fillna(0.0)
        panel[:, idx] = s.values.astype(np.float32)

    print(f"[{split_dir.name}] Panel shape: {panel.shape}")
    return panel


def fit_scaler(panel):
    scaler = StandardScaler()
    flat = panel.reshape(-1, 1)
    scaler.fit(flat)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[scaler] Saved to {SCALER_PATH}")
    return scaler


def apply_scaler(panel, scaler):
    flat = panel.reshape(-1, 1)
    scaled = scaler.transform(flat)
    return scaled.reshape(panel.shape).astype(np.float32)


# ============================================================================
# LAZY ST-GNN DATASET (NO MEMORY EXPLOSION)
# ============================================================================

class STGNNDataset(Dataset):
    def __init__(self, panel, window, horizon):
        self.panel = panel
        self.window = window
        self.horizon = horizon
        self.T, self.N = panel.shape
        self.length = self.T - self.window - self.horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.panel[idx: idx + self.window, :]
        y = self.panel[idx + self.window:
                       idx + self.window + self.horizon, :]

        x = torch.from_numpy(x).unsqueeze(-1)  # [24, N, 1]
        y = torch.from_numpy(y)                # [6, N]
        return x, y


# ============================================================================
# DATALOADER BUILDERS
# ============================================================================

def get_base_dataloaders():
    node_map = load_node_map()

    print("\n=== Building BASE PANELS ===")
    train_panel = load_split_timeseries(TRAIN_DIR, node_map)
    val_panel = load_split_timeseries(VAL_DIR, node_map)
    test_panel = load_split_timeseries(TEST_DIR, node_map)

    print("\n=== Fitting scaler ===")
    scaler = fit_scaler(train_panel)

    train_panel = apply_scaler(train_panel, scaler)
    val_panel = apply_scaler(val_panel, scaler)
    test_panel = apply_scaler(test_panel, scaler)

    train_ds = STGNNDataset(train_panel, WINDOW_SIZE, HORIZON)
    val_ds = STGNNDataset(val_panel, WINDOW_SIZE, HORIZON)
    test_ds = STGNNDataset(test_panel, WINDOW_SIZE, HORIZON)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    print(f"[Base Loaders] Train={len(train_ds):,} Val={len(val_ds):,} Test={len(test_ds):,}")
    return train_loader, val_loader, test_loader


def get_cl_dataloaders():
    node_map = load_node_map()
    scaler = joblib.load(SCALER_PATH)

    cl_loaders = {}
    for cl in ["CL_1", "CL_2", "CL_3", "CL_4"]:
        cl_dir = CL_DIR / cl
        if not cl_dir.exists():
            continue

        print(f"\n=== Building {cl} ===")
        panel = load_split_timeseries(cl_dir, node_map)
        panel = apply_scaler(panel, scaler)

        ds = STGNNDataset(panel, WINDOW_SIZE, HORIZON)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

        cl_loaders[cl] = loader
        print(f"[{cl}] Samples = {len(ds):,}")

    return cl_loaders


# ============================================================================
# DIRECT TEST
# ============================================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_base_dataloaders()
    cl_loaders = get_cl_dataloaders()

    print("\n✅ GPU TEST BATCH:")
    for X, Y in train_loader:
        print("X:", X.shape)  # [B, 24, 8442, 1]
        print("Y:", Y.shape)  # [B, 6, 8442]
        break
