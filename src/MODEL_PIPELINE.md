# Project Pipeline & Script Flow

This document explains **which scripts to run, in what order**, and **what each script does** in the `energy_efficient_stgnn` project.

It assumes that the **raw dataset has already been downloaded** into:

```
data/raw/7362094/
```

and that the basic pre-COVID / post-COVID splitting has already been performed by your teammate.

---

## 0. Data Extraction & Splits (already done)

These steps were completed earlier and normally do not need to be rerun unless you change the raw data.

### Scripts (earlier stage, not part of src/ now):

**`extract_dataset.py`**

Extracts `.tzst` archives into per-household CSV files.

**Output:**
```
data/extracted/imp-pre/*.csv
data/extracted/imp-in/*.csv
data/extracted/imp-post/*.csv
```

**`split_pre_covid_dataset.py`, `create_continual_splits.py`**

Create train / val / test and continual learning (CL_1–CL_4) splits.

**Output:**
```
data/splits/train/*.csv
data/splits/val/*.csv
data/splits/test/*.csv

data/splits/continual/CL_1/*.csv
data/splits/continual/CL_2/*.csv
data/splits/continual/CL_3/*.csv
data/splits/continual/CL_4/*.csv
```

**→ You start from here.**

---

## 1. Spatial Graph Construction & Validation

### 1.1 `graph_construction.py`

**Command:**
```bash
python src/graph_construction.py
```

**What it does:**

- Reads `metadata.csv` from `data/raw/7362094/`
- Uses the training split households (`data/splits/train/`) to:
  - Build a node list (`node_map.json`: household → node index)
  - Extract spatial / contract features:
    - Province & municipality (one-hot)
    - Contracted power p1–p6 (standardized)
    - Tariff type (one-hot)
  - Compute a similarity matrix between households
  - Build a K-nearest neighbour (K-NN) graph
  - Optionally normalize adjacency

**Outputs** (in `data/processed/`):
- `adjacency_matrix.npz`
- `node_map.json`
- `graph_stats.json`

These are used by the STGNN to model spatial dependencies between households.

### 1.2 `graph_validation.py`

**Command:**
```bash
python src/graph_validation.py
```

**What it does:**

- Loads `adjacency_matrix.npz` and `node_map.json`
- Checks:
  - Matrix shape and sparsity
  - Node count consistency
  - Approximate symmetry
  - Degree and edge-weight statistics
- Prints a detailed summary so you can confirm the graph is sensible before training

---

## 2. Temporal Preprocessing & DataLoaders

### `data_preprocessing.py`

**Command (optional smoke test):**
```bash
python src/data_preprocessing.py
```

**What it does:**

- Reads the per-household CSVs from:
  ```
  data/splits/train/
  data/splits/val/
  data/splits/test/
  data/splits/continual/CL_1 … CL_4
  ```
- Aligns timestamps across all households (only keeps timestamps that exist for everyone)
- Builds large (time × nodes) panels for:
  - Train, validation, test
  - Continual learning windows CL_1–CL_4
- Fits a `StandardScaler` on the training panel and saves it as:
  ```
  data/processed/scaler_stgnn.pkl
  ```
- Creates sliding windows:
  - **Input window:** 24 hours
  - **Forecast horizon:** 6 hours
  - **Shape:**
    - X: `[B, 24, N, 1]`
    - Y: `[B, 6, N]`
- Wraps everything into custom `Dataset` and `DataLoader` objects:
  - `get_base_dataloaders()` → train / val / test loaders
  - `get_cl_dataloaders()` → CL_1–CL_4 loaders

**Note:** In practice, you normally do not call this script manually. It is imported and used inside `train.py`, `evaluate.py`, `continual_learning.py`, and `energy_tracking.py`.

---

## 3. STGNN Model Definition

### `model_stgnn.py`

**Command (shape sanity check):**
```bash
python src/model_stgnn.py
```

**What it does:**

- Loads the spatial adjacency (`adjacency_matrix.npz`)
- Defines the spatio-temporal GNN architecture:
  - `SpatialGCNLayer` → graph convolution over households
  - Temporal GRU (`GRU(16 → 32)`) over time dimension
  - Output layer → 6-step ahead forecast per node
- When run directly, it creates a dummy batch:
  - **Input:** `[4, 24, 8442, 1]`
  - **Output:** `[4, 6, 8442]`

**Note:** This file is imported by `train.py`, `evaluate.py`, and all continual learning scripts.

---

## 4. Base Training

### `train.py`  **MAIN TRAINING ENTRYPOINT**

**Command:**
```bash
python src/train.py
```

**What it does:**

- Detects GPU (RTX 4050) and sets device
- Calls `get_base_dataloaders()` to build train / val / test loaders
- Builds the STGNN model via `build_stgnn()`
- Trains for up to `max_epochs` with:
  - **Loss:** MSE
  - **Metrics:** MSE, RMSE, MAE on validation
  - **Early stopping** on validation RMSE (patience = 6)
- Prints training curves and progress bars (via `tqdm`)
- After each epoch, if validation RMSE improves:
  - Saves model to: `src/models/stgnn_best.pt`
- After training finishes, it evaluates once on the test set to print final test MSE / RMSE / MAE

---

## 5. Detailed Evaluation & Visualization

### `evaluate.py`

**Command:**
```bash
python src/evaluate.py
```

**What it does:**

- Loads the best model: `src/models/stgnn_best.pt`
- Calls `get_base_dataloaders()` and re-evaluates on:
  - Validation set
  - Test set
- Computes additional metrics:
  - MSE, RMSE, MAE
  - MAPE
  - R²
  - Per-horizon (1–6 hours) RMSE / MAE
- Generates plots:
  - Sample predictions vs. ground truth for selected nodes
  - Error distribution & Q–Q plot
- Saves everything under:
  ```
  results/evaluation/
    ├── evaluation_results.json
    ├── test_predictions.png
    └── test_error_dist.png
  ```

---

## 6. Continual Learning Baseline

### `continual_learning.py`

**Command:**
```bash
python src/continual_learning.py
```

**What it does:**

- Loads base train / val / test loaders and CL_1–CL_4 loaders
- Loads the base model from `src/models/stgnn_best.pt`
- Evaluates baseline performance on the original test set
- For each CL window (CL_1 → CL_4), sequentially:
  - Fine-tunes the model for a small number of epochs (3 by default) on that CL window
  - Evaluates on the CL window itself (new data performance)
  - Evaluates on the original test set again to measure forgetting
  - Saves the updated model to: `src/models/continual/stgnn_CL_X.pt`
- Aggregates results and writes:
  ```
  src/results/continual_learning/continual_learning_results.json
  ```
- Prints a per-window summary:
  - New data RMSE
  - Old test RMSE
  - Forgetting percentage
  - Time per update

**Note:** This gives a baseline continual learning behaviour without any energy awareness.

---

## 7. Energy Tracking (Continual Learning + Power)

### `energy_tracking.py`

**Command:**
```bash
python src/energy_tracking.py
```

**Important:** This script internally runs the full continual learning loop. You do not need to run `continual_learning.py` separately if you use this; it will:

- Start a CodeCarbon tracker to estimate:
  - Energy consumption (kWh)
  - CO₂ emissions (kg)
- Call `run_continual_learning()` from `continual_learning.py` to:
  - Load the base model
  - Build all DataLoaders
  - Run CL_1 → CL_4 fine-tuning with metrics & forgetting
- Stop the energy tracker and combine:
  - Runtime
  - Energy
  - Continual learning metrics

**Outputs:**
```
results/energy_tracking/
  ├── cl_energy_results.json
  └── codecarbon_logs/  (raw CodeCarbon output)
```

**Use this script when you want:**
- The same CL experiment as `continual_learning.py`, plus
- Measured energy use and estimated emissions

---

## 8. Energy-Aware Scheduling Simulation

### `energy_aware_scheduling.py`

**Command:**
```bash
python src/energy_aware_scheduling.py
```

**What it does (simulation only):**

- Does not run any model training
- Builds a synthetic 24-hour grid price + carbon-intensity profile
- Defines a set of jobs:
  - Base training job (e.g., 3 h @ 0.30 kW)
  - Four CL update jobs (e.g., 1 h @ 0.25 kW each)
- Simulates three scheduling strategies:
  - **`immediate`** – run jobs as soon as possible
  - **`night_only`** – restrict to 22:00–06:00
  - **`carbon_aware`** – place each job in the lowest-emission feasible slots
- For each strategy, computes:
  - Total energy (kWh)
  - Total cost (relative currency units)
  - Total CO₂ emissions (kg)
  - Per-job schedule: start time, end time, cost, emissions
- Writes all results to:
  ```
  results/energy_scheduling/energy_scheduling_results.json
  ```
- Prints a clean comparison table in the terminal so you can see how energy-aware scheduling can reduce emissions and cost, even with the same total compute

---

## Quick Run Order Summary

If you start from existing splits and want to reproduce the full baseline:

```bash
# 1) Graph
python src/graph_construction.py
python src/graph_validation.py

# 2) Base training
python src/train.py

# 3) Detailed evaluation
python src/evaluate.py

# 4) Continual learning baseline
python src/continual_learning.py

# 5) Continual learning + energy tracking
python src/energy_tracking.py

# 6) Energy-aware scheduling simulation (offline)
python src/energy_aware_scheduling.py
```

**Note:** `energy_tracking.py` includes running the continual learning loop internally, so you can use it alone if you only care about CL + energy results.

