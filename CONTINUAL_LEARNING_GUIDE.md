# Continual Learning Guide

Quick reference for running continual learning (CL) on the STGNN model.

---

## Prerequisites

### 1. Base Model Training
```bash
cd src/
python train.py
```
**Output:** `src/models/stgnn_best.pt`

### 2. Energy-Optimized Training
```bash
cd src/
python energy_optmization.py
```
**Output:** `src/models/stgnn_best_opt.pt`

### 3. Data Requirements
- Train/val/test splits: `data/splits/train_panel.npz`, `val_panel.npz`, `test_panel.npz`
- CL windows: `data/splits/continual/CL_1/`, `CL_2/`, `CL_3/`, `CL_4/`
- Scaler: `data/processed/scaler_stgnn.pkl` (created during base training)

---

## Running Continual Learning

### Base Continual Learning
Uses the base model (`stgnn_best.pt`) without energy optimizations.

```bash
cd src/
python continual_learning.py
```

**What it does:**
- Loads `stgnn_best.pt`
- Fine-tunes sequentially on CL_1 → CL_2 → CL_3 → CL_4
- Uses standard training: 3 epochs, AdamW optimizer
- Measures forgetting on old test set after each update

**Outputs:**
- Updated models: `src/models/continual/stgnn_CL_1.pt`, `stgnn_CL_2.pt`, etc.
- Results: `src/results/continual_learning/continual_learning_results.json`

**With Energy Tracking:**
```bash
cd src/
python energy_tracking_continual.py
```
Additional outputs:
- Energy metrics added to results JSON
- Emissions log: `src/results/continual_learning/emissions_continual.csv`

---

### Energy-Optimized Continual Learning
Uses the energy-optimized model (`stgnn_best_opt.pt`) with optimizations applied during CL.

```bash
cd src/
python energy_optimized_continual_learning.py
```

**What it does:**
- Loads `stgnn_best_opt.pt`
- Fine-tunes sequentially on CL_1 → CL_2 → CL_3 → CL_4
- Applies optimizations during fine-tuning:
  - Enhanced early stopping (patience=3, min_delta=1e-4)
  - Structured pruning (30% of weights)
- Measures forgetting on old test set after each update

**Outputs:**
- Updated models: `src/models/continual/stgnn_energy_optimized_CL_1.pt`, `stgnn_energy_optimized_CL_2.pt`, etc.
- Results: `src/results/continual_learning/energy_optimized_continual_learning_results.json`

**With Energy Tracking:**
```bash
cd src/
python energy_tracking_continual_opt.py
```
Additional outputs:
- Energy metrics added to results JSON
- Emissions log: `src/results/continual_learning/emissions_energy_optimized_cl.csv`

---

## Configuration

### Continual Learning Parameters
Defined in `continual_learning.py` and `energy_optimized_continual_learning.py`:
- `CL_EPOCHS = 3`: Fine-tuning epochs per window
- `CL_LEARNING_RATE = 5e-4`: Learning rate for CL updates
- `WEIGHT_DECAY = 1e-4`: L2 regularization

### Energy Optimizations (Optimized CL only)
- **Enhanced Early Stopping:**
  - `EARLY_STOP_PATIENCE = 3`: Stop after 3 epochs without improvement
  - `EARLY_STOP_MIN_DELTA = 1e-4`: Minimum loss improvement threshold
- **Structured Pruning:**
  - `PRUNING_AMOUNT = 0.3`: Remove 30% of weights in Linear layers

---

## Interpreting Results

### Key Metrics
- **New Data RMSE/MAE**: Performance on the new CL window
- **Old Test RMSE/MAE**: Performance on original test set (forgetting check)
- **Forgetting**: RMSE/MAE change on old data after CL update
  - Negative = improved on old data
  - <5% = minimal forgetting ✅
  - 5-10% = moderate forgetting ⚠️
  - >10% = significant forgetting ❌

### Example Output
```
Window     New RMSE     Old RMSE     Forgetting      Time (s)   Epochs
--------------------------------------------------------------------------------
CL_1       0.123456     0.145678     +2.34%          45.3       3
CL_2       0.134567     0.147890     +1.85%          38.7       2
CL_3       0.125678     0.148901     +0.98%          41.2       3
CL_4       0.126789     0.149012     +1.45%          36.8       2
```

### Results Files
- **JSON**: Full results with per-window metrics, forgetting analysis, training stats
- **CSV** (with energy tracking): CodeCarbon emissions log with timestamps, energy (kWh), CO2 (gCO2eq)

---

## Comparison: Base vs Energy-Optimized CL

| Aspect | Base CL | Energy-Optimized CL |
|--------|---------|---------------------|
| Base Model | `stgnn_best.pt` | `stgnn_best_opt.pt` |
| Early Stopping | None | Enhanced (patience=3, min_delta=1e-4) |
| Pruning | None | Structured (30%) after each update |
| Expected Time | Baseline | ~20-40% faster |
| Expected Energy | Baseline | ~25-45% lower |
| Model Size | Larger | Smaller (due to pruning) |

---

## Troubleshooting

### Missing Base Model
```
FileNotFoundError: Base model not found: src/models/stgnn_best.pt
```
**Solution:** Run `python train.py` first.

### Missing Optimized Model
```
FileNotFoundError: Energy-optimized model not found: src/models/stgnn_best_opt.pt
```
**Solution:** Run `python energy_optmization.py` first.

### No CL Windows Found
```
ValueError: No CL windows found! Check data/splits/continual/
```
**Solution:** Ensure `data/splits/continual/CL_1/`, `CL_2/`, `CL_3/`, `CL_4/` exist with CSV files.

### Missing Scaler
```
FileNotFoundError: Scaler not found: data/processed/scaler_stgnn.pkl
```
**Solution:** Run `python train.py` to generate the scaler.

---

## Notes

- **Temporal Order:** CL windows must be processed sequentially (CL_1 → CL_2 → CL_3 → CL_4)
- **Scaler:** Uses scaler from base training (no refitting on CL data)
- **Shuffle:** CL DataLoaders use `shuffle=False` to maintain temporal integrity
- **Device:** Automatically uses GPU if available, otherwise CPU
- **Memory:** Energy-optimized CL uses less memory due to pruning

---

## Quick Start Commands

```bash
# Full pipeline from scratch
cd src/

# 1. Base training
python train.py

# 2. Energy-optimized training
python energy_optmization.py

# 3a. Base CL
python continual_learning.py

# 3b. Energy-optimized CL
python energy_optimized_continual_learning.py

# 4a. Base CL with energy tracking
python energy_tracking_continual.py

# 4b. Optimized CL with energy tracking
python energy_tracking_continual_opt.py
```
