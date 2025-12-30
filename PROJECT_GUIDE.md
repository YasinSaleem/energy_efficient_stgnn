# Energy-Efficient STGNN: Project Guide

Complete guide for running the energy-efficient spatio-temporal graph neural network pipeline.

---

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Navigate to project
cd energy_efficient_stgnn/
```

---

## Execution Pipeline

### Phase 1: Data Preparation (One-Time Setup)

#### 1. Extract Dataset
```bash
cd scripts/
python extract_dataset.py
```
**What it does:** Extracts `.tzst` archives into per-household CSV files  
**Output:** `data/extracted/imp-pre/`, `imp-in/`, `imp-post/`  
**Status:** ✅ Usually pre-completed

---

#### 2. Split Data
```bash
python split_pre_covid_dataset.py
```
**What it does:** Creates train (60%), val (10%), test (10%), continual learning (20%) temporal splits  
**Output:** `data/splits/train/`, `val/`, `test/`, `continual/CL_1-4/`

---

#### 3. Validate Splits (Optional)
```bash
python validate_splits.py
python analyze_dataset_quality.py
```
**What it does:** Checks data quality, missing values, timestamp alignment

---

### Phase 2: Graph Construction

#### 4. Build Spatial Graph
```bash
cd ../src/
python graph_construction.py
```
**What it does:** Builds K-NN graph (K=10) from household features (province, tariff, contracted power)  
**Output:** `data/processed/adjacency_matrix.npz`, `node_map.json`  
**Time:** ~2-3 minutes

---

#### 5. Validate Graph
```bash
python graph_validation.py
```
**What it does:** Verifies graph properties (8,442 nodes, ~84K edges, ~99.88% sparse)

---

### Phase 3: Baseline Training & Evaluation

#### 6. Train Baseline Model
```bash
python train.py
```
**What it does:**
- Trains STGNN on 24h window → 6h forecast
- AdamW optimizer + ReduceLROnPlateau scheduler
- Early stopping (patience=3)
- Huber loss, dropout=0.3

**Output:** `src/models/stgnn_best.pt`  
**Time:** ~88 minutes  
**Results:** Test RMSE 0.713, Energy 0.114 kWh

---

#### 7. Evaluate Model
```bash
python evaluate.py
```
**What it does:** Computes comprehensive metrics (MSE, RMSE, MAE, MAPE, R²)  
**Output:** `src/results/evaluation/evaluation_results.json`

---

#### 8. Baseline Continual Learning
```bash
python continual_learning.py
```
**What it does:**
- Fine-tunes on CL_1 → CL_2 → CL_3 → CL_4 (3 epochs each, LR=5e-4)
- Measures forgetting on original test set

**Output:** `src/models/continual/stgnn_CL_*.pt`  
**Time:** ~30 minutes  
**Results:** Avg forgetting +1.12%

---

### Phase 4: Energy-Optimized Training & CL

#### 9. Train Energy-Optimized Model
```bash
python train_optimized.py
```
**What it does:**
- **Gradient accumulation 4x** (75% fewer optimizer calls)
- **Larger eval batches 4x** (faster validation)
- Conservative early stopping (~35-40 epochs vs ~45)

**Output:** `src/models/stgnn_best_optimized.pt`  
**Time:** ~75 minutes (-14.7%)  
**Results:** Test RMSE 0.737 (+3.4%), Energy 0.095 kWh (**-16.5%**)

> **🔬 Research Novelty #1:** Sparse GNNs can achieve 16.5% energy savings via gradient accumulation when mixed precision is unavailable

---

#### 10. Ultra-Aggressive Continual Learning
```bash
python energy_optimized_continual_learning.py
```
**What it does:**
- **1 epoch per window** (vs 3 in baseline)
- **Gradient accumulation 16x** (93.75% fewer backward passes)
- **High LR 8e-4** with 50-step warmup
- Light regularization

**Output:** `src/models/continual_aggressive/stgnn_ultra_aggressive_CL_*.pt`  
**Time:** ~6 minutes (**-80%**)  
**Results:** Avg forgetting **-4.44%** (improvement!), Energy 0.0005 kWh/update (**-83.3%**)

> **🔬 Research Novelty #2 (Counterintuitive):** Single-epoch high-LR training produces LESS forgetting than multi-epoch low-LR training

> **🔬 Research Novelty #3 (Compounding Savings):** Optimization benefits grow with deployment frequency:
> - Single training: 16.5% savings
> - With 52 weekly updates/year: **55.7% annual savings**

---

### Phase 5: Energy Tracking (Optional)

#### 11-14. Track Energy Consumption
```bash
# Baseline training energy
python energy_tracking_base.py

# Optimized training energy
python energy_tracking_base_opt.py

# Baseline CL energy
python energy_tracking_continual.py

# Optimized CL energy
python energy_tracking_continual_opt.py
```

**What it does:** Wraps training/CL scripts with CodeCarbon tracker to measure kWh and CO₂  
**Output:** `src/results/energy_tracking/*.json`

> **🔬 Research Novelty #4:** Precise real-world energy measurements (kWh, CO₂) vs theoretical FLOPs

---

## Key Results Summary

### Initial Training

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Test RMSE | 0.713 | 0.737 | +3.4% |
| Time | 88 min | 75 min | -14.7% |
| Energy | 0.114 kWh | 0.095 kWh | **-16.5%** |

### Continual Learning (4 windows)

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Avg Forgetting | +1.12% | **-4.44%** | Improvement! |
| Total Time | 30 min | 6 min | **-80%** |
| Energy/Update | 0.003 kWh | 0.0005 kWh | **-83.3%** |

### Annual Lifecycle (52 weekly updates)

| Scenario | Energy | Savings |
|----------|--------|---------|
| Baseline (retrain weekly) | 5.928 kWh | - |
| Baseline CL | 0.270 kWh | -95.4% |
| **Optimized CL** | **0.120 kWh** | **-98.0%** |

---

## Core Script Reference

### Data Preparation
- `extract_dataset.py` - Extract `.tzst` archives to CSVs
- `split_pre_covid_dataset.py` - Create temporal splits
- `validate_splits.py` - Check data quality
- `analyze_dataset_quality.py` - Dataset statistics

### Graph Construction
- `graph_construction.py` - Build K-NN spatial graph from household features
- `graph_validation.py` - Validate graph properties

### Core Components
- `model_stgnn.py` - STGNN architecture (GCN + GRU)
- `data_preprocessing.py` - Create PyTorch DataLoaders with sliding windows
- `utils/config.py` - Centralized hyperparameter config

### Training
- `train.py` - Baseline training (88 min, 0.114 kWh)
- `train_optimized.py` - Energy-optimized training (**-16.5% energy**, gradient accumulation 4x)

### Continual Learning
- `continual_learning.py` - Baseline CL (3 epochs, 30 min)
- `energy_optimized_continual_learning.py` - Ultra-aggressive CL (**-83.3% energy**, 1 epoch, 6 min)

### Energy Tracking
- `energy_tracking_base.py` - Track baseline training energy
- `energy_tracking_base_opt.py` - Track optimized training energy
- `energy_tracking_continual.py` - Track baseline CL energy
- `energy_tracking_continual_opt.py` - Track optimized CL energy

### Evaluation
- `evaluate.py` - Comprehensive metrics and visualizations

---

## Research Contributions

1. **Energy-Efficient Sparse GNN Training** - 16.5% savings via gradient accumulation when mixed precision unavailable
2. **Negative Forgetting via Aggressive Optimization** - Single-epoch high-LR training improves stability (-4.44% vs +1.12%)
3. **Compounding Lifecycle Savings** - 55.7% annual savings with 52 weekly CL updates
4. **Real-World Energy Characterization** - 0.0005 kWh per CL update (30 seconds @ smartphone power)
5. **Temporary Accuracy Loss Recovery** - 3.4% initial gap recovers to 0.14% after CL

---

## Quick Commands

```bash
# Minimal baseline pipeline
cd src/
python graph_construction.py
python train.py
python continual_learning.py

# Energy-optimized pipeline
python train_optimized.py
python energy_optimized_continual_learning.py

# With energy tracking
python energy_tracking_base_opt.py
python energy_tracking_continual_opt.py
```

---

## Troubleshooting

**Missing adjacency matrix:** Run `python graph_construction.py`  
**Missing model file:** Run `python train.py` or `python train_optimized.py`  
**Out of memory:** Batch size already at minimum (1), use CPU or close other apps  
**CodeCarbon not tracking:** Verify installation `pip show codecarbon`, requires NVIDIA drivers for GPU tracking
