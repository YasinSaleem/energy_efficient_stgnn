# Quick Start Guide: Next Training Run

## What Was Changed

Based on your training results showing severe overfitting (validation RMSE: 0.525 → 0.651), I've implemented comprehensive regularization improvements:

### Key Changes:
1. ✅ **Learning Rate Scheduler** - Automatically reduces LR when validation plateaus
2. ✅ **Enhanced Dropout** - Increased from 0.1 to 0.3 across all layers (spatial, temporal, final)
3. ✅ **Gradient Clipping** - Prevents exploding gradients (max_norm=1.0)
4. ✅ **Stronger Weight Decay** - Increased from 1e-4 to 5e-4
5. ✅ **Faster Early Stopping** - Reduced patience from 6 to 3 epochs
6. ✅ **Centralized Configuration** - All settings in `utils/config.py`

---

## Files Modified

1. **`utils/config.py`** - NEW: Centralized hyperparameter configuration
2. **`src/model_stgnn.py`** - Enhanced with temporal dropout and config integration
3. **`src/train.py`** - Added scheduler, gradient clipping, uses config
4. **`TRAINING_DIAGNOSIS.md`** - Detailed analysis of overfitting issues
5. **`IMPLEMENTATION_SUMMARY.md`** - Complete implementation details

---

## Before Running Training

### Step 1: Verify Configuration
```bash
cd /Users/yasinsaleem/CourseWork/energy_efficient_stgnn
python3 utils/config.py
```

**Expected**: Should print configuration and validation success message.

### Step 2: Test Model Build (Optional)
```bash
python3 src/model_stgnn.py
```

**Expected**: Should load adjacency matrix and show input/output shapes.

### Step 3: Start Training
```bash
python3 src/train.py
```

---

## What to Expect During Training

### Configuration Display
Training will start by showing:
```
================================================================================
TRAINING CONFIGURATION
================================================================================
  Max Epochs:              20
  Learning Rate:           0.001
  Weight Decay:            0.0005
  Gradient Clip Norm:      1.0
  Early Stopping Patience: 3
  
  Spatial Dropout:         0.3
  Temporal Dropout:        0.3
  Final Dropout:           0.3
  
  Scheduler Factor:        0.5
  Scheduler Patience:      2
================================================================================
```

### Training Progress
Each epoch will show:
```
========== EPOCH 1/20 ==========
Train MSE: 0.380000
Val   MSE: 0.260000
Val   RMSE: 0.510000
Val   MAE: 0.082000
Current LR: 1.00e-03
💾 Best Model Saved → .../models/stgnn_best.pt
```

### Learning Rate Reductions
When validation plateaus, you'll see:
```
Epoch 00003: reducing learning rate of group 0 to 5.0000e-04.
```

### Early Stopping
Training will stop automatically:
```
⏳ EarlyStopping Counter: 3/3
🛑 Early stopping triggered to save energy.
```

---

## Expected Results

### Baseline (Your Previous Run):
- Epoch 1 Val RMSE: **0.525** ← Best
- Final Val RMSE: 0.651 (degraded)
- Test RMSE: **0.655**
- Generalization Gap: **24.7%**
- Stopped at: Epoch 7

### Improved (Expected):
- Best Val RMSE: **0.45-0.50** (↓ 5-15%)
- Test RMSE: **0.50-0.55** (↓ 16-23%)
- Generalization Gap: **<15%** (↓ ~10%)
- Stops at: Epoch 8-12
- **Smoother convergence** (no validation degradation)

---

## Monitoring Checklist

### ✅ Good Signs:
- [ ] Epoch 1 train MSE: 0.35-0.45 (slightly higher than baseline due to dropout)
- [ ] Validation RMSE steadily decreases or stays stable
- [ ] LR reduces when validation plateaus (around epoch 3-4)
- [ ] Gap between train/val MSE stays <20%
- [ ] Early stopping triggers between epochs 8-12

### ⚠️ Warning Signs:
- [ ] Train MSE stuck >0.45 after epoch 3 → Dropout may be too high
- [ ] Val RMSE increases immediately → Check data loading
- [ ] LR reduces every epoch → Increase scheduler patience
- [ ] Out of memory error → Verify batch size still 4

---

## If Results Don't Meet Expectations

### Still Overfitting (Val RMSE increases):
Edit `utils/config.py`:
```python
SPATIAL_DROPOUT = 0.4      # Increase from 0.3
TEMPORAL_DROPOUT = 0.4     # Increase from 0.3
WEIGHT_DECAY = 1e-3        # Increase from 5e-4
```

### Underfitting (Train MSE stuck high):
Edit `utils/config.py`:
```python
SPATIAL_DROPOUT = 0.2      # Decrease from 0.3
TEMPORAL_DROPOUT = 0.2     # Decrease from 0.3
GRU_HIDDEN = 48            # Increase from 32
```

### Slow Convergence:
Edit `utils/config.py`:
```python
LEARNING_RATE = 2e-3       # Increase from 1e-3
SCHEDULER_PATIENCE = 3     # Increase from 2
```

---

## After Training Completes

### 1. Check Final Metrics
Training will automatically evaluate on test set:
```
========== FINAL TEST RESULTS ==========
Test MSE : 0.250000
Test RMSE: 0.500000
Test MAE : 0.095000
=========================================
```

### 2. Compare to Baseline
Create a comparison:
```
Metric          | Baseline | Improved | Change
----------------|----------|----------|---------
Val RMSE (best) | 0.525    | 0.4XX    | -XX%
Test RMSE       | 0.655    | 0.5XX    | -XX%
Gen. Gap        | 24.7%    | XX%      | -XX%
Epochs          | 7        | XX       | +XX
```

### 3. Next Steps
- If good: Update other scripts (`continual_learning.py`, `evaluate.py`)
- If needs tuning: Adjust hyperparameters as shown above
- Document final hyperparameters for paper/report

---

## Rollback Instructions

If something goes wrong, revert to original values in `utils/config.py`:
```python
EPOCHS = 30
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6
SPATIAL_DROPOUT = 0.1
TEMPORAL_DROPOUT = 0.0
FINAL_DROPOUT = 0.1
# Remove scheduler by commenting out scheduler lines in train.py
# Remove gradient clipping by commenting out clip_grad_norm_ line
```

---

## Support Files

- **`TRAINING_DIAGNOSIS.md`** - Full technical analysis of overfitting
- **`IMPLEMENTATION_SUMMARY.md`** - Detailed implementation documentation
- **`utils/config.py`** - All hyperparameters in one place
- **`src/MODEL_PIPELINE.md`** - Original pipeline documentation

---

## Summary

**You're ready to run training!** The improvements are conservative and follow best practices. Expected outcome is significantly better generalization with similar or better validation performance.

**Command to run**:
```bash
python3 src/train.py
```

Good luck! 🚀
