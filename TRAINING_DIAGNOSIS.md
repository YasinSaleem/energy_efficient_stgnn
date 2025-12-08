# Training Diagnosis & Improvement Plan

## Executive Summary

**Issue**: Severe overfitting after epoch 1, with validation RMSE degrading from 0.525 to 0.651 while training MSE improved from 0.398 to 0.162.

**Root Cause**: Insufficient regularization given the model's capacity (8,442 nodes, large GRU hidden states) and relatively small effective batch size (B=4).

**Generalization Gap**: Test RMSE (0.655) is 24.7% worse than best validation RMSE (0.525).

---

## Detailed Analysis

### Current Configuration Issues

#### 1. **Insufficient Dropout** ⚠️
- **Current**: `dropout=0.1` in `SpatialGCNLayer`, but `dropout=0.0` in GRU
- **Problem**: GRU has 32 hidden units processing 8,442 nodes → massive parameter count with no temporal regularization
- **Impact**: Model memorizes training sequences instead of learning generalizable patterns

#### 2. **No Learning Rate Scheduling** ⚠️
- **Current**: Fixed LR=1e-3 throughout training
- **Problem**: Large learning rate continues to push model into overfitting territory after initial convergence
- **Impact**: Model cannot settle into a generalizable minimum

#### 3. **Small Batch Size** ⚠️
- **Current**: `BATCH_SIZE=4` (RTX 4050 safe)
- **Problem**: Very small effective gradient estimates → noisy updates
- **Impact**: Harder to find stable generalizable solutions
- **Note**: Cannot increase due to GPU memory constraints

#### 4. **No Gradient Clipping** ⚠️
- **Current**: No clipping implemented
- **Problem**: GRU can experience gradient explosion, especially with sparse graph signals
- **Impact**: Training instability and poor generalization

#### 5. **Weak Weight Decay** ⚠️
- **Current**: `WEIGHT_DECAY=1e-4`
- **Problem**: Too weak for a model with this many parameters on spatial-temporal data
- **Impact**: Insufficient L2 regularization to prevent overfitting

#### 6. **Early Stopping Patience Too High** ⚠️
- **Current**: `EARLY_STOPPING_PATIENCE=6`
- **Problem**: Allows 6 epochs of degradation before stopping
- **Impact**: Wastes energy and risks saving a worse model if validation temporarily improves

---

## Recommended Improvements

### Priority 1: Critical (Must Implement)

#### 1.1 **Add Learning Rate Scheduler**
```python
# ReduceLROnPlateau: reduce LR when validation stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,        # Halve LR
    patience=2,        # After 2 epochs without improvement
    min_lr=1e-6,
    verbose=True
)
```
**Expected Impact**: Prevents overfitting by reducing learning rate after epoch 1-2, allowing model to converge to better minimum.

#### 1.2 **Enable GRU Dropout**
```python
# In model_stgnn.py STGNNModel.__init__()
self.gru = nn.GRU(
    input_size=gcn_hidden,
    hidden_size=gru_hidden,
    num_layers=1,
    batch_first=True,
    dropout=0.3  # ← CHANGED from 0.0
)
```
**Expected Impact**: 15-20% reduction in overfitting, improved test performance.

#### 1.3 **Add Gradient Clipping**
```python
# In train_one_epoch()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
**Expected Impact**: Stabilize training, prevent gradient explosions, smoother convergence.

#### 1.4 **Increase Spatial Dropout**
```python
# In build_stgnn()
model = STGNNModel(
    ...
    dropout=0.3,  # ← CHANGED from 0.1
)
```
**Expected Impact**: Better spatial feature regularization.

#### 1.5 **Reduce Early Stopping Patience**
```python
EARLY_STOPPING_PATIENCE = 3  # ← CHANGED from 6
```
**Expected Impact**: Save energy, stop faster when overfitting begins.

---

### Priority 2: Important (Should Implement)

#### 2.1 **Increase Weight Decay**
```python
WEIGHT_DECAY = 5e-4  # ← CHANGED from 1e-4
```
**Expected Impact**: Stronger L2 regularization, smoother parameter distributions.

#### 2.2 **Adjust Max Epochs**
```python
EPOCHS = 20  # ← CHANGED from 30
```
**Expected Impact**: Realistic upper bound given early stopping at epoch 7.

#### 2.3 **Add Batch Normalization Option**
```python
# In SpatialGCNLayer (optional enhancement)
self.bn = nn.BatchNorm1d(out_features) if use_bn else None
```
**Expected Impact**: Normalize activations, potentially improve convergence.

---

### Priority 3: Optional (Nice to Have)

#### 3.1 **Data Augmentation**
- Add Gaussian noise to input features during training
- Temporal masking (randomly mask some time steps)

#### 3.2 **Layer Freezing Strategy**
- Freeze first GCN layer after initial epochs
- Only fine-tune temporal components

#### 3.3 **Warmup Learning Rate**
- Start with lower LR for first 2 epochs
- Gradually increase to target LR

---

## Implementation Strategy

### Phase 1: Core Regularization (This Run)
1. Add learning rate scheduler (ReduceLROnPlateau)
2. Enable GRU dropout (0.3)
3. Increase spatial dropout (0.1 → 0.3)
4. Add gradient clipping (max_norm=1.0)
5. Reduce early stopping patience (6 → 3)
6. Increase weight decay (1e-4 → 5e-4)

### Phase 2: Validation (Next Run)
1. Monitor training curves for improvements
2. Compare test RMSE against baseline (0.655)
3. Check generalization gap (target: <10%)
4. Verify energy efficiency (should stop earlier)

### Phase 3: Fine-tuning (If Needed)
1. Adjust dropout rates if still overfitting
2. Try different scheduler strategies (CosineAnnealing, StepLR)
3. Experiment with different weight decay values

---

## Expected Outcomes

### Conservative Estimates
- **Validation RMSE**: 0.45-0.50 (vs. current 0.525)
- **Test RMSE**: 0.50-0.55 (vs. current 0.655)
- **Generalization Gap**: 8-12% (vs. current 24.7%)
- **Training Epochs**: 8-12 (vs. current 7)
- **Energy Savings**: 20-30% fewer epochs to convergence

### Success Metrics
✅ Test RMSE < 0.55  
✅ Generalization gap < 15%  
✅ Validation loss does not degrade after epoch 3  
✅ Model stops naturally via early stopping (not max epochs)  

---

## Risk Assessment

### Low Risk Changes ✅
- Learning rate scheduler (standard practice)
- Gradient clipping (prevents instability)
- Early stopping patience adjustment
- Weight decay increase (moderate)

### Medium Risk Changes ⚠️
- Dropout increases (might hurt initial learning)
  - Mitigation: Start with 0.2, increase to 0.3 if needed
- GRU dropout (can slow convergence)
  - Mitigation: Monitor epoch 1-3 performance

### Not Recommended ❌
- Changing model architecture (GCN/GRU hidden dims)
- Modifying batch size (GPU memory constrained)
- Removing weight decay
- Switching to different optimizer

---

## Configuration File Centralization

Create `utils/config.py` to centralize all hyperparameters for easier experimentation:

```python
# Training
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CLIP_NORM = 1.0

# Model Architecture
GCN_HIDDEN = 16
GCN_LAYERS = 1
GRU_HIDDEN = 32
SPATIAL_DROPOUT = 0.3
TEMPORAL_DROPOUT = 0.3
HORIZON = 6

# Data
BATCH_SIZE = 4
WINDOW_SIZE = 24
NUM_WORKERS = 0

# Scheduler
SCHEDULER_MODE = 'min'
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 1e-6
```

This allows version control of hyperparameters and easier experimental tracking.

---

## Conclusion

The current setup is fundamentally sound but lacks sufficient regularization for the model's capacity. The proposed changes are conservative, well-established techniques that should significantly improve generalization without radical restructuring. Implementation is low-risk and should yield measurable improvements in the next training run.
