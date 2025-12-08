# Detailed Script Overview & Next-Run Improvements

This document provides a comprehensive technical explanation of **what each script does**, how it integrates into the overall pipeline, and what **improvements** should be targeted in the next experimental run. Each section details the methodology, rationale, and opportunities for enhancement.

---

## 0. Early Stage Scripts (Already Completed)

### `extract_dataset.py`

#### What it does

This script performs the initial data extraction from compressed archives into a structured format suitable for analysis.

**Technical Process:**
- Decompresses `.tzst` (Zstandard) archives using streaming decompression to minimize memory overhead
- Extracts **one CSV file per household** where:
  - Filename corresponds to an anonymized user hash (ensures privacy)
  - Columns include: `timestamp`, `kWh`, and optionally `imputed` (indicates whether data was imputed during preprocessing)
- Organizes extracted files into time-based directories:
  - `data/extracted/imp-pre/` (pre-COVID period)
  - `data/extracted/imp-in/` (during COVID period)
  - `data/extracted/imp-post/` (post-COVID period)

#### How it helps

The file-per-household structure provides several critical advantages:

- **Scalability:** Individual household files can be loaded independently, reducing memory requirements
- **Flexibility:** Easy to create custom train/validation/test splits by selecting subsets of household files
- **Parallelization:** Multiple households can be processed simultaneously in parallel workflows
- **Maintainability:** Corrupted or problematic household data can be identified and handled in isolation

#### Next-run improvements

**Data Integrity Validation:**
- Implement timestamp validation to ensure strict monotonic increasing order within each household
- Verify hourly spacing consistency (detect and flag duplicate or missing timestamps)
- Add checksums or hash verification for extracted files

**Statistical Logging:**
- Generate per-archive summary statistics including:
  - Date range (min/max timestamps)
  - Average hourly consumption
  - Percentage of missing or imputed values
  - Household count per archive

**Storage Optimization:**
- Consider converting extracted CSVs to `.parquet` format for:
  - Faster I/O operations (columnar storage)
  - Better compression ratios
  - Type safety and schema enforcement

---

### `split_pre_covid_dataset.py` & `create_continual_splits.py`

#### What they do

These scripts partition the dataset into training, validation, testing, and continual learning subsets based on temporal characteristics.

**Base Splits Creation:**
- `data/splits/train/` - Training set for initial model development
- `data/splits/val/` - Validation set for hyperparameter tuning and early stopping
- `data/splits/test/` - Test set for final performance evaluation

**Continual Learning Windows:**
- `data/splits/continual/CL_1/` through `CL_4/` - Sequential time windows representing distribution shifts

**Key Characteristics:**
- Each split/window contains the **same set of household IDs** but covers **different time ranges**
- This ensures the model is evaluated on consistent households across temporal shifts
- Splits are created based on metadata and timestamp availability

#### How it helps

**Time-Aware Partitioning:**
- The base model learns patterns from a stable historical period
- Continual learning windows represent realistic distribution shifts such as:
  - Seasonal variations (summer vs. winter consumption patterns)
  - Behavioral changes (work-from-home during COVID)
  - Economic factors (energy price fluctuations)

**Catastrophic Forgetting Evaluation:**
- By maintaining consistent household IDs across splits, we can measure:
  - How well the model adapts to new temporal patterns
  - Whether adaptation causes degradation on previously learned patterns

#### Next-run improvements

**Configurable Date Ranges:**
- Make splitting logic configurable via a YAML or JSON configuration file
- Target specific calendar periods (e.g., "pre-COVID: 2018-01 to 2020-02")
- Support flexible window sizes and overlap parameters

**Split Metadata Documentation:**
- Store comprehensive metadata for each split in JSON format:
  ```json
  {
    "split_name": "train",
    "date_range": {"start": "2018-01-01", "end": "2019-12-31"},
    "num_households": 8442,
    "num_hours": 17520,
    "avg_consumption_kwh": 0.543
  }
  ```

**Stratified Splitting:**
- Implement province-wise or tariff-wise splits to study:
  - Geographic heterogeneity in consumption patterns
  - Impact of different tariff structures on model performance
  - Regional distribution shift characteristics

---

## 1. Spatial Graph Construction

### `graph_construction.py`

#### What it does

This script constructs the spatial graph that encodes relationships between households, enabling the spatio-temporal graph neural network to leverage spatial dependencies.

**Step-by-Step Process:**

**1. Metadata Loading**
- Reads `data/raw/7362094/metadata.csv` containing household characteristics
- Filters to retain only households present in the training split (ensures graph corresponds to actual training data)

**2. Node Mapping**
- Creates a bijective mapping from household identifiers to sequential node indices:
  ```
  household_hash → node_index (0, 1, ..., N-1)
  ```
- Saved as `node_map.json` for consistent referencing across all scripts

**3. Feature Extraction**

**Geographic Features:**
- One-hot encodes `province` and `municipality` fields
- Captures spatial proximity and regional regulatory differences

**Contracted Power Features:**
- Extracts power capacity values `p1` through `p6` (contracted power limits for different tariff periods)
- Normalizes using `StandardScaler` to ensure comparable scales
- Reflects household size and consumption capacity

**Tariff Type Features:**
- One-hot encodes `contracted_tariff` (e.g., 2.0A, 2.0DHA, 2.1A, etc.)
- Captures pricing structure and consumption patterns associated with different tariff types

**4. Similarity Computation**

Computes pairwise similarity between households using multiple metrics:

**Geographic Similarity:**
- Cosine similarity of one-hot encoded geographic features
- High similarity indicates households in the same region

**Power Similarity:**
- Distance-based similarity on normalized contracted power values
- Measures similarity in consumption capacity

**Tariff Similarity:**
- Cosine similarity of tariff type encodings
- Captures shared pricing structures

**Combined Similarity:**
```
similarity = 0.5 × geo_similarity + 0.3 × power_similarity + 0.2 × tariff_similarity
```

These weights reflect the relative importance of each factor in determining household relationships.

**5. K-Nearest Neighbors (K-NN) Graph Construction**

For each household node:
- Selects the top-K most similar neighbors based on combined similarity
- Creates directed edges from the node to its K neighbors

**Graph Post-Processing:**
- Symmetrization: If node A is connected to B, ensure B is also connected to A
- Self-loops: Add self-connections to allow nodes to retain their own information
- Degree normalization: Normalize adjacency matrix to prevent numerical instability during message passing

**6. Output Files**

Saves to `data/processed/`:
- `adjacency_matrix.npz` - Sparse adjacency matrix in CSR format
- `node_map.json` - Household ID to node index mapping
- `graph_stats.json` - Graph statistics (number of nodes, edges, average degree, sparsity)

#### How it helps

**Spatial Inductive Bias:**
- Encodes the assumption that similar households have related consumption patterns
- Enables the STGNN to perform message passing, where each node aggregates information from its neighbors

**Graph-Based Regularization:**
- Similar households influence each other's predictions, providing a form of spatial smoothing
- Reduces overfitting by sharing information across structurally similar nodes

**Interpretability:**
- The graph structure can be analyzed to understand which household characteristics drive similarity
- Enables visualization of household clusters and communities

#### Next-run improvements

**Physical Distance Integration:**
- If geographic coordinates (latitude/longitude) become available:
  - Compute Haversine distance between households
  - Incorporate physical proximity into similarity metric
  - Model spatial autocorrelation more accurately

**Multi-Graph Architecture:**
- Construct separate graphs for different similarity types:
  - Geographic graph (province/municipality similarity)
  - Tariff graph (pricing structure similarity)
  - Capacity graph (contracted power similarity)
- Learn separate graph convolutions for each graph and fuse representations

**Adaptive Graph Learning:**
- Learn edge weights during training instead of using fixed K-NN
- Use attention mechanisms to dynamically weight neighbor contributions

**Sparsity Optimization:**
- Experiment with different values of K (e.g., K=5, 10, 20)
- Consider radius-based graphs instead of K-NN (connect all nodes within similarity threshold)
- Evaluate trade-off between graph density and computational cost

---

### `graph_validation.py`

#### What it does

This script performs comprehensive validation of the constructed spatial graph to ensure correctness before model training.

**Validation Checks:**

**1. Structural Properties:**
- Matrix shape verification (should be N × N square matrix)
- Node count consistency (matches `node_map.json`)
- Edge count and sparsity calculation

**2. Mathematical Properties:**
- Symmetry check: Verifies A ≈ A^T (graph should be undirected)
- Self-loop presence: Confirms diagonal elements exist
- Positive edge weights: Ensures all edge weights are non-negative

**3. Statistical Analysis:**
- Degree distribution (min, mean, median, max, standard deviation)
- Edge weight statistics (min, mean, max)
- Connected components analysis (checks if graph is fully connected)

**4. Output:**
Prints a detailed summary report including:
```
Graph Validation Report
=======================
Nodes: 8442
Edges: 126630
Average Degree: 15.0
Sparsity: 99.82%
Symmetry Error: 1.2e-10
...
```

#### How it helps

**Early Error Detection:**
- Catches implementation bugs before expensive training runs
- Identifies data quality issues (e.g., missing household metadata)

**Graph Quality Assessment:**
- Extremely high sparsity (>99.9%) may indicate insufficient connectivity
- Very high average degree may cause computational bottlenecks
- Asymmetric graphs may violate GCN assumptions

**Reproducibility:**
- Documents graph properties for comparison across experiments
- Enables debugging when model performance is unexpectedly poor

#### Next-run improvements

**Spectral Analysis:**
- Compute eigenvalues of the normalized Laplacian matrix
- Check for zero eigenvalues indicating disconnected components
- Analyze spectral gap for community structure

**Connectivity Analysis:**
- Identify isolated nodes (degree = 0) and flag for manual review
- Compute connected components and ensure most nodes are in the largest component
- Implement fallback connection strategy for isolated nodes (e.g., connect to province centroid)

**Visualization:**
- Generate graph visualization for small subgraphs
- Create degree distribution histograms
- Plot similarity matrix heatmaps for random subsets

---

## 2. Temporal Data Pipeline

### `data_preprocessing.py`

#### What it does

This script transforms raw per-household time series data into structured tensors suitable for spatio-temporal graph neural network training.

**Detailed Pipeline:**

**1. Node Map Loading**
- Loads `node_map.json` to ensure consistent column ordering
- All data tensors will have shape `[..., N, ...]` where N is ordered by node indices

**2. Panel Construction**

For each data split (train, validation, test):

**Timestamp Alignment:**
- Reads all household CSV files in the split directory
- Identifies the intersection of timestamps across all households
- Retains only timestamps present in **all** household files to ensure complete data

**Panel Assembly:**
- Constructs a 2D matrix of shape `[time_steps, num_nodes]`
- Each column represents one household's consumption time series
- Missing values at this stage would indicate data quality issues

**3. Normalization**

**Scaler Fitting:**
- Fits `sklearn.preprocessing.StandardScaler` **exclusively on training data**
- Computes mean (μ) and standard deviation (σ) per node
- Saves scaler to `data/processed/scaler_stgnn.pkl` for later use

**Scaler Application:**
- Transforms training data: `X_train_scaled = (X_train - μ) / σ`
- Applies same transformation to validation, test, and continual learning splits
- This prevents data leakage and ensures consistent scaling

**4. Sliding Window Creation**

Converts panels into supervised learning samples:

**Window Parameters:**
- Input window size: 24 hours (past consumption)
- Forecast horizon: 6 hours (future consumption to predict)
- Stride: Typically 1 hour (can be adjusted for computational efficiency)

**Tensor Shapes:**
- Input X: `[num_samples, 24, N, 1]`
  - `num_samples`: Number of sliding windows
  - `24`: Input sequence length
  - `N`: Number of household nodes (8442)
  - `1`: Feature dimension (kWh consumption)
- Output Y: `[num_samples, 6, N]`
  - `6`: Forecast horizon length

**5. DataLoader Creation**

**Memory-Efficient Batching:**
- Uses PyTorch `DataLoader` for mini-batch iteration
- Enables GPU training without loading entire dataset into memory
- Supports multi-worker data loading for I/O parallelization

**Exposed Functions:**
- `get_base_dataloaders()` → Returns train, validation, test loaders
- `get_cl_dataloaders()` → Returns dictionary of continual learning loaders

#### How it helps

**Model-Ready Data:**
- Converts raw CSV files into tensors compatible with PyTorch models
- Handles all data alignment, scaling, and windowing in a centralized script

**Consistent Preprocessing:**
- All splits use the same scaler fitted on training data
- Prevents subtle bugs from inconsistent preprocessing across splits

**Efficient Training:**
- DataLoaders enable mini-batch gradient descent
- Automatic batching and shuffling for each epoch
- GPU memory management through on-demand loading

#### Next-run improvements

**Exogenous Features:**
- Incorporate external variables as additional feature channels:
  - **Temperature:** Historical weather data per province
  - **Calendar features:** Day of week, month, holidays
  - **Time-of-use indicators:** Peak vs. off-peak hours
- Expand feature dimension from 1 to F (e.g., F=5 for multiple features)

**Per-Node Normalization:**
- Implement node-specific normalization: `(x_i - μ_i) / σ_i`
- Accounts for heterogeneous consumption distributions across households
- May improve model performance when household behaviors vary significantly

**Imputation Flag as Feature:**
- If imputation was performed during extraction, include binary flag as second channel
- Model can learn to weight imputed vs. observed values differently

**Variable Horizon Outputs:**
- Support multi-horizon training (1h, 3h, 6h, 12h, 24h)
- Enables analysis of how prediction quality degrades with forecast distance

---

## 3. Model Architecture

### `model_stgnn.py`

#### What it does

Implements the Spatio-Temporal Graph Neural Network (STGNN) architecture that jointly models spatial dependencies between households and temporal dynamics of consumption patterns.

**Architecture Components:**

**1. Input Layer**

Accepts tensors of shape:
```
[batch_size, T_in=24, N_nodes=8442, F_in=1]
```

Where:
- `batch_size`: Number of samples in mini-batch
- `T_in`: Input sequence length (24 hours)
- `N_nodes`: Number of household nodes
- `F_in`: Feature dimension (1 for kWh only)

**2. Spatial Module (Graph Convolution)**

For each time step t ∈ {1, ..., 24}:

**Graph Convolution Operation:**
```
X_t' = σ(A_norm · X_t · W + b)
```

Where:
- `A_norm`: Normalized adjacency matrix (degree-normalized)
- `X_t`: Node features at time t, shape `[N, F_in]`
- `W`: Learnable weight matrix, shape `[F_in, hidden_dim]`
- `b`: Learnable bias vector, shape `[hidden_dim]`
- `σ`: Activation function (typically ReLU)

**Spatial Message Passing:**
- Each node aggregates features from its K neighbors weighted by edge weights
- Captures spatial dependencies: similar households influence each other
- Lifts feature dimension from `F_in=1` to `hidden_dim` (e.g., 16 or 32)

**3. Temporal Module**

After spatial encoding, the sequence becomes:
```
[batch_size, T_in=24, N, hidden_dim]
```

**Gated Recurrent Unit (GRU):**
- Processes the spatially-encoded sequence temporally
- Captures temporal dependencies within each node's time series
- Architecture: `GRU(input_size=hidden_dim, hidden_size=hidden_dim*2)`

**Why GRU?**
- Handles variable-length sequences efficiently
- Mitigates vanishing gradients through gating mechanisms
- Lighter than LSTM, reducing computational cost

**Alternative: Temporal Convolution:**
- Could use 1D CNN over time dimension
- Trade-off: CNNs are faster but may miss long-range temporal patterns

**4. Forecast Head**

Maps GRU hidden states to multi-step forecasts:

**Linear Projection:**
```
Y_hat = Linear(h_final, N × H_out)
```

Where:
- `h_final`: Final hidden state from GRU, shape `[batch_size, N, hidden_dim*2]`
- `H_out`: Forecast horizon (6 hours)

**Output Shape:**
```
Y_hat: [batch_size, 6, N]
```

Each sample produces 6-hour ahead forecasts for all N households simultaneously.

**5. Device Management**

**GPU Acceleration:**
- Automatically detects CUDA availability
- Moves adjacency matrix and model parameters to GPU
- Supports both CPU and CUDA execution for flexibility

#### How it helps

**Spatial Inductive Bias:**
- Graph convolution encodes the assumption that neighboring households have correlated consumption
- Reduces parameter count compared to fully connected layers (exploits sparsity)

**Temporal Pattern Learning:**
- GRU captures daily cycles, weekly patterns, and trends
- Handles sequential dependencies that simple feed-forward networks miss

**Multi-Step Forecasting:**
- Predicts entire 6-hour horizon in one forward pass
- More efficient than auto-regressive prediction (predicting one step at a time)

#### Next-run improvements

**Deeper Architecture:**
- Stack multiple ST-blocks: `[Spatial → Temporal] × L layers`
- Add residual connections to enable gradient flow: `X_out = X_in + ST-Block(X_in)`
- Increases model capacity to learn complex patterns

**Attention Mechanisms:**

**Spatial Attention:**
- Learn dynamic edge weights instead of fixed K-NN adjacency
- Allows model to focus on most relevant neighbors per prediction

**Temporal Attention:**
- Weight different time steps by importance
- Useful when recent hours are more predictive than distant hours

**Multi-Head Attention:**
- Learn multiple attention patterns simultaneously
- Capture different aspects of spatio-temporal relationships

**Multi-Feature Input:**
- Extend `F_in` beyond 1 to include:
  - Imputation flags (binary)
  - Lagged statistics (rolling mean, std)
  - Exogenous variables (temperature, holidays)
- Requires adjusting first layer weight dimensions

---

## 4. Training Script

### `train.py`

#### What it does

Orchestrates the complete model training pipeline from initialization to final evaluation.

**Training Pipeline:**

**1. Device Configuration**
- Detects CUDA-capable GPU (e.g., RTX 4050)
- Falls back to CPU if GPU unavailable
- Sets `torch.backends.cudnn.benchmark=True` for performance optimization

**2. Data Loading**
- Calls `get_base_dataloaders()` from `data_preprocessing.py`
- Obtains train, validation, and test DataLoaders
- Verifies data shapes and batch sizes

**3. Model Initialization**
- Calls `build_stgnn()` to construct model with adjacency matrix
- Moves model to detected device (GPU or CPU)
- Prints model architecture and parameter count

**4. Optimizer and Loss Configuration**

**Optimizer:**
- Uses Adam optimizer: `torch.optim.Adam(lr=1e-3)`
- Adam combines benefits of momentum and adaptive learning rates
- Suitable for sparse gradients in graph neural networks

**Loss Function:**
- Mean Squared Error (MSE) between predictions and ground truth
- MSE = `(1/N) Σ (y_pred - y_true)²`
- Penalizes large errors more heavily than small ones

**5. Training Loop**

For each epoch:

**Training Phase:**
```python
for batch_X, batch_Y in train_loader:
    optimizer.zero_grad()
    predictions = model(batch_X)
    loss = mse_loss(predictions, batch_Y)
    loss.backward()
    optimizer.step()
```

**Validation Phase:**
```python
with torch.no_grad():
    for batch_X, batch_Y in val_loader:
        predictions = model(batch_X)
        val_loss += mse_loss(predictions, batch_Y)
```

**Metrics Computed:**
- Training MSE (loss on training set)
- Validation MSE, RMSE, MAE (evaluation on validation set)

**6. Early Stopping**

**Mechanism:**
- Monitors validation RMSE after each epoch
- If RMSE does not improve for `patience` epochs (e.g., 6), stop training
- Prevents overfitting by halting when generalization performance plateaus

**Why RMSE instead of MSE?**
- RMSE is in the same units as the target (kWh)
- More interpretable for setting thresholds

**7. Checkpointing**

After each epoch:
- If validation RMSE improved:
  - Save model state: `torch.save(model.state_dict(), 'src/models/stgnn_best.pt')`
  - Update best RMSE tracker

**8. Final Test Evaluation**

After training completes:
- Load best model checkpoint
- Evaluate on held-out test set
- Print final MSE, RMSE, MAE on test data

#### How it helps

**Reproducible Baseline:**
- Provides a standard training procedure for all experiments
- Consistent hyperparameters enable fair comparisons

**Generalization:**
- Early stopping prevents overfitting to training data
- Validation-based checkpointing ensures best generalization is saved

**Efficiency:**
- GPU acceleration reduces training time from days to hours
- Progress bars (tqdm) provide real-time feedback

#### Next-run improvements

**Learning Rate Scheduling:**
- Implement `ReduceLROnPlateau`: Reduce learning rate when validation loss plateaus
- Example: `scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)`
- Helps model converge to better local minima

**Mixed-Precision Training:**
- Use `torch.cuda.amp` (Automatic Mixed Precision)
- Reduces memory usage and power consumption
- Speeds up training by using float16 where possible

**Hyperparameter Tuning:**
- Implement grid search or random search over:
  - Learning rate: {1e-4, 5e-4, 1e-3, 5e-3}
  - Hidden dimensions: {16, 32, 64}
  - Dropout rates: {0.0, 0.1, 0.2}
  - Batch size: {32, 64, 128}
- Use validation RMSE as objective
- Consider Bayesian optimization for efficiency

**Regularization:**
- Add L2 weight decay to optimizer
- Implement dropout layers in model
- Apply gradient clipping to prevent exploding gradients

---

## 5. Evaluation Script

### `evaluate.py`

#### What it does

Performs comprehensive evaluation of the trained model, computing multiple metrics and generating visualizations.

**Evaluation Pipeline:**

**1. Model Loading**
- Loads best checkpoint: `src/models/stgnn_best.pt`
- Verifies model architecture matches saved state
- Moves model to appropriate device

**2. Data Loading**
- Obtains validation and test DataLoaders
- Uses same preprocessing as training (via saved scaler)

**3. Prediction Generation**

For validation and test sets:
```python
with torch.no_grad():
    for batch_X, batch_Y in dataloader:
        predictions = model(batch_X)
        # Collect predictions and ground truth
```

**4. Metric Computation**

**Point Metrics:**
- **MSE (Mean Squared Error):** `(1/N) Σ (y_pred - y_true)²`
  - Sensitive to large errors
- **RMSE (Root Mean Squared Error):** `√MSE`
  - Same units as target (kWh)
- **MAE (Mean Absolute Error):** `(1/N) Σ |y_pred - y_true|`
  - More robust to outliers than MSE
- **MAPE (Mean Absolute Percentage Error):** `(100/N) Σ |y_pred - y_true| / |y_true|`
  - Scale-independent, useful for comparing across different households
- **R² (Coefficient of Determination):** `1 - (SS_res / SS_tot)`
  - Measures proportion of variance explained by model

**Per-Horizon Metrics:**
- Computes RMSE and MAE separately for each forecast step (h=1, 2, ..., 6)
- Reveals how prediction quality degrades with forecast distance

**5. Statistical Summary**
- Mean, standard deviation, min, max of predictions vs. ground truth
- Error distribution statistics (skewness, kurtosis)

**6. Visualization Generation**

**Sample Predictions Plot:**
- Selects 4-6 random households
- Plots predicted vs. actual consumption for each forecast horizon
- Includes confidence intervals if model provides uncertainty estimates

**Error Distribution:**
- Histogram of prediction errors
- Q-Q plot to check if errors are normally distributed
- Identifies systematic biases (e.g., consistent over-prediction)

**7. Output Files**

Saves to `results/evaluation/`:
- `evaluation_results.json` - All computed metrics
- `test_predictions.png` - Sample prediction plots
- `test_error_dist.png` - Error distribution visualization

#### How it helps

**Quantitative Performance Assessment:**
- Multiple metrics provide comprehensive view of model quality
- Per-horizon metrics identify weaknesses (e.g., poor long-term forecasting)

**Qualitative Analysis:**
- Visualizations reveal patterns not captured by aggregate metrics
- Error distributions indicate whether assumptions (e.g., Gaussian errors) hold

**Model Diagnostics:**
- Systematic errors (e.g., Q-Q plot deviations) suggest model improvements
- Performance variation across households indicates need for personalization

#### Next-run improvements

**Granular Analysis:**

**Per-Node Metrics:**
- Compute RMSE/MAE for each household
- Identify which households are hardest to predict
- Correlate prediction quality with household characteristics (tariff, province, consumption level)

**Per-Province Heatmaps:**
- Generate spatial heatmaps showing RMSE by province
- Reveals geographic patterns in prediction quality

**Temporal Analysis:**
- Evaluate performance separately for:
  - Peak vs. off-peak hours
  - Weekdays vs. weekends
  - Different seasons

**Peak Demand Errors:**
- Compute metrics specifically for high-consumption periods
- Peak errors have higher consequences for grid management

**Confidence Intervals:**
- If model provides uncertainty estimates, plot confidence intervals
- Assess calibration: do 95% intervals contain true values 95% of the time?

---

## 6. Continual Learning Baseline

### `continual_learning.py`

#### What it does

Implements a baseline continual learning evaluation to assess the model's ability to adapt to new data while retaining performance on old data.

**Continual Learning Setup:**

**1. Initialization**
- Loads base model: `src/models/stgnn_best.pt`
- Loads base DataLoaders (train, val, test)
- Loads continual learning DataLoaders: `CL_1, CL_2, CL_3, CL_4`

**2. Baseline Evaluation**
- Evaluates base model on original test set **before any updates**
- Records initial MSE, RMSE, MAE as baseline metrics

**3. Sequential Adaptation Loop**

For each continual learning window (CL_1 → CL_4):

**Fine-Tuning Phase:**
- Uses CL window data as new training set
- Fine-tunes model for a small number of epochs (e.g., 3)
- Uses same optimizer and loss as base training
- Learning rate may be reduced to avoid catastrophic forgetting

**Adaptation Evaluation:**
- Evaluates model on the **same CL window** it was fine-tuned on
- Measures: Can the model learn new patterns from this time period?

**Forgetting Evaluation:**
- Evaluates model on the **original test set** again
- Measures: Did adapting to new data degrade performance on old data?

**Forgetting Metrics:**
```
ΔRMSE = RMSE_after_CL - RMSE_baseline
Forgetting % = (ΔRMSE / RMSE_baseline) × 100
```

**4. Model Checkpointing**
- Saves updated model after each CL window:
  - `src/models/continual/stgnn_CL_1.pt`
  - `src/models/continual/stgnn_CL_2.pt`
  - ...

**5. Results Aggregation**
- Compiles all metrics into JSON:
  ```json
  {
    "baseline": {"test_rmse": 0.145, ...},
    "CL_1": {
      "new_data_rmse": 0.152,
      "old_test_rmse": 0.158,
      "forgetting_pct": 8.97,
      "time_seconds": 247.3
    },
    ...
  }
  ```
- Saves to: `src/results/continual_learning/continual_learning_results.json`

**6. Summary Report**

Prints per-window summary:
```
CL Window 1:
  New Data RMSE: 0.152
  Old Test RMSE: 0.158 (+8.97% forgetting)
  Update Time: 247.3 seconds
```

#### How it helps

**Realistic Evaluation:**
- Real-world models must adapt to distribution shifts (seasonality, behavior changes)
- Continual learning evaluation captures this dynamic setting

**Catastrophic Forgetting Quantification:**
- Measures the fundamental tension in continual learning:
  - Adaptation: Improve on new data
  - Retention: Maintain performance on old data
- Provides baseline for comparing continual learning strategies

**Temporal Generalization:**
- Tests whether model can generalize beyond its training time period
- Identifies temporal drift in consumption patterns

#### Next-run improvements

**Continual Learning Strategies:**

**Regularization-Based Methods:**
- **Elastic Weight Consolidation (EWC):**
  - Add penalty term to loss: `L = L_new + λ Σ F_i (θ_i - θ_i*)²`
  - `F_i`: Fisher information (importance of parameter θ_i for old tasks)
  - Protects important weights from large changes
  
- **Learning without Forgetting (LwF):**
  - Add distillation loss: `L = L_new + α KL(P_old || P_new)`
  - Forces new model predictions to match old model predictions on old data

**Replay-Based Methods:**
- **Experience Replay:**
  - Store small subset of old data in replay buffer
  - Mix old and new samples during CL updates
  - Directly prevents forgetting through rehearsal

- **Generative Replay:**
  - Train generative model to synthesize old-style data
  - Replay synthetic samples instead of storing real data
  - Addresses privacy and storage concerns

**Architecture-Based Methods:**
- **Progressive Neural Networks:**
  - Freeze old model weights
  - Add new parameters for new tasks
  - Prevents forgetting but increases model size

- **Parameter Freezing:**
  - Freeze spatial GCN layers (spatial structure is stable)
  - Fine-tune only temporal layers or forecast head
  - Reduces forgetting while maintaining adaptation

**Comparative Analysis:**
- Compare multiple approaches:
  - No CL (frozen base model)
  - Simple fine-tuning (current baseline)
  - EWC, LwF, Replay
- Evaluate trade-offs: adaptation quality vs. forgetting vs. computational cost

---

## 7. Energy Tracking

### `energy_tracking.py`

#### What it does

Wraps the continual learning pipeline with energy consumption monitoring to quantify the environmental impact of model training and adaptation.

**Energy Tracking Pipeline:**

**1. CodeCarbon Initialization**

Uses the CodeCarbon library to track:
- **Energy consumption (kWh):** Actual electricity used by hardware
- **CO₂ emissions (kg):** Estimated carbon footprint based on regional grid carbon intensity
- **Tracking scope:** Includes CPU, GPU, RAM

**Configuration:**
```python
tracker = EmissionsTracker(
    project_name="stgnn_continual_learning",
    output_dir="results/energy_tracking/codecarbon_logs/",
    measure_power_secs=15  # Sample power every 15 seconds
)
```

**2. Continual Learning Execution**

Starts energy tracker, then calls:
```python
tracker.start()
results = run_continual_learning()
emissions = tracker.stop()
```

The `run_continual_learning()` function:
- Loads base model
- Builds all DataLoaders
- Executes CL_1 → CL_4 fine-tuning sequentially
- Collects performance metrics (RMSE, forgetting)

**Important:** You do not need to run `continual_learning.py` separately when using this script. It internally triggers the complete CL pipeline.

**3. Combined Results**

Merges continual learning metrics with energy data:
```json
{
  "total_runtime_seconds": 1847.2,
  "total_energy_kwh": 0.342,
  "total_co2_kg": 0.128,
  "avg_power_watts": 247.5,
  "continual_learning_results": {
    "baseline": {...},
    "CL_1": {...},
    ...
  }
}
```

**4. Output Files**

Saves to `results/energy_tracking/`:
- `cl_energy_results.json` - Combined metrics
- `codecarbon_logs/emissions.csv` - Raw CodeCarbon output with timestamped measurements

#### How it helps

**Environmental Accountability:**
- Quantifies the carbon footprint of machine learning training
- Connects methodological choices to real-world environmental impact
- Supports the "energy-efficient" narrative of the project

**Cost Estimation:**
- Energy consumption can be converted to monetary cost using local electricity rates
- Informs decision-making about training frequency and model complexity

**Optimization Target:**
- Energy becomes an explicit optimization metric alongside prediction accuracy
- Enables multi-objective optimization: minimize both RMSE and kWh

#### Next-run improvements

**Comprehensive Energy Tracking:**
- Track base training energy (not just continual learning)
- Compare energy consumption across:
  - Different model architectures
  - Various continual learning strategies
  - Multiple hyperparameter configurations

**Normalized Metrics:**
- Report energy per performance unit:
  - **Energy efficiency:** kWh per 1% RMSE reduction
  - **Sample efficiency:** kWh per 1000 samples processed
- Enables fair comparison across experiments with different scales

**Power Profiling:**
- Break down power consumption by component:
  - GPU utilization
  - CPU utilization
  - Memory bandwidth
- Identify bottlenecks for optimization

**Carbon-Aware Training:**
- Integrate real-time grid carbon intensity data
- Implement training schedulers that:
  - Pause training during high-carbon periods
  - Resume during low-carbon periods
- Reduces carbon footprint without changing model quality

---

## 8. Energy-Aware Scheduling Simulation

### `energy_aware_scheduling.py`

#### What it does

Simulates intelligent scheduling of training jobs to minimize cost and carbon emissions without changing the computational workload.

**Note:** This script does **not** run actual model training. It is a **simulation** that estimates potential savings from strategic scheduling.

**Simulation Setup:**

**1. Synthetic Grid Profile Creation**

Generates 24-hour profiles (48 30-minute slots):

**Electricity Price Profile:**
- Models time-of-use pricing:
  - High during peak demand (morning/evening)
  - Low during off-peak (night/early morning)
- Example: `price_per_kwh[t] = 0.15 + 0.10 × sin(2π × t/24)`

**Carbon Intensity Profile:**
- Models renewable energy availability:
  - Low during solar peak (midday)
  - High when fossil fuels dominate (night)
- Example: `carbon_kg_per_kwh[t] = 0.4 - 0.2 × sin(2π × (t-6)/24)`

**2. Job Definition**

Defines training jobs with:
- **Duration:** How long the job runs (e.g., 3 hours)
- **Power draw:** Average power consumption (e.g., 0.30 kW)
- **Energy:** Total energy = duration × power

**Example Jobs:**
- **Base training:** 3 hours @ 0.30 kW = 0.90 kWh
- **CL update 1:** 1 hour @ 0.25 kW = 0.25 kWh
- **CL update 2:** 1 hour @ 0.25 kW = 0.25 kWh
- **CL update 3:** 1 hour @ 0.25 kW = 0.25 kWh
- **CL update 4:** 1 hour @ 0.25 kW = 0.25 kWh

**3. Scheduling Strategies**

**Immediate Strategy:**
- Schedule jobs as soon as possible (starting at hour 0)
- Simulates typical on-demand training
- No optimization

**Night-Only Strategy:**
- Restrict all jobs to 22:00–06:00 window
- Leverages lower off-peak electricity prices
- May reduce cost but not optimally

**Carbon-Aware Strategy:**
- For each job, find the time window that minimizes:
  ```
  Objective = α × cost + β × emissions
  ```
- Considers both economic and environmental factors
- Uses greedy or optimization algorithm to place jobs

**4. Metrics Computation**

For each strategy, calculates:

**Energy (kWh):**
```
Total energy = Σ (job_duration × job_power)
```
Note: Energy is the same across strategies (same workload)

**Cost:**
```
Cost = Σ (job_energy × price_at_scheduled_time)
```

**Emissions (kg CO₂):**
```
Emissions = Σ (job_energy × carbon_intensity_at_scheduled_time)
```

**Per-Job Schedule:**
- Start time, end time, cost, emissions for each job

**5. Results Output**

Generates comparison table:
```
Strategy        Energy(kWh)  Cost     Emissions(kg)
-------------------------------------------------
Immediate       2.00         0.432    0.876
Night-Only      2.00         0.364    0.823
Carbon-Aware    2.00         0.341    0.687
```

Savings vs. immediate:
- Cost reduction: 21.1%
- Emission reduction: 21.6%

Saves detailed results to:
```
results/energy_scheduling/energy_scheduling_results.json
```

#### How it helps

**Demonstrates Scheduling Impact:**
- Shows that **when** you train matters as much as **how** you train
- Same computational work can have vastly different environmental impact

**Policy Implications:**
- Informs decisions about:
  - When to schedule batch training jobs
  - Whether to implement job queuing systems
  - How to set organizational training policies

**Complements Energy Tracking:**
- Energy tracking measures **what is**
- Scheduling simulation proposes **what could be**
- Together they provide complete picture

#### Next-run improvements

**Real-World Grounding:**
- Use actual durations and power measurements from `energy_tracking.py`
- Replace synthetic profiles with:
  - Real electricity price data (e.g., from utility APIs)
  - Real carbon intensity data (e.g., from electricityMap API)

**Constraint Modeling:**

**Deadline Constraints:**
- Add maximum acceptable completion time
- Example: "CL updates must complete within 24 hours"

**Resource Constraints:**
- Model limited GPU availability
- Prevent overlapping jobs if only one GPU available

**Dependency Constraints:**
- Enforce ordering: "CL update 2 cannot start before CL update 1 completes"

**Advanced Scheduling:**

**Optimization Algorithms:**
- Replace greedy scheduling with:
  - Integer Linear Programming (ILP)
  - Dynamic Programming
  - Genetic Algorithms
- Find globally optimal schedules

**Predictive Scheduling:**
- Use historical grid data to forecast future prices/carbon
- Schedule based on predictions rather than current conditions

**Adaptive Scheduling:**
- Dynamically adjust schedule based on:
  - Job completion time variations
  - Unexpected high/low carbon periods
  - Grid emergencies or peak demand events

---

## 9. Global Next-Run Improvement Plan

Based on the baseline run, the following improvements should be prioritized for the next experimental iteration:

### Model Improvements

**Architectural Enhancements:**
- Implement deeper STGNN with multiple stacked spatio-temporal blocks
- Add residual connections to facilitate gradient flow through deep networks
- Integrate attention mechanisms:
  - Spatial attention for dynamic neighbor weighting
  - Temporal attention for adaptive time-step importance

**Multi-Feature Input:**
- Extend beyond single kWh feature to include:
  - Binary imputation flags
  - Temporal features (hour, day, month, holiday indicators)
  - External variables (temperature, weather conditions)
- Modify input layer to handle multi-channel data

**Uncertainty Quantification:**
- Implement probabilistic forecasting (e.g., quantile regression, Monte Carlo dropout)
- Provide prediction intervals alongside point estimates
- Enable risk-aware decision making

### Data & Preprocessing Improvements

**External Data Integration:**
- Obtain and incorporate:
  - Historical weather data aligned by province
  - Calendar information (public holidays, special events)
  - Economic indicators (electricity prices over time)
- Create feature engineering pipeline for exogenous variables

**Normalization Strategies:**
- Experiment with per-node normalization to handle heterogeneous households
- Test per-tariff group normalization to account for structural differences
- Compare global vs. local scaling strategies

**Node Selection Analysis:**
- Investigate whether using a subset of households improves:
  - Training stability
  - Computational efficiency
  - Prediction accuracy
- Identify optimal household sampling strategies

### Continual Learning Improvements

**Strategy Implementation:**
- Implement at least two continual learning approaches:
  - **Regularization-based:** EWC or LwF
  - **Replay-based:** Experience replay with buffer

**Comparative Evaluation:**
- Compare three scenarios:
  - No adaptation (frozen base model)
  - Naive fine-tuning (current baseline)
  - Advanced CL strategy A
  - Advanced CL strategy B
- Measure trade-offs across:
  - New data performance
  - Old data retention
  - Computational cost
  - Energy consumption

**Selective Adaptation:**
- Freeze spatial GCN layers (stable structure)
- Fine-tune only temporal components or output layers
- Hypothesis: Spatial relationships are time-invariant, temporal patterns drift

### Energy & Scheduling Enhancements

**Integrated Energy Tracking:**
- Track energy for all pipeline stages:
  - Data preprocessing
  - Base training
  - Evaluation
  - Continual learning updates
- Build comprehensive energy profile of entire workflow

**Real-World Scheduling:**
- Replace synthetic profiles with real data:
  - Use electricity pricing APIs (e.g., utility providers)
  - Use carbon intensity APIs (e.g., electricityMap, WattTime)
- Implement actual scheduling system that:
  - Monitors grid conditions
  - Queues jobs
  - Executes during optimal windows

**Energy-Performance Pareto Analysis:**
- Generate Pareto frontiers showing trade-off between:
  - Prediction accuracy (RMSE)
  - Energy consumption (kWh)
  - Training time (hours)
- Identify configurations that are Pareto-optimal
- Inform decision-making based on priorities (accuracy vs. efficiency)

### Experimental Design

**Ablation Studies:**
Systematically evaluate component contributions:

- **Graph ablation:** Train with vs. without spatial graph
- **Temporal ablation:** Train with vs. without temporal module
- **K-NN ablation:** Compare K ∈ {5, 10, 15, 20} in graph construction
- **Feature ablation:** Remove one feature type at a time

These studies strengthen evidence for architectural decisions.

**Statistical Significance:**
- Run multiple random seeds (e.g., 5 seeds)
- Report mean ± standard deviation for all metrics
- Perform statistical tests (t-test, Wilcoxon) to confirm improvements

**Documentation:**
- Maintain detailed logs of all experiments
- Version control configuration files
- Enable exact reproducibility

### Narrative Construction

This comprehensive improvement plan enables a compelling research narrative:

**Baseline (Current Run):**
- Establish STGNN + naive continual learning + energy tracking + scheduling simulation
- Document baseline performance, energy consumption, and scheduling potential

**Next Run:**
- Architectural improvements → Better accuracy
- Advanced CL strategies → Reduced forgetting
- Real-world energy modeling → Actionable sustainability insights

**Final Contribution:**
- Demonstrate that spatio-temporal graph neural networks can:
  - Achieve accurate multi-step forecasting
  - Adapt to temporal distribution shifts with minimal forgetting
  - Operate efficiently with reduced environmental impact through intelligent scheduling
- Provide practical framework for deploying energy-efficient ML in smart grid applications

---

