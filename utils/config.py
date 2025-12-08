#!/usr/bin/env python3
"""
Centralized Configuration for Energy-Efficient STGNN

This module contains all hyperparameters and settings for reproducible experiments.
Modify values here instead of hardcoding in individual scripts.

Author: Energy-Efficient STGNN Project
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

# Data directories
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
SPLITS_DIR = DATA_ROOT / "splits"

# Results directories
MODELS_DIR = PROJECT_ROOT / "src" / "models"
LOGS_DIR = RESULTS_ROOT / "logs"
PLOTS_DIR = RESULTS_ROOT / "plots"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Basic training settings
EPOCHS = 20                          # Maximum epochs (reduced from 30)
LEARNING_RATE = 1e-3                 # Initial learning rate
WEIGHT_DECAY = 5e-4                  # L2 regularization (increased from 1e-4)
EARLY_STOPPING_PATIENCE = 3          # Stop after N bad epochs (reduced from 6)

# Gradient clipping
GRADIENT_CLIP_NORM = 1.0             # Max gradient norm for clipping

# Learning rate scheduler (ReduceLROnPlateau)
SCHEDULER_MODE = 'min'               # Minimize validation metric
SCHEDULER_FACTOR = 0.5               # Reduce LR by this factor
SCHEDULER_PATIENCE = 2               # Epochs to wait before reducing LR
SCHEDULER_MIN_LR = 1e-6              # Minimum learning rate
SCHEDULER_VERBOSE = True             # Print LR changes

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Spatial GCN settings
GCN_HIDDEN = 16                      # Hidden features in GCN layers
GCN_LAYERS = 1                       # Number of GCN layers

# Temporal GRU settings
GRU_HIDDEN = 32                      # Hidden state size in GRU
GRU_LAYERS = 1                       # Number of GRU layers

# Regularization
SPATIAL_DROPOUT = 0.3                # Dropout in GCN layers (increased from 0.1)
TEMPORAL_DROPOUT = 0.3               # Dropout in GRU (enabled, was 0.0)
FINAL_DROPOUT = 0.3                  # Dropout before output layer

# Forecasting
HORIZON = 6                          # Hours to forecast ahead

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Temporal window
WINDOW_SIZE = 24                     # Input window (hours)

# DataLoader settings
BATCH_SIZE = 4                       # Batch size (RTX 4050 safe)
NUM_WORKERS = 0                      # DataLoader workers (0 for Windows safety)
PIN_MEMORY = True                    # Pin memory for GPU transfer

# ============================================================================
# CONTINUAL LEARNING CONFIGURATION
# ============================================================================

CL_EPOCHS = 3                        # Epochs per CL update
CL_LEARNING_RATE = 5e-4              # Lower LR for fine-tuning
CL_WEIGHT_DECAY = 5e-4               # Same as base training

# ============================================================================
# GRAPH CONSTRUCTION SETTINGS
# ============================================================================

# K-NN graph
K_NEIGHBORS = 10                     # Number of nearest neighbors
SIMILARITY_THRESHOLD = 0.0           # Minimum similarity to create edge

# Normalization
NORMALIZE_ADJACENCY = True           # Apply symmetric normalization

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Metrics to track
METRICS = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']

# Per-horizon evaluation
EVALUATE_PER_HORIZON = True          # Compute metrics for each forecast step

# ============================================================================
# ENERGY TRACKING SETTINGS
# ============================================================================

# CodeCarbon configuration
TRACK_EMISSIONS = True               # Enable carbon tracking
EMISSIONS_LOG_DIR = RESULTS_ROOT / "energy_tracking" / "codecarbon_logs"
COUNTRY_ISO_CODE = "USA"             # Update based on your location

# ============================================================================
# LOGGING & VISUALIZATION
# ============================================================================

# Logging
LOG_LEVEL = "INFO"                   # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True                   # Save logs to file
LOG_FILE = LOGS_DIR / "training.log"

# Progress bars
USE_TQDM = True                      # Show progress bars
TQDM_LEAVE = False                   # Remove progress bars after completion

# Plotting
SAVE_PLOTS = True                    # Save plots to disk
PLOT_FORMAT = 'png'                  # png, pdf, svg
PLOT_DPI = 300                       # Resolution for saved plots

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42                     # Random seed for reproducibility
DETERMINISTIC = True                 # Use deterministic algorithms (slower but reproducible)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config():
    """Get model architecture configuration as a dictionary."""
    return {
        'gcn_hidden': GCN_HIDDEN,
        'gcn_layers': GCN_LAYERS,
        'gru_hidden': GRU_HIDDEN,
        'horizon': HORIZON,
        'spatial_dropout': SPATIAL_DROPOUT,
        'temporal_dropout': TEMPORAL_DROPOUT,
        'final_dropout': FINAL_DROPOUT,
    }


def get_training_config():
    """Get training configuration as a dictionary."""
    return {
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'gradient_clip_norm': GRADIENT_CLIP_NORM,
        'batch_size': BATCH_SIZE,
        'scheduler': {
            'mode': SCHEDULER_MODE,
            'factor': SCHEDULER_FACTOR,
            'patience': SCHEDULER_PATIENCE,
            'min_lr': SCHEDULER_MIN_LR,
        }
    }


def get_data_config():
    """Get data configuration as a dictionary."""
    return {
        'window_size': WINDOW_SIZE,
        'horizon': HORIZON,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': PIN_MEMORY,
    }


def print_config():
    """Print all configuration settings."""
    print("\n" + "="*80)
    print("ENERGY-EFFICIENT STGNN CONFIGURATION")
    print("="*80)
    
    print("\n[MODEL ARCHITECTURE]")
    for k, v in get_model_config().items():
        print(f"  {k:20s}: {v}")
    
    print("\n[TRAINING]")
    cfg = get_training_config()
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2:18s}: {v2}")
        else:
            print(f"  {k:20s}: {v}")
    
    print("\n[DATA]")
    for k, v in get_data_config().items():
        print(f"  {k:20s}: {v}")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration values."""
    assert EPOCHS > 0, "EPOCHS must be positive"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
    assert WEIGHT_DECAY >= 0, "WEIGHT_DECAY must be non-negative"
    assert EARLY_STOPPING_PATIENCE > 0, "EARLY_STOPPING_PATIENCE must be positive"
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert WINDOW_SIZE > 0, "WINDOW_SIZE must be positive"
    assert HORIZON > 0, "HORIZON must be positive"
    assert 0 <= SPATIAL_DROPOUT < 1, "SPATIAL_DROPOUT must be in [0, 1)"
    assert 0 <= TEMPORAL_DROPOUT < 1, "TEMPORAL_DROPOUT must be in [0, 1)"
    assert GRADIENT_CLIP_NORM > 0, "GRADIENT_CLIP_NORM must be positive"
    print("[config] ✅ Configuration validated successfully")


if __name__ == "__main__":
    # Test configuration
    validate_config()
    print_config()
