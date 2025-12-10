#!/usr/bin/env python3
"""
Centralized Configuration for Energy-Efficient STGNN
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
# TRAINING CONFIGURATION (STABILITY + MEMORY FRIENDLY)
# ============================================================================

EPOCHS = 50                         # Early stopping will cut this
LEARNING_RATE = 1e-4                # Gentle LR
WEIGHT_DECAY = 1e-3                 # Stronger regularization
EARLY_STOPPING_PATIENCE = 5         # Stop if no improvement

# Loss function: "mse", "mae", "huber"
LOSS_FN = "huber"
HUBER_DELTA = 1.0

# Gradient clipping
GRADIENT_CLIP_NORM = 0.5

# Scheduler
SCHEDULER_MODE = 'min'
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR = 1e-6
SCHEDULER_VERBOSE = True

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Smaller dims to help memory
GCN_HIDDEN = 16                     # Spatial hidden dim
GCN_LAYERS = 1

GRU_HIDDEN = 16                     # Temporal hidden dim
GRU_LAYERS = 1

SPATIAL_DROPOUT = 0.3
TEMPORAL_DROPOUT = 0.3
FINAL_DROPOUT   = 0.3

HORIZON = 6                         # forecast horizon

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

WINDOW_SIZE = 24                    # temporal window size

# Memory-sensitive: batch_size=1 to start; you can try 2 later
BATCH_SIZE = 1
NUM_WORKERS = 0
PIN_MEMORY  = True

# ============================================================================
# CONTINUAL LEARNING (unused here but kept)
# ============================================================================

CL_EPOCHS = 3
CL_LEARNING_RATE = 5e-4
CL_WEIGHT_DECAY = 5e-4

# EMA (disabled)
USE_EMA = False
EMA_DECAY = 0.995
EMA_UPDATE_AFTER_STEP = 100

# ============================================================================
# GRAPH SETTINGS
# ============================================================================

K_NEIGHBORS = 10
SIMILARITY_THRESHOLD = 0.0
NORMALIZE_ADJACENCY = True

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

METRICS = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
EVALUATE_PER_HORIZON = True

# ============================================================================
# ENERGY TRACKING SETTINGS
# ============================================================================

TRACK_EMISSIONS = True
EMISSIONS_LOG_DIR = RESULTS_ROOT / "energy_tracking" / "codecarbon_logs"
COUNTRY_ISO_CODE = "USA"

# ============================================================================
# LOGGING & VISUALIZATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE = LOGS_DIR / "training.log"

USE_TQDM = True
TQDM_LEAVE = False

SAVE_PLOTS = True
PLOT_FORMAT = 'png'
PLOT_DPI = 300

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42
DETERMINISTIC = True

# ============================================================================
# HELPERS
# ============================================================================

def get_model_config():
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
    return {
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'gradient_clip_norm': GRADIENT_CLIP_NORM,
        'loss_fn': LOSS_FN,
        'huber_delta': HUBER_DELTA,
        'batch_size': BATCH_SIZE,
        'scheduler': {
            'mode': SCHEDULER_MODE,
            'factor': SCHEDULER_FACTOR,
            'patience': SCHEDULER_PATIENCE,
            'min_lr': SCHEDULER_MIN_LR,
        }
    }

def get_data_config():
    return {
        'window_size': WINDOW_SIZE,
        'horizon': HORIZON,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': PIN_MEMORY,
    }

def print_config():
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

def validate_config():
    assert EPOCHS > 0
    assert LEARNING_RATE > 0
    assert WEIGHT_DECAY >= 0
    assert EARLY_STOPPING_PATIENCE > 0
    assert BATCH_SIZE > 0
    assert WINDOW_SIZE > 0
    assert HORIZON > 0
    assert 0 <= SPATIAL_DROPOUT < 1
    assert 0 <= TEMPORAL_DROPOUT < 1
    assert GRADIENT_CLIP_NORM > 0
    print("[config] ✅ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print_config()
