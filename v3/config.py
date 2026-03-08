"""Configuration constants for the sleep staging pipeline."""

import os

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ZIP = os.path.join(os.path.dirname(PROJECT_DIR), "data.zip")
EXTRACT_DIR = os.path.join(PROJECT_DIR, ".data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Reproducibility
RANDOM_STATE = 42
CV_SPLITS = 5

# Preprocessing
TOLERANCE = 30  # merge_asof tolerance in seconds
EPOCH_SECONDS = 30
MOVEMENT_BUFFER = 0.05

# Smoothing
SMOOTHING_WINDOW = 5  # epochs (2.5 min)

# Label maps
LABEL_MAP_3STAGE = {0: 'Wake', 1: 'NREM', 2: 'NREM', 3: 'NREM', 5: 'REM'}
LABEL_MAP_5STAGE = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'REM'}

# Stage encodings (for smoothing)
STAGE_TO_INT_3 = {'NREM': 0, 'REM': 1, 'Wake': 2}
INT_TO_STAGE_3 = {0: 'NREM', 1: 'REM', 2: 'Wake'}

STAGE_TO_INT_5 = {'N1': 0, 'N2': 1, 'N3': 2, 'REM': 3, 'Wake': 4}
INT_TO_STAGE_5 = {0: 'N1', 1: 'N2', 2: 'N3', 3: 'REM', 4: 'Wake'}

# Feature lists
BASELINE_FEATURES_3 = [
    'mean_vm', 'std_vm', 'time_since_last_movement',
    'roll_mean_vm_2m', 'roll_std_vm_2m',
    'roll_mean_vm_5m', 'roll_std_vm_5m',
]

BASELINE_FEATURES_5 = BASELINE_FEATURES_3 + [
    'roll_std_vm_10m', 'roll_mean_vm_20m',
]

# Default model parameters (matching v2)
RF_PARAMS_3 = dict(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE)
RF_PARAMS_5 = dict(n_estimators=150, max_depth=12, min_samples_leaf=4, random_state=RANDOM_STATE)

XGB_PARAMS_3 = dict(
    n_estimators=150, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=3,
    random_state=RANDOM_STATE, tree_method='hist',
)

XGB_PARAMS_5 = dict(
    n_estimators=150, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    random_state=RANDOM_STATE, tree_method='hist',
)
