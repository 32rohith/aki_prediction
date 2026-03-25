"""
Configuration file for AKI Prediction Pipeline
Contains all key parameters and paths for reproducibility
"""

import os

# Random seed for reproducibility
RANDOM_SEED = 42

# Directory paths
RAW_DATA_DIR = "raw_data"
PROCESSED_DATA_DIR = "processed_data"
FIGURES_DIR = "figures"
MODELS_DIR = "models"
RESULTS_DIR = "results"
LOGS_DIR = "logs"
SCRIPTS_DIR = "scripts"

# Data file paths (relative to raw_data directory)
ICUSTAYS_FILE = os.path.join(RAW_DATA_DIR, "icustays.csv")
LABEVENTS_FILE = os.path.join(RAW_DATA_DIR, "labevents.csv")
PATIENTS_FILE = os.path.join(RAW_DATA_DIR, "patients.csv")
D_LABITEMS_FILE = os.path.join(RAW_DATA_DIR, "d_labitems.csv")
DISCHARGE_FILE = os.path.join(RAW_DATA_DIR, "discharge.csv")

# KDIGO criteria parameters
# All blood/serum creatinine itemids verified from d_labitems.csv
CREATININE_ITEMIDS = [50912, 52546, 52024, 51081]
MIN_ICU_DURATION_HOURS = 24
TEMPORAL_CUTOFF_HOURS = 24
AKI_48H_INCREASE_THRESHOLD = 0.3  # mg/dL
AKI_7D_RATIO_THRESHOLD = 1.5

# Feature engineering parameters
LAB_COVERAGE_THRESHOLD = 0.30  # 30% minimum coverage
AKI_RELEVANT_LABS = [
    "BUN", "Sodium", "Potassium", "Chloride", "Bicarbonate",
    "Lactate", "WBC", "Hemoglobin", "Platelets", "Glucose",
    "Calcium", "Magnesium", "Phosphate"
]

# Data splitting parameters
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
AKI_PREVALENCE_WARNING_THRESHOLD = 0.10  # 10 percentage points

# Model training parameters
LOGISTIC_REGRESSION_PARAMS = {
    "random_state": RANDOM_SEED,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced"
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_SEED,
    "max_depth": 10,
    "class_weight": "balanced"
}

MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "solver": "adam",
    "random_state": RANDOM_SEED,
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.1
}

# Text processing parameters
BIOCLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
TEXT_MAX_LENGTH = 512
TEXT_BATCH_SIZE = 32
EMBEDDING_DIM = 768

# Visualization parameters
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Memory management
LARGE_FILE_THRESHOLD_GB = 1.0
CHUNK_SIZE = 10000
CHECKPOINT_INTERVAL = 1000

# Calibration parameters
CALIBRATION_BINS = 10

# Logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
