"""
Configuration settings for ECG Classification System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "features", 
                  CACHE_DIR, MODELS_DIR / "trained_models"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "ptbxl": {
        "path": DATA_DIR / "raw" / "ptbxl",
        "metadata_files": {
            "database": "ptbxl_database.csv",
            "scp_statements": "scp_statements.csv"
        },
        "urls": {
            "database": "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv",
            "scp_statements": "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv",
            "records_base": "https://physionet.org/files/ptb-xl/1.0.3/"
        }
    },
    "ecg_arrhythmia": {
        "path": DATA_DIR / "raw" / "ecg-arrhythmia-dataset",
        "url": "https://physionet.org/files/ecg-arrhythmia/1.0.0/"
    }
}

# Model configuration
TARGET_CONDITIONS = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SAMPLING_RATES = {
    "low": 100,   # 100 Hz
    "high": 500   # 500 Hz
}

# Processing configuration
PROCESSING_CONFIG = {
    "batch_size": 1000,
    "n_jobs": 4,
    "use_cache": True,
    "cache_format": "pickle"  # or "parquet" for larger datasets
}

# App configuration
APP_CONFIG = {
    "title": "ECG Classification System",
    "page_icon": "🫀",
    "layout": "wide"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}