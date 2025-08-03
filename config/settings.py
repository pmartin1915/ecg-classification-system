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

# Model configuration - COMPREHENSIVE CARDIAC DETECTION
# Expanded from 5 basic to 30 comprehensive cardiac conditions
TARGET_CONDITIONS = [
    # === NORMAL ===
    'NORM',           # Normal ECG
    
    # === MYOCARDIAL INFARCTION (Multiple Types) ===
    'AMI',            # Anterior MI (includes ASMI, ALMI)
    'IMI',            # Inferior MI (includes ILMI, IPLMI, IPMI)
    'LMI',            # Lateral MI
    'PMI',            # Posterior MI
    
    # === ARRHYTHMIAS & RHYTHM DISORDERS ===
    'PVC',            # Premature Ventricular Contractions
    'PAC',            # Premature Atrial Contractions
    'AFIB',           # Atrial Fibrillation
    'AFLT',           # Atrial Flutter
    'SVTAC',          # Supraventricular Tachycardia
    'VTAC',           # Ventricular Tachycardia
    
    # === CONDUCTION DISORDERS ===
    'AVB1',           # 1st Degree AV Block
    'AVB2',           # 2nd Degree AV Block
    'AVB3',           # 3rd Degree AV Block (Complete Heart Block)
    'RBBB',           # Right Bundle Branch Block (complete/incomplete)
    'LBBB',           # Left Bundle Branch Block (complete/incomplete)
    'LAFB',           # Left Anterior Fascicular Block
    'LPFB',           # Left Posterior Fascicular Block
    'IVCD',           # Intraventricular Conduction Delay
    'WPW',            # Wolff-Parkinson-White Syndrome
    
    # === HYPERTROPHY & CHAMBER ABNORMALITIES ===
    'LVH',            # Left Ventricular Hypertrophy
    'RVH',            # Right Ventricular Hypertrophy
    'LAE',            # Left Atrial Enlargement
    'RAE',            # Right Atrial Enlargement
    
    # === ST-T CHANGES & ISCHEMIA ===
    'ISCH',           # Ischemic Changes (various locations)
    'STTC',           # Non-specific ST-T Changes
    'LNGQT',          # Long QT Interval
    
    # === OTHER CLINICALLY SIGNIFICANT ===
    'PACE',           # Paced Rhythm
    'DIG',            # Digitalis Effect
    'LOWT'            # Low T-wave Voltage
]

# Clinical Priority Classification
CLINICAL_PRIORITY = {
    'CRITICAL': ['AMI', 'IMI', 'VTAC', 'AVB3', 'LMI', 'PMI'],  # Immediate medical attention
    'HIGH': ['PVC', 'AFIB', 'AVB2', 'LBBB', 'WPW', 'AFLT'],   # Close monitoring required
    'MEDIUM': ['AVB1', 'RBBB', 'LVH', 'ISCH', 'SVTAC'],       # Regular follow-up
    'LOW': ['NORM', 'STTC', 'DIG', 'LOWT', 'LAE', 'RAE']      # Routine care
}
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