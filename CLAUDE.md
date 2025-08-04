# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running the Application
```bash
# Local development
streamlit run app/main.py

# With ngrok for sharing
python app/main.py --use-ngrok

# Windows batch launchers (various configurations)
QUICK_LAUNCH.bat                 # Simple launch
LAUNCH_ECG_COMPREHENSIVE.bat     # Full featured launch
PROFESSIONAL_LAUNCHER.bat        # Professional configuration
STREAMLINED_LAUNCHER.bat         # Optimized launch
```

### Testing Phase Components
```bash
# Test individual phases
python test_phase1.py    # Data loading foundation
python test_phase2.py    # Preprocessing and signal filtering
python test_phase3.py    # Feature extraction
python test_phase4.py    # Model training and evaluation

# Test combined dataset integration (for improved MI detection)
python test_combined_dataset.py

# Test imports
python test_imports.py
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Windows installation helpers
ECG_PROFESSIONAL_INSTALLER.bat   # Professional setup
SIMPLE_INSTALLER.bat             # Basic installation
INSTALL_DESKTOP_SHORTCUT.bat     # Create desktop shortcuts
```

## Architecture Overview

This is an ECG classification system built for clinical decision support that follows a 7-phase development approach:

### Phase Structure
1. **Phase 1**: Foundation & Data Loading (`app/utils/dataset_manager.py`, `app/utils/data_loader.py`)
2. **Phase 2**: Preprocessing & Signal Filtering (`models/preprocessing/`)
3. **Phase 3**: Feature Engineering (`models/feature_extraction/`)
4. **Phase 4**: Model Training & Evaluation (`models/training/`)
5. **Phase 5**: Deployment Interface (`app/`)
6. **Phase 6**: Production Deployment (`deployment/`)
7. **Phase 7**: Clinical Validation

### Core Components

**Data Pipeline**:
- `app/utils/dataset_manager.py`: High-level dataset management and unified interface
- `app/utils/data_loader.py`: PTB-XL, MIT-BIH, and ECG Arrhythmia dataset loaders
- **ECG Arrhythmia Integration**: Enhanced MI detection with 45,152 physician-labeled records
- Uses caching system in `data/cache/` for preprocessed signals and features

**Processing Pipeline**:
- `models/preprocessing/preprocessing_pipeline.py`: Complete preprocessing pipeline
- Signal quality assessment, filtering, normalization, and label processing
- Configurable through `config/preprocessing_config.py`

**Feature Extraction**:
- `models/feature_extraction/feature_extraction_pipeline.py`: Main feature pipeline
- Clinical features: temporal, frequency, wavelet, ST-segment analysis
- Feature selection and PCA transformation
- Configurable through `config/feature_config.py`

**Model Training**:
- `models/training/training_pipeline.py`: Complete training pipeline
- Supports multiple algorithms (Random Forest, SVM, etc.)
- Comprehensive evaluation and visualization
- Configurable through `config/model_config.py`

### Target Classifications (Comprehensive - 30 Conditions)
The system has been expanded from 5 basic to 30 comprehensive cardiac conditions:

**Core Original Classifications:**
- **NORM**: Normal sinus rhythm
- **MI**: Myocardial Infarction (multiple types: AMI, IMI, LMI, PMI)
- **STTC**: ST/T Changes  
- **CD**: Conduction Disorders
- **HYP**: Hypertrophy

**Enhanced Detection Includes:**
- Multiple MI types (Anterior, Inferior, Lateral, Posterior)
- Comprehensive arrhythmias and rhythm disorders
- Conduction abnormalities
- Axis deviations and chamber enlargements

### Key Architecture Principles
- **Configuration-driven**: All components use config classes for easy parameter tuning
- **Caching system**: Extensive use of pickle caching for preprocessed data and features
- **Modular design**: Each phase can be run independently for testing and development
- **Clinical focus**: Features and outputs designed for healthcare professional use

### Data Structure
- **Raw data**: 
  - `data/raw/ptbxl/` (PTB-XL dataset with 21,388 samples)
  - `data/raw/ecg-arrhythmia-dataset/` (ECG Arrhythmia dataset with 45,152 samples)
- **Processed data**: `data/processed/` (preprocessed signals, features, labels)
- **Cache**: `data/cache/` (intermediate processing results)
- **Results**: `data/results/` and `data/visualizations/` (analysis outputs)

### Enhanced MI Detection
The system now supports **combined dataset loading** for dramatically improved MI detection:
```python
from app.utils.dataset_manager import run_combined_dataset_loading

# Load combined PTB-XL + ECG Arrhythmia dataset
X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
    ptbxl_max_records=5000,
    arrhythmia_max_records=2000,
    target_mi_records=1000,  # Ensures adequate MI representation
    sampling_rate=100
)
```

**Key MI Improvements**:
- **Label Mapping**: Comprehensive mapping from arrhythmia conditions to target classes
- **MI Focus**: Prioritizes STEMI, NSTEMI, AMI, IMI, LMI records
- **Clinical Validation**: All ECG Arrhythmia labels are physician-validated
- **WFDB Support**: Handles `.mat` + `.hea` file formats automatically

### Configuration Management
All major components are configurable through files in `config/`:
- `settings.py`: Global paths and target conditions
- `preprocessing_config.py`: Signal processing parameters
- `feature_config.py`: Feature extraction settings
- `model_config.py`: Training algorithm configurations

The system uses preset configurations (e.g., 'standard', 'clinical_optimized') that can be easily switched for different use cases.