# Dataset Setup Guide

This guide explains how to set up the datasets for the ECG Classification System, including the new ECG Arrhythmia dataset for improved MI detection.

## Required Datasets

### 1. PTB-XL Dataset (Primary)
- **Source**: PhysioNet
- **Records**: 21,388 ECG recordings
- **Format**: WFDB (.dat + .hea files)
- **Download**: Automatic via the system

### 2. ECG Arrhythmia Dataset (Enhanced MI Detection)
- **Source**: PhysioNet - "A large scale 12-lead electrocardiogram database for arrhythmia study"
- **URL**: https://physionet.org/content/ecg-arrhythmia/1.0.0/
- **Records**: 45,152 ECG recordings
- **Format**: WFDB (.mat + .hea files)
- **Size**: ~15 GB compressed
- **Download**: Manual (see instructions below)

## Setup Instructions

### Step 1: Create Data Directories
The system will auto-create these, but you can create them manually:
```bash
mkdir -p data/raw/ptbxl
mkdir -p data/raw/ecg-arrhythmia-dataset
mkdir -p data/processed
mkdir -p data/cache
```

### Step 2: PTB-XL Dataset (Automatic)
The PTB-XL dataset downloads automatically when you run:
```bash
python test_phase1.py
```

### Step 3: ECG Arrhythmia Dataset (Manual)

#### Download
1. Go to: https://physionet.org/content/ecg-arrhythmia/1.0.0/
2. Click "Download the ZIP file" (requires PhysioNet account)
3. Download `ecg-arrhythmia-1.0.0.zip` (~15 GB)

#### Extract
1. Save the ZIP file to: `data/raw/ecg-arrhythmia-dataset/`
2. Extract the contents:
   ```bash
   cd data/raw/ecg-arrhythmia-dataset/
   unzip ecg-arrhythmia-1.0.0.zip
   ```

#### Verify Structure
After extraction, you should have:
```
data/raw/ecg-arrhythmia-dataset/
├── WFDBRecords/
│   ├── 01/
│   │   ├── 010/
│   │   │   ├── JS00001.mat
│   │   │   ├── JS00001.hea
│   │   │   ├── JS00002.mat
│   │   │   ├── JS00002.hea
│   │   │   └── ... (100 records per subfolder)
│   │   ├── 011/
│   │   └── ... (10 subfolders per main folder)
│   ├── 02/
│   └── ... (46 main folders total)
└── README (if included)
```

### Step 4: Test the Setup
Run the combined dataset test to verify everything works:
```bash
python test_combined_dataset.py
```

## Expected Improvements

### Before ECG Arrhythmia Integration
- MI Sensitivity: 0.000 (unable to detect heart attacks)
- Limited MI training data from PTB-XL

### After ECG Arrhythmia Integration
- Dramatic MI detection improvement
- 1000+ physician-labeled MI records available
- Clinical-grade validation for all labels

## Troubleshooting

### Issue: "ECG Arrhythmia dataset not found"
**Solution**: Ensure the WFDBRecords folder is directly under `data/raw/ecg-arrhythmia-dataset/`

### Issue: "Permission denied" during extraction
**Solution**: 
```bash
chmod +x data/raw/ecg-arrhythmia-dataset/
unzip ecg-arrhythmia-1.0.0.zip
```

### Issue: Dataset loading is slow
**Solution**: The first load will be slow as it scans 45,152 records. Subsequent loads use caching.

### Issue: Memory errors
**Solution**: Reduce the number of records loaded:
```python
# In your training script
X, labels, ids, _, _, _ = run_combined_dataset_loading(
    ptbxl_max_records=1000,      # Reduce from 5000
    arrhythmia_max_records=500,  # Reduce from 2000
    target_mi_records=200        # Reduce from 1000
)
```

## Using Combined Datasets

### Quick Test (Small Dataset)
```python
from app.utils.dataset_manager import run_combined_dataset_loading

X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
    ptbxl_max_records=100,
    arrhythmia_max_records=50,
    target_mi_records=20,
    sampling_rate=100
)
```

### Production Training (Large Dataset)
```python
X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
    ptbxl_max_records=5000,
    arrhythmia_max_records=2000,
    target_mi_records=1000,
    sampling_rate=100
)
```

### PTB-XL Only (Fallback)
```python
from app.utils.dataset_manager import run_phase1_foundation

X, labels, ids, metadata, target_conditions, _ = run_phase1_foundation(
    max_records=5000,
    sampling_rate=100
)
```

## Dataset Statistics

| Dataset | Records | MI Records* | Size | Format |
|---------|---------|-------------|------|--------|
| PTB-XL | 21,388 | Low | ~2 GB | .dat/.hea |
| ECG Arrhythmia | 45,152 | High | ~15 GB | .mat/.hea |
| **Combined** | **66,540** | **Excellent** | **~17 GB** | **Mixed** |

*Estimated based on label mapping and clinical validation

## Technical Notes

- **Sampling Rates**: PTB-XL (100Hz), ECG Arrhythmia (500Hz native, resampled as needed)
- **Lead Count**: Both datasets provide 12-lead ECG data
- **Label Mapping**: Comprehensive mapping from arrhythmia conditions to 5 target classes
- **Caching**: First load is slow, subsequent loads are fast due to intelligent caching
- **Memory Management**: Batched processing prevents memory overflow

## Support

If you encounter issues with dataset setup:
1. Check the troubleshooting section above
2. Verify your PhysioNet account can access the datasets
3. Ensure sufficient disk space (~20 GB free recommended)
4. Run `python test_combined_dataset.py` for diagnostic information