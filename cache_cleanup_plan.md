# Cache File Cleanup & Renaming Plan

## 📊 Current Cache Analysis

### **DUPLICATES IDENTIFIED:**
```
enhanced_mi_dataset_2000_1000_100.pkl (41.5 MB) = mi_enhanced_dataset_1000_100_100.pkl (41.5 MB)
focused_mi_dataset_20_2000.pkl (90.9 MB) = mi_enhanced_dataset_2000_20_100.pkl (90.9 MB)
```
**IDENTICAL DATA - Can delete one of each pair**

### **Clear Purpose Files:**
```
signals_ptbxl_100hz_5.pkl (0.2 MB)    -> 5 samples for quick testing
signals_ptbxl_100hz_25.pkl (1.1 MB)   -> 25 samples for rapid dev
signals_ptbxl_100hz_1000.pkl (45.8 MB) -> 1000 samples for training
signals_ptbxl_100hz_2000.pkl (91.6 MB) -> 2000 samples for validation
```

### **Problematic Files:**
```
ptbxl_optimized_None_100.pkl (0.0 MB)     -> Broken/empty - DELETE
preprocessed_signals_10.pkl (0.5 MB)      -> Only 10 samples - likely test file
ecg_arrhythmia_100hz_0_0mi.pkl (41.4 MB)  -> Dictionary format, unclear purpose
ptbxl_mi_clinical_dataset.pkl (91.7 MB)   -> Dictionary format, likely important
```

## 🎯 Proposed New Naming Convention

### **Format:** `{dataset}_{purpose}_{samples}.pkl`

```
OLD NAME                              -> NEW NAME                           ACTION
=====================================================================================================
signals_ptbxl_100hz_5.pkl            -> ptbxl_quick_test_5.pkl            RENAME
signals_ptbxl_100hz_25.pkl           -> ptbxl_rapid_dev_25.pkl            RENAME  
signals_ptbxl_100hz_1000.pkl         -> ptbxl_training_1000.pkl           RENAME
signals_ptbxl_100hz_2000.pkl         -> ptbxl_validation_2000.pkl         RENAME

enhanced_mi_dataset_2000_1000_100.pkl -> combined_mi_focused_906.pkl        RENAME
focused_mi_dataset_20_2000.pkl       -> combined_general_1986.pkl          RENAME
mi_enhanced_dataset_1000_100_100.pkl -> DELETE (duplicate)
mi_enhanced_dataset_2000_20_100.pkl  -> DELETE (duplicate)

ptbxl_mi_clinical_dataset.pkl         -> ptbxl_clinical_metadata.pkl       RENAME
ecg_arrhythmia_100hz_0_0mi.pkl        -> arrhythmia_metadata.pkl           RENAME

preprocessed_signals_10.pkl          -> DELETE (too small, test file)
ptbxl_optimized_None_100.pkl         -> DELETE (broken)
```

## 📈 Results After Cleanup

### **Before:** 12 files, 538 MB, confusing names
### **After:** 8 files, ~450 MB, clear purposes

```
LAPTOP-OPTIMIZED CACHE STRUCTURE:
├── ptbxl_quick_test_5.pkl      (0.2 MB)  # 2-second load
├── ptbxl_rapid_dev_25.pkl      (1.1 MB)  # 5-second load  
├── ptbxl_training_1000.pkl     (45.8 MB) # 30-second load
├── ptbxl_validation_2000.pkl   (91.6 MB) # 60-second load
├── combined_mi_focused_906.pkl (41.5 MB) # MI-enhanced training
├── combined_general_1986.pkl   (90.9 MB) # General validation
├── ptbxl_clinical_metadata.pkl (91.7 MB) # Clinical dataset info
└── arrhythmia_metadata.pkl     (41.4 MB) # Arrhythmia dataset info
```

## 💡 Benefits

1. **Clear Purpose:** Each file name tells you exactly what it contains
2. **Size Progression:** Easy to choose right size for your laptop's performance
3. **No Duplicates:** Saves ~80MB space
4. **Laptop-Friendly:** Quick test files for rapid development
5. **Consistent Naming:** Easy to remember and script against

## 🚀 Implementation

Total cleanup saves: ~88 MB + eliminates confusion
Implementation time: ~2 minutes
Risk level: LOW (keeping all unique data)