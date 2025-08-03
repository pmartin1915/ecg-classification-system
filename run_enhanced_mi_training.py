"""
Enhanced MI Training with Smart Dataset Combination
Uses PTB-XL (proven working) + ECG Arrhythmia (additional data)
"""
import sys
from pathlib import Path
import warnings
import numpy as np
from collections import Counter
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.utils.data_loader import ECGDataLoader
from fix_arrhythmia_loader import FixedECGArrhythmiaLoader


def load_enhanced_mi_dataset():
    """Load enhanced dataset combining PTB-XL + ECG Arrhythmia"""
    print("ENHANCED MI DATASET LOADING")
    print("=" * 60)
    
    # Step 1: Load PTB-XL (we know this works perfectly)
    print("1. Loading PTB-XL dataset (proven working)...")
    ptbxl_loader = ECGDataLoader("ptbxl")
    
    try:
        # Load larger sample from PTB-XL since it works well
        X_ptbxl, labels_ptbxl, ids_ptbxl = ptbxl_loader.load_data(
            max_records=800,  # Increase PTB-XL contribution
            sampling_rate=100
        )
        print(f"   OK: Loaded {len(X_ptbxl)} PTB-XL records")
        print(f"   Shape: {X_ptbxl.shape}")
        
        # Check PTB-XL MI count
        ptbxl_mi_count = np.sum(labels_ptbxl == 1)  # MI is label 1 in PTB-XL
        print(f"   MI records in PTB-XL: {ptbxl_mi_count}")
        
    except Exception as e:
        print(f"   ERROR: PTB-XL loading failed: {e}")
        return None
    
    # Step 2: Add ECG Arrhythmia data (supplementary)
    print("\n2. Loading ECG Arrhythmia dataset (supplementary)...")
    arrhythmia_loader = FixedECGArrhythmiaLoader()
    
    try:
        # Find additional records from ECG Arrhythmia
        records = arrhythmia_loader.scan_for_records(max_records=100, target_mi_records=50)
        
        if records:
            print(f"   OK: Found {len(records)} valid ECG Arrhythmia records")
            
            # Load signals
            X_arr, labels_arr, ids_arr = arrhythmia_loader.load_records(records[:50])  # Limit to 50
            
            if len(X_arr) > 0:
                print(f"   OK: Loaded {len(X_arr)} ECG Arrhythmia signals")
                print(f"   Shape: {X_arr.shape}")
                
                # Map condition names to PTB-XL label format
                condition_to_label = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
                labels_arr_numeric = [condition_to_label.get(cond, 0) for cond in labels_arr]
                
                arrhythmia_mi_count = sum(1 for label in labels_arr_numeric if label == 1)
                print(f"   MI records in ECG Arrhythmia: {arrhythmia_mi_count}")
                
            else:
                print("   WARNING: No ECG Arrhythmia signals loaded, using PTB-XL only")
                X_arr, labels_arr_numeric, ids_arr = np.array([]), [], []
        else:
            print("   WARNING: No valid ECG Arrhythmia records found, using PTB-XL only")
            X_arr, labels_arr_numeric, ids_arr = np.array([]), [], []
            
    except Exception as e:
        print(f"   WARNING: ECG Arrhythmia loading failed ({e}), using PTB-XL only")
        X_arr, labels_arr_numeric, ids_arr = np.array([]), [], []
    
    # Step 3: Combine datasets
    print("\n3. Combining datasets...")
    
    if len(X_arr) > 0:
        # Ensure compatible shapes by resampling ECG Arrhythmia to match PTB-XL
        if X_arr.shape[1] != X_ptbxl.shape[1]:
            print(f"   Resampling ECG Arrhythmia from {X_arr.shape[1]} to {X_ptbxl.shape[1]} time steps")
            # Simple resampling
            target_length = X_ptbxl.shape[1]
            X_arr_resampled = np.zeros((X_arr.shape[0], target_length, X_arr.shape[2]))
            for i in range(X_arr.shape[0]):
                for lead in range(X_arr.shape[2]):
                    X_arr_resampled[i, :, lead] = np.interp(
                        np.linspace(0, 1, target_length),
                        np.linspace(0, 1, X_arr.shape[1]),
                        X_arr[i, :, lead]
                    )
            X_arr = X_arr_resampled
        
        # Combine
        X_combined = np.vstack([X_ptbxl, X_arr])
        labels_combined = np.concatenate([labels_ptbxl, labels_arr_numeric])
        ids_combined = list(ids_ptbxl) + [f"arr_{id_}" for id_ in ids_arr]
        
        print(f"   OK: Combined dataset created")
        print(f"   Total records: {len(X_combined)}")
        print(f"   PTB-XL: {len(X_ptbxl)}, ECG Arrhythmia: {len(X_arr)}")
    else:
        # Use PTB-XL only
        X_combined = X_ptbxl
        labels_combined = labels_ptbxl  
        ids_combined = ids_ptbxl
        print(f"   Using PTB-XL only: {len(X_combined)} records")
    
    # Step 4: Check final MI count
    total_mi_count = np.sum(labels_combined == 1)
    print(f"\n   Final MI records: {total_mi_count}")
    print(f"   Total records: {len(X_combined)}")
    print(f"   MI percentage: {total_mi_count/len(X_combined)*100:.1f}%")
    
    if total_mi_count >= 50:
        print("   OK: Sufficient MI data for training!")
    else:
        print("   WARNING: Limited MI data - results may vary")
    
    return X_combined, labels_combined, ids_combined


def train_enhanced_mi_model(X, labels, ids):
    """Train MI-enhanced model with the combined dataset"""
    print("\n" + "=" * 60)
    print("TRAINING MI-ENHANCED MODEL")
    print("=" * 60)
    
    # Import training components
    from models.training.training_pipeline import ModelTrainingPipeline
    from config.model_config import ModelConfig
    
    # Use clinical-optimized configuration
    config = ModelConfig('clinical_optimized')
    trainer = ModelTrainingPipeline(config)
    
    # Prepare data
    print("Preparing training data...")
    
    # Target conditions mapping
    target_conditions = {0: 'NORM', 1: 'MI', 2: 'STTC', 3: 'CD', 4: 'HYP'}
    
    # Train the model
    print("Starting training...")
    results = trainer.train_and_evaluate(
        X, labels, ids, 
        model_name='enhanced_mi_model',
        save_model=True
    )
    
    return results


def main():
    """Main enhanced MI training workflow"""
    print("ENHANCED MI DETECTION TRAINING")
    print("=" * 60)
    print("Smart combination of PTB-XL + ECG Arrhythmia datasets")
    print("Fallback to PTB-XL if ECG Arrhythmia has issues")
    print("=" * 60)
    
    # Load enhanced dataset
    dataset_result = load_enhanced_mi_dataset()
    
    if dataset_result is None:
        print("\nERROR: Dataset loading failed completely")
        return False
    
    X, labels, ids = dataset_result
    
    if len(X) == 0:
        print("\nERROR: No data loaded - cannot proceed")
        return False
    
    # Train model
    try:
        results = train_enhanced_mi_model(X, labels, ids)
        
        print("\n" + "=" * 60)
        print("✅ ENHANCED MI TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Model performance:")
        print(f"  Overall accuracy: {results.get('accuracy', 'N/A')}")
        print(f"  MI sensitivity: {results.get('mi_sensitivity', 'N/A')}")
        print(f"  Model saved: {results.get('model_path', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nSUCCESS: Your MI detection is now enhanced!")
        print("Next: streamlit run app/main.py")
    else:
        print("\nWARNING: Training had issues - check logs above")
        print("Fallback: Use existing PTB-XL model for now")