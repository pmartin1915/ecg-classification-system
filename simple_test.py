"""
Very simple test - just PTB-XL dataset without arrhythmia
"""
import sys
import warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_ptbxl_only():
    """Test just PTB-XL dataset loading"""
    print("SIMPLE PTB-XL TEST")
    print("=" * 40)
    
    try:
        from app.utils.dataset_manager import DatasetManager
        
        manager = DatasetManager()
        print("Loading PTB-XL dataset only...")
        
        # Load just PTB-XL dataset
        ptbxl_data = manager.load_ptbxl_complete(
            max_records=20,
            sampling_rate=100,
            use_cache=True
        )
        
        X, labels, ids, metadata, target_conditions = ptbxl_data
        
        if len(X) > 0:
            print(f"OK: PTB-XL loaded successfully!")
            print(f"  Records: {len(X)}")
            print(f"  Shape: {X.shape}")
            
            # Check labels
            label_counts = Counter(labels)
            print(f"  Labels: {dict(label_counts)}")
            
            return True
        else:
            print("ERROR: No PTB-XL records loaded")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing():
    """Test preprocessing pipeline"""
    print("\nPREPROCESSING TEST")
    print("=" * 40)
    
    try:
        from app.utils.dataset_manager import DatasetManager
        from models.preprocessing.preprocessing_pipeline import PreprocessingPipeline
        
        # Load minimal data
        manager = DatasetManager()
        X, labels, ids, metadata, target_conditions = manager.load_ptbxl_complete(
            max_records=10,
            sampling_rate=100,
            use_cache=True
        )
        
        if len(X) == 0:
            print("ERROR: No data for preprocessing test")
            return False
        
        print("Running preprocessing...")
        preprocessing = PreprocessingPipeline()
        results = preprocessing.run(X, labels, max_records=len(X))
        
        if len(results['X_preprocessed']) > 0:
            print(f"OK: Preprocessing successful!")
            print(f"  Input: {X.shape}")
            print(f"  Output: {results['X_preprocessed'].shape}")
            return True
        else:
            print("ERROR: Preprocessing failed")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("SIMPLE DEVELOPMENT TEST")
    print("=" * 50)
    
    # Test 1: PTB-XL loading
    ptbxl_ok = test_ptbxl_only()
    
    # Test 2: Preprocessing
    if ptbxl_ok:
        preprocessing_ok = test_preprocessing()
        
        print(f"\n{'='*50}")
        print("SUMMARY:")
        print(f"PTB-XL loading: {'OK' if ptbxl_ok else 'FAILED'}")
        print(f"Preprocessing: {'OK' if preprocessing_ok else 'FAILED'}")
        
        if ptbxl_ok and preprocessing_ok:
            print("\nSUCCESS: Basic functionality working!")
            print("Next: Add ECG Arrhythmia dataset for MI enhancement")
        else:
            print("\nISSUES FOUND - check errors above")
    else:
        print("\nERROR: Basic PTB-XL loading failed")