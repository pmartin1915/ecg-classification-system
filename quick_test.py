"""
Quick test for combined dataset functionality - optimized for speed
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

def quick_combined_test():
    """Quick test of combined dataset with minimal records"""
    print("QUICK COMBINED DATASET TEST")
    print("=" * 50)
    
    try:
        from app.utils.dataset_manager import run_combined_dataset_loading
        
        print("Loading minimal combined dataset...")
        X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
            ptbxl_max_records=25,     # Very small PTB-XL subset
            arrhythmia_max_records=5, # Very small ECG Arrhythmia subset  
            target_mi_records=3,      # Just a few MI records
            sampling_rate=100,
            use_cache=True
        )
        
        if len(X) > 0:
            print(f"OK: Combined dataset loaded successfully!")
            print(f"  Total records: {len(X)}")
            print(f"  Signal shape: {X.shape}")
            print(f"  Memory usage: {stats['memory_gb']:.3f} GB")
            
            # Check label distribution
            label_counts = Counter(labels)
            print(f"  Label distribution: {dict(label_counts)}")
            
            mi_count = label_counts.get('MI', 0)
            print(f"  MI records: {mi_count}")
            
            if mi_count > 0:
                print("  SUCCESS: MI records found!")
            else:
                print("  WARNING: No MI records in this tiny subset")
            
            return True
        else:
            print("ERROR: No records loaded")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_pipeline_test():
    """Quick test of preprocessing pipeline"""
    print("\nQUICK PIPELINE TEST")
    print("=" * 50)
    
    try:
        from models.preprocessing.preprocessing_pipeline import PreprocessingPipeline
        from app.utils.dataset_manager import run_combined_dataset_loading
        
        # Load minimal data
        print("Loading tiny dataset for pipeline test...")
        X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
            ptbxl_max_records=10,
            arrhythmia_max_records=2,
            target_mi_records=1,
            sampling_rate=100,
            use_cache=True
        )
        
        if len(X) == 0:
            print("✗ No data for pipeline test")
            return False
        
        print("Running preprocessing...")
        preprocessing = PreprocessingPipeline()
        results = preprocessing.run(X, labels, max_records=len(X))
        
        if len(results['X_preprocessed']) > 0:
            print(f"OK: Preprocessing successful!")
            print(f"  Processed records: {len(results['X_preprocessed'])}")
            print(f"  Output shape: {results['X_preprocessed'].shape}")
            return True
        else:
            print("ERROR: Preprocessing failed")
            return False
            
    except Exception as e:
        print(f"ERROR: Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("QUICK DEVELOPMENT TEST")
    print("=" * 60)
    
    # Test 1: Combined dataset loading
    dataset_ok = quick_combined_test()
    
    # Test 2: Pipeline processing
    if dataset_ok:
        pipeline_ok = quick_pipeline_test()
        
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"Dataset loading: {'OK' if dataset_ok else 'FAILED'}")
        print(f"Pipeline processing: {'OK' if pipeline_ok else 'FAILED'}")
        
        if dataset_ok and pipeline_ok:
            print("\nSUCCESS: Core functionality working! Ready for full dataset.")
        else:
            print("\nWARNING: Issues found - check errors above.")
    else:
        print("\nERROR: Cannot test pipeline - dataset loading failed")