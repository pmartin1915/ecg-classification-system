"""
Working combined dataset test with memory optimizations
"""
import gc
import sys
import warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_ptbxl_dataset():
    """Test PTB-XL dataset loading with minimal records"""
    print("TESTING PTB-XL DATASET")
    print("=" * 40)
    
    try:
        from app.utils.dataset_manager import DatasetManager
        
        manager = DatasetManager()
        print("Loading PTB-XL dataset (10 records)...")
        
        # Load just PTB-XL dataset with very small number of records
        ptbxl_results = manager.load_ptbxl_complete(
            max_records=10,
            sampling_rate=100,
            use_cache=False  # Don't cache to save memory
        )
        
        X = ptbxl_results['X']
        labels = ptbxl_results['labels']
        ids = ptbxl_results['ids']
        metadata = ptbxl_results['metadata']
        target_conditions = ptbxl_results['target_conditions']
        stats = ptbxl_results['stats']
        
        if len(X) > 0:
            print(f"OK: PTB-XL loaded successfully!")
            print(f"  Records: {len(X)}")
            print(f"  Shape: {X.shape}")
            print(f"  Memory: {X.nbytes / (1024**2):.1f} MB")
            
            # Check labels (convert lists to strings if needed)
            if len(labels) > 0:
                # Handle case where labels might be lists
                if isinstance(labels[0], list):
                    label_strings = [str(label) for label in labels]
                else:
                    label_strings = labels
                label_counts = Counter(label_strings)
                print(f"  Labels: {dict(label_counts)}")
            else:
                print("  Labels: No labels found")
            
            return True, X, labels, ids
        else:
            print("ERROR: No PTB-XL records loaded")
            return False, None, None, None
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None
    finally:
        gc.collect()

def test_preprocessing_pipeline(X, labels, ids):
    """Test preprocessing with loaded data"""
    print("\nTESING PREPROCESSING PIPELINE")
    print("=" * 40)
    
    try:
        from models.preprocessing.preprocessing_pipeline import PreprocessingPipeline
        
        print("Running preprocessing...")
        preprocessing = PreprocessingPipeline()
        results = preprocessing.run(X, labels, ids)
        
        if len(results['X_preprocessed']) > 0:
            print(f"OK: Preprocessing successful!")
            print(f"  Input shape: {X.shape}")
            print(f"  Output shape: {results['X_preprocessed'].shape}")
            print(f"  Processing time: {results.get('processing_time', 'N/A')}")
            return True, results
        else:
            print("ERROR: Preprocessing failed")
            return False, None
            
    except Exception as e:
        print(f"ERROR: Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        gc.collect()

def test_feature_extraction(preprocessing_results):
    """Test feature extraction with preprocessed data"""
    print("\nTESTING FEATURE EXTRACTION")
    print("=" * 40)
    
    try:
        from models.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
        
        print("Running feature extraction...")
        feature_pipeline = FeatureExtractionPipeline()
        
        # Check what's available in preprocessing results
        print(f"  Available keys: {list(preprocessing_results.keys())}")
        
        # Use label_info instead of label_encoder if it exists
        label_encoder = preprocessing_results.get('label_encoder', preprocessing_results.get('label_info', None))
        
        feature_results = feature_pipeline.run(
            preprocessing_results['X_preprocessed'],
            preprocessing_results['y_encoded'],
            label_encoder,
            max_features=20  # Keep features low for memory
        )
        
        if len(feature_results['X_features']) > 0:
            print(f"OK: Feature extraction successful!")
            print(f"  Input shape: {preprocessing_results['X_preprocessed'].shape}")
            print(f"  Features shape: {feature_results['X_features'].shape}")
            print(f"  Feature count: {len(feature_results['feature_names'])}")
            return True, feature_results
        else:
            print("ERROR: Feature extraction failed")
            return False, None
            
    except Exception as e:
        print(f"ERROR: Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        gc.collect()

def main():
    """Main test function"""
    print("WORKING COMBINED DATASET TEST")
    print("=" * 50)
    
    # Step 1: Test PTB-XL dataset loading
    ptbxl_ok, X, labels, ids = test_ptbxl_dataset()
    
    preprocessing_ok = False
    feature_ok = False
    
    if ptbxl_ok and X is not None:
        # Step 2: Test preprocessing
        preprocessing_ok, preprocessing_results = test_preprocessing_pipeline(X, labels, ids)
        
        if preprocessing_ok and preprocessing_results is not None:
            # Step 3: Test feature extraction
            feature_ok, feature_results = test_feature_extraction(preprocessing_results)
    
    # Summary
    print(f"\n{'='*50}")
    print("PIPELINE TEST SUMMARY")
    print("=" * 50)
    print(f"1. PTB-XL Dataset Loading:  {'PASS' if ptbxl_ok else 'FAIL'}")
    print(f"2. Preprocessing Pipeline:  {'PASS' if preprocessing_ok else 'FAIL'}")
    print(f"3. Feature Extraction:      {'PASS' if feature_ok else 'FAIL'}")
    
    if ptbxl_ok and preprocessing_ok and feature_ok:
        print("\nSUCCESS: Complete pipeline working!")
        print("\nREADY FOR:")
        print("- Larger dataset sizes")
        print("- Combined dataset integration")
        print("- Model training (Phase 4)")
        print("- Production deployment")
    elif ptbxl_ok:
        print("\nPARTIAL SUCCESS: Basic loading works")
        print("Issues found in processing pipeline - check errors above")
    else:
        print("\nCRITICAL FAILURE: Basic dataset loading failed")
        print("System needs troubleshooting before proceeding")
    
    # Clean up memory
    gc.collect()

if __name__ == "__main__":
    main()