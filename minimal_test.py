"""
Minimal test - just check imports and basic functionality
"""
import gc
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test basic imports work"""
    print("TESTING IMPORTS")
    print("=" * 30)
    
    try:
        from app.utils.dataset_manager import DatasetManager
        print("OK: DatasetManager import")
        
        from models.preprocessing.preprocessing_pipeline import PreprocessingPipeline
        print("OK: PreprocessingPipeline import")
        
        from models.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
        print("OK: FeatureExtractionPipeline import")
        
        return True
    except Exception as e:
        print(f"ERROR: Import failed: {e}")
        return False

def test_tiny_dataset():
    """Test loading just 3 records"""
    print("\nTEST TINY DATASET")
    print("=" * 30)
    
    try:
        from app.utils.data_loader import ECGDataLoader
        
        loader = ECGDataLoader(dataset_name="ptbxl")
        
        # Load metadata only first
        print("Loading metadata...")
        metadata_result = loader.load_metadata()
        Y, scp_statements = metadata_result  # Unpack tuple
        print(f"Metadata loaded: {len(Y)} records")
        
        # Filter to just 3 records to save memory
        Y_tiny = Y.head(3)
        print(f"Using tiny subset: {len(Y_tiny)} records")
        
        # Clear memory
        del Y
        del scp_statements
        gc.collect()
        
        # Try to load signals for just these 3 records
        print("Loading 3 signal records...")
        X, labels, ids = loader.load_signals(
            Y_tiny, 
            max_records=3,
            sampling_rate=100,
            use_cache=False  # Don't cache to save memory
        )
        
        if len(X) > 0:
            print(f"OK: Loaded {len(X)} records")
            print(f"Shape: {X.shape}")
            return True
        else:
            print("ERROR: No signals loaded")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up memory
        gc.collect()

if __name__ == "__main__":
    print("MINIMAL FUNCTIONALITY TEST")
    print("=" * 40)
    
    # Test 1: Basic imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test 2: Tiny dataset loading
        dataset_ok = test_tiny_dataset()
        
        print(f"\n{'='*40}")
        print("RESULTS:")
        print(f"Imports: {'OK' if imports_ok else 'FAILED'}")
        print(f"Dataset: {'OK' if dataset_ok else 'FAILED'}")
        
        if imports_ok and dataset_ok:
            print("\nSUCCESS: Core functionality working!")
            print("System ready for larger datasets.")
        else:
            print("\nBASIC ISSUES FOUND")
    else:
        print("\nCRITICAL: Import failures - check dependencies")