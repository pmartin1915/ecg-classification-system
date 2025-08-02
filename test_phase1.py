"""
Test script for Phase 1 data loading
Run this to verify your setup is working correctly
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.utils.dataset_manager import DatasetManager, run_phase1_foundation


def test_basic_loading():
    """Test basic data loading functionality"""
    print("TESTING PHASE 1 DATA LOADING")
    print("=" * 60)
    
    # Test 1: Load a tiny subset
    print("\n1. Testing with 10 records...")
    try:
        X, labels, ids, Y_filtered, TARGET_CONDITIONS, _ = run_phase1_foundation(
            max_records=10,
            sampling_rate=100,
            use_cache=False  # Don't cache test data
        )
        
        if len(X) > 0:
            print("OK: Basic loading test passed!")
            print(f"   Loaded shape: {X.shape}")
        else:
            print("ERROR: No data loaded")
    except Exception as e:
        print(f"ERROR: Error: {e}")
        return False
    
    # Test 2: Dataset Manager
    print("\n2. Testing DatasetManager...")
    try:
        manager = DatasetManager()
        results = manager.load_ptbxl_complete(max_records=5)
        
        if results['stats']['total_records'] > 0:
            print("OK: DatasetManager test passed!")
        else:
            print("ERROR: DatasetManager failed to load data")
    except Exception as e:
        print(f"ERROR: Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("SUCCESS: All tests passed! Your Phase 1 setup is working correctly.")
    return True


def test_full_loading():
    """Test loading the full dataset (use with caution - takes time!)"""
    print("\nLOADING FULL DATASET")
    print("WARNING: This will take several minutes...")
    
    manager = DatasetManager()
    results = manager.load_ptbxl_complete(
        max_records=None,  # Load all records
        sampling_rate=100,
        use_cache=True
    )
    
    # Get train/test split
    if results['stats']['total_records'] > 0:
        train_data, test_data = manager.get_train_test_split(results)
        print(f"\nTrain set size: {len(train_data['X'])}")
        print(f"Test set size: {len(test_data['X'])}")


if __name__ == "__main__":
    # Run basic tests first
    if test_basic_loading():
        print("\n" + "=" * 60)
        response = input("\nRun full dataset loading test? (y/n): ")
        if response.lower() == 'y':
            test_full_loading()
    else:
        print("\nERROR: Please fix the errors before proceeding.")
        print("\nCommon issues:")
        print("1. Make sure you've installed all requirements: pip install -r requirements.txt")
        print("2. Check that you have internet connection for downloading data")
        print("3. Ensure you have enough disk space for the datasets")