"""
Debug version with immediate feedback - no hanging!
"""
import sys
import time
import os
from pathlib import Path
from datetime import datetime

print("=== PHASE 2 DEBUG TEST STARTING ===")
print(f"Time: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

# Add project root to Python path
project_root = Path(__file__).parent
print(f"Script location: {Path(__file__).absolute()}")
print(f"Project root: {project_root.absolute()}")
sys.path.append(str(project_root))
print(f"Added to Python path: {project_root}")
print()

def immediate_print(message):
    """Print with immediate flush"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    sys.stdout.flush()  # Force immediate output

# Test 1: Basic imports
immediate_print("STEP 1: Testing basic imports...")
try:
    import numpy as np
    immediate_print("SUCCESS: numpy imported")
    
    from pathlib import Path
    immediate_print("SUCCESS: pathlib imported")
    
    immediate_print("SUCCESS: All basic imports work")
except Exception as e:
    immediate_print(f"ERROR: Basic import failed: {e}")
    sys.exit(1)

# Test 2: Test config imports
immediate_print("STEP 2: Testing config imports...")
try:
    from config.settings import DATA_DIR
    immediate_print(f"SUCCESS: DATA_DIR = {DATA_DIR}")
    
    if DATA_DIR.exists():
        immediate_print(f"SUCCESS: DATA_DIR exists")
    else:
        immediate_print(f"WARNING: DATA_DIR does not exist yet")
        
except Exception as e:
    immediate_print(f"ERROR: Config import failed: {e}")
    immediate_print("This might be normal if you haven't run Phase 1 yet")

# Test 3: Test preprocessing imports
immediate_print("STEP 3: Testing preprocessing imports...")
try:
    from config.preprocessing_config import PREPROCESSING_PRESETS
    immediate_print("SUCCESS: PREPROCESSING_PRESETS imported")
    
    from config.preprocessing_config import PreprocessingConfig
    immediate_print("SUCCESS: PreprocessingConfig imported")
    
    config = PreprocessingConfig()
    immediate_print(f"SUCCESS: Config created with sampling_rate = {config.sampling_rate}")
    
except Exception as e:
    immediate_print(f"ERROR: Preprocessing config failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test model imports
immediate_print("STEP 4: Testing model imports...")
try:
    from models.preprocessing.signal_quality import SignalQualityAssessor
    immediate_print("SUCCESS: SignalQualityAssessor imported")
    
    from models.preprocessing.signal_filters import ECGFilterBank
    immediate_print("SUCCESS: ECGFilterBank imported")
    
    from models.preprocessing.preprocessing_pipeline import PreprocessingPipeline
    immediate_print("SUCCESS: PreprocessingPipeline imported")
    
except Exception as e:
    immediate_print(f"ERROR: Model import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test data manager import (this might be slow)
immediate_print("STEP 5: Testing data manager import...")
immediate_print("WARNING: This might take a moment...")
try:
    from app.utils.dataset_manager import DatasetManager
    immediate_print("SUCCESS: DatasetManager imported")
    
except Exception as e:
    immediate_print(f"ERROR: DatasetManager import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Create dummy data test
immediate_print("STEP 6: Testing with dummy data...")
try:
    # Create simple test data
    import numpy as np
    dummy_data = np.random.randn(3, 1000, 12)  # 3 samples, 1000 time points, 12 leads
    immediate_print(f"SUCCESS: Created dummy data with shape {dummy_data.shape}")
    
    # Test quality assessor on dummy data
    config = PreprocessingConfig()
    assessor = SignalQualityAssessor(config)
    immediate_print("SUCCESS: Quality assessor created")
    
    # Test on one signal
    quality = assessor.assess_signal(dummy_data[0])
    immediate_print(f"SUCCESS: Quality assessment complete")
    immediate_print(f"  - Signal valid: {quality['is_valid']}")
    immediate_print(f"  - SNR estimate: {quality.get('snr_estimate', 'N/A')}")
    
except Exception as e:
    immediate_print(f"ERROR: Dummy data test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Check for cached data (without loading)
immediate_print("STEP 7: Checking for cached data...")
try:
    phase1_file = DATA_DIR / 'processed' / 'phase1_output.pkl'
    if phase1_file.exists():
        file_size = phase1_file.stat().st_size / (1024*1024)  # MB
        immediate_print(f"SUCCESS: Found cached Phase 1 data ({file_size:.1f} MB)")
        immediate_print(f"  Location: {phase1_file}")
    else:
        immediate_print("INFO: No cached Phase 1 data found")
        immediate_print("  This is normal if you haven't run Phase 1 yet")
        
except Exception as e:
    immediate_print(f"ERROR: Cache check failed: {e}")

# Test 8: Try to create a pipeline (without running it)
immediate_print("STEP 8: Testing pipeline creation...")
try:
    pipeline = PreprocessingPipeline(PREPROCESSING_PRESETS['fast'])
    immediate_print("SUCCESS: Pipeline created with 'fast' preset")
    
    pipeline = PreprocessingPipeline(PREPROCESSING_PRESETS['standard'])
    immediate_print("SUCCESS: Pipeline created with 'standard' preset")
    
except Exception as e:
    immediate_print(f"ERROR: Pipeline creation failed: {e}")
    import traceback
    traceback.print_exc()

# Final status
immediate_print("=" * 50)
immediate_print("DEBUG TEST COMPLETE!")
immediate_print("=" * 50)

# Decision point
immediate_print("All basic components are working!")
immediate_print("")
immediate_print("Next step options:")
immediate_print("1. If you have Phase 1 data cached, we can test loading it")
immediate_print("2. If not, we can test with very minimal real data")
immediate_print("3. Or we can test the full pipeline with dummy data")
immediate_print("")

response = input("Which would you like to try? (1/2/3 or 'q' to quit): ").strip()

if response == '1':
    immediate_print("TESTING: Loading cached Phase 1 data...")
    try:
        import pickle
        phase1_file = DATA_DIR / 'processed' / 'phase1_output.pkl'
        
        immediate_print("Loading cached data (this might take 10-30 seconds)...")
        start_time = time.time()
        
        with open(phase1_file, 'rb') as f:
            phase1_data = pickle.load(f)
        
        elapsed = time.time() - start_time
        immediate_print(f"SUCCESS: Loaded in {elapsed:.1f} seconds!")
        immediate_print(f"  Shape: {phase1_data['X'].shape}")
        immediate_print(f"  Samples available: {len(phase1_data['X'])}")
        
    except Exception as e:
        immediate_print(f"ERROR: Loading failed: {e}")

elif response == '2':
    immediate_print("TESTING: Loading minimal real data...")
    try:
        immediate_print("Creating DatasetManager (might take a moment)...")
        manager = DatasetManager()
        
        immediate_print("Loading 2 records (this will take 1-5 minutes)...")
        immediate_print("Please wait - downloading/processing ECG data...")
        start_time = time.time()
        
        results = manager.load_ptbxl_complete(max_records=2, use_cache=True)
        
        elapsed = time.time() - start_time
        immediate_print(f"SUCCESS: Loaded in {elapsed:.1f} seconds!")
        immediate_print(f"  Shape: {results['X'].shape}")
        
    except Exception as e:
        immediate_print(f"ERROR: Real data loading failed: {e}")
        import traceback
        traceback.print_exc()

elif response == '3':
    immediate_print("TESTING: Full pipeline with dummy data...")
    try:
        # Create more realistic dummy ECG data
        np.random.seed(42)  # Reproducible
        n_samples = 3
        n_timepoints = 1000
        n_leads = 12
        
        # Generate somewhat ECG-like signals
        t = np.linspace(0, 10, n_timepoints)
        dummy_X = []
        
        for i in range(n_samples):
            # Simple ECG-like signal with different frequencies per lead
            signal = np.zeros((n_timepoints, n_leads))
            for lead in range(n_leads):
                freq = 1.0 + 0.2 * lead  # Different frequency per lead
                signal[:, lead] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(n_timepoints)
            dummy_X.append(signal)
        
        dummy_X = np.array(dummy_X)
        dummy_labels = ['NORM', 'MI', 'STTC']
        dummy_ids = ['test_001', 'test_002', 'test_003']
        
        immediate_print(f"Created dummy data: {dummy_X.shape}")
        
        # Run preprocessing pipeline
        immediate_print("Running preprocessing pipeline...")
        pipeline = PreprocessingPipeline(PREPROCESSING_PRESETS['fast'])
        
        start_time = time.time()
        results = pipeline.run(
            X=dummy_X,
            labels=dummy_labels,
            ids=dummy_ids,
            use_cache=False,
            visualize=False
        )
        
        elapsed = time.time() - start_time
        immediate_print(f"SUCCESS: Pipeline completed in {elapsed:.1f} seconds!")
        immediate_print(f"  Input shape: {dummy_X.shape}")
        immediate_print(f"  Output shape: {results['X_preprocessed'].shape}")
        
    except Exception as e:
        immediate_print(f"ERROR: Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

else:
    immediate_print("Exiting. Your Phase 2 setup looks good!")

immediate_print("\nDone!")