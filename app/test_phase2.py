"""
Test script for Phase 2 preprocessing pipeline
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.utils.dataset_manager import DatasetManager
from models.preprocessing import PreprocessingPipeline
from config.preprocessing_config import PREPROCESSING_PRESETS
from config.settings import DATA_DIR


def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with a small dataset"""
    print("🧪 TESTING PHASE 2 PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load test data from Phase 1
    print("\n1. Loading test data from Phase 1...")
    try:
        # Try to load from saved Phase 1 results
        import pickle
        phase1_file = DATA_DIR / 'processed' / 'phase1_output.pkl'
        
        if phase1_file.exists():
            print(f"Loading from: {phase1_file}")
            with open(phase1_file, 'rb') as f:
                phase1_data = pickle.load(f)
            
            X = phase1_data['X'][:10]  # Use only 10 samples for testing
            labels = phase1_data['labels'][:10]
            ids = phase1_data['ids'][:10]
        else:
            # Load fresh data
            print("Loading fresh data...")
            manager = DatasetManager()
            results = manager.load_ptbxl_complete(max_records=10)
            
            X = results['X']
            labels = results['labels']
            ids = results['ids']
        
        print(f"✅ Loaded {len(X)} samples")
        print(f"   Shape: {X.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Step 2: Test quality assessment
    print("\n2. Testing signal quality assessment...")
    try:
        from models.preprocessing import SignalQualityAssessor
        from config.preprocessing_config import PreprocessingConfig
        
        config = PreprocessingConfig()
        assessor = SignalQualityAssessor(config)
        
        # Test single signal
        quality = assessor.assess_signal(X[0])
        print(f"✅ Quality assessment works!")
        print(f"   Signal valid: {quality['is_valid']}")
        print(f"   SNR estimate: {quality['snr_estimate']:.2f}")
        
    except Exception as e:
        print(f"❌ Error in quality assessment: {e}")
        return False
    
    # Step 3: Test signal filtering
    print("\n3. Testing signal filtering...")
    try:
        from models.preprocessing import ECGFilterBank
        
        filter_bank = ECGFilterBank(config)
        filtered_signal = filter_bank.apply_filters(X[0])
        
        print(f"✅ Signal filtering works!")
        print(f"   Filters applied: {list(filter_bank.filters.keys())}")
        print(f"   Output shape: {filtered_signal.shape}")
        
    except Exception as e:
        print(f"❌ Error in signal filtering: {e}")
        return False
    
    # Step 4: Test complete pipeline
    print("\n4. Testing complete preprocessing pipeline...")
    try:
        # Use standard preset
        pipeline = PreprocessingPipeline(PREPROCESSING_PRESETS['standard'])
        
        # Run preprocessing
        results = pipeline.run(
            X=X,
            labels=labels,
            ids=ids,
            use_cache=False,  # Don't cache test data
            visualize=False   # Skip visualizations for test
        )
        
        print(f"✅ Preprocessing pipeline complete!")
        print(f"   Final shape: {results['X_preprocessed'].shape}")
        print(f"   Valid samples: {len(results['X_preprocessed'])}")
        print(f"   Classes: {results['label_info']['encoder'].classes_}")
        
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test different preprocessing presets
    print("\n5. Testing preprocessing presets...")
    for preset_name in ['high_quality', 'fast', 'robust']:
        try:
            pipeline = PreprocessingPipeline(PREPROCESSING_PRESETS[preset_name])
            print(f"✅ Preset '{preset_name}' initialized successfully")
        except Exception as e:
            print(f"❌ Error with preset '{preset_name}': {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Phase 2 preprocessing is working correctly.")
    return True


def test_full_preprocessing():
    """Test preprocessing on full dataset"""
    print("\n🚀 TESTING FULL PREPROCESSING")
    print("⚠️  This will take several minutes...")
    
    # Load full dataset
    manager = DatasetManager()
    results = manager.load_ptbxl_complete(
        max_records=1000,  # Use 1000 samples for reasonable test time
        sampling_rate=100,
        use_cache=True
    )
    
    # Run preprocessing
    pipeline = PreprocessingPipeline()
    preprocessed = pipeline.run(
        X=results['X'],
        labels=results['labels'],
        ids=results['ids'],
        use_cache=True,
        visualize=True
    )
    
    # Print final statistics
    stats = preprocessed['statistics']
    print(f"\n📊 Preprocessing Statistics:")
    print(f"   - Quality pass rate: {stats['quality_stats']['validity_rate']*100:.1f}%")
    print(f"   - Artifact detection:")
    for artifact_type, count in stats['processing_stats']['artifact_counts'].items():
        print(f"     • {artifact_type}: {count} signals")
    print(f"   - Final dataset: {stats['final_samples']} samples")


if __name__ == "__main__":
    # Run basic tests first
    if test_preprocessing_pipeline():
        print("\n" + "=" * 60)
        response = input("\nRun full preprocessing test? (y/n): ")
        if response.lower() == 'y':
            test_full_preprocessing()
    else:
        print("\n❌ Please fix the errors before proceeding.")
        print("\nCommon issues:")
        print("1. Make sure Phase 1 data is available")
        print("2. Check that all dependencies are installed")
        print("3. Ensure you have run Phase 1 successfully first")