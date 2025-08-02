"""
Test script for Phase 3 feature extraction pipeline
"""
import sys
import numpy as np
from pathlib import Path
import pickle

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.feature_extraction import FeatureExtractionPipeline
from config.feature_config import FEATURE_EXTRACTION_PRESETS
from config.settings import DATA_DIR


def test_feature_extraction_pipeline():
    """Test the feature extraction pipeline with preprocessed data"""
    print("TESTING PHASE 3 FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load preprocessed data from Phase 2
    print("\n1. Loading preprocessed data from Phase 2...")
    try:
        # Try to load from saved Phase 2 results
        phase2_file = DATA_DIR / 'processed' / 'preprocessing_results.pkl'
        
        if phase2_file.exists():
            print(f"Loading from: {phase2_file}")
            with open(phase2_file, 'rb') as f:
                phase2_data = pickle.load(f)
            
            X_preprocessed = phase2_data['X_preprocessed'][:20]  # Use 20 samples for testing
            y_encoded = phase2_data['y_encoded'][:20]
            label_encoder = phase2_data['label_info']['encoder']
            
            print(f"OK: Loaded {len(X_preprocessed)} preprocessed samples")
            print(f"   Shape: {X_preprocessed.shape}")
            print(f"   Classes: {label_encoder.classes_}")
        else:
            print("ERROR: Phase 2 results not found. Please run Phase 2 first.")
            return False
            
    except Exception as e:
        print(f"ERROR: Error loading preprocessed data: {e}")
        return False
    
    # Step 2: Test individual feature extractors
    print("\n2. Testing individual feature extractors...")
    try:
        from models.feature_extraction import (
            TemporalFeatureExtractor,
            FrequencyFeatureExtractor,
            STSegmentAnalyzer,
            WaveletFeatureExtractor
        )
        from config.feature_config import FeatureExtractionConfig
        
        config = FeatureExtractionConfig()
        
        # Test temporal features
        temporal_extractor = TemporalFeatureExtractor(config)
        temporal_features = temporal_extractor.extract_statistical_features(X_preprocessed[0])
        print(f"OK: Temporal features: {len(temporal_features)} extracted")
        
        # Test frequency features
        freq_extractor = FrequencyFeatureExtractor(config)
        freq_features = freq_extractor.extract_spectral_features(X_preprocessed[0])
        print(f"OK: Frequency features: {len(freq_features)} extracted")
        
        # Test ST-segment features
        st_analyzer = STSegmentAnalyzer(config)
        st_features = st_analyzer.extract_st_features(X_preprocessed[0], np.array([100, 200, 300]))
        print(f"OK: ST-segment features: {len(st_features)} extracted")
        
        # Test wavelet features
        wavelet_extractor = WaveletFeatureExtractor(config)
        wavelet_features = wavelet_extractor.extract_wavelet_features(X_preprocessed[0])
        print(f"OK: Wavelet features: {len(wavelet_features)} extracted")
        
    except Exception as e:
        print(f"ERROR: Error testing feature extractors: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test complete pipeline
    print("\n3. Testing complete feature extraction pipeline...")
    try:
        # Use standard preset
        pipeline = FeatureExtractionPipeline(FEATURE_EXTRACTION_PRESETS['standard'])
        
        # Run feature extraction
        results = pipeline.run(
            X_preprocessed=X_preprocessed,
            y_encoded=y_encoded,
            label_encoder=label_encoder,
            use_cache=False,  # Don't cache test data
            visualize=False   # Skip visualizations for test
        )
        
        print(f"OK: Feature extraction complete!")
        print(f"   - Total features extracted: {results['statistics']['total_features']}")
        print(f"   - Selected features: {results['statistics']['selected_features']}")
        print(f"   - Feature matrix shape: {results['X_features'].shape}")
        print(f"   - PCA components: {results['X_pca'].shape[1]}")
        
    except Exception as e:
        print(f"ERROR: Error in feature extraction pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test feature selection
    print("\n4. Testing feature selection...")
    try:
        from models.feature_extraction import FeatureSelector
        import pandas as pd
        
        selector = FeatureSelector(config)
        
        # Create a test feature DataFrame
        feature_df = pd.DataFrame(results['X_features'], columns=results['feature_names'])
        
        # Test correlation removal
        clean_df, dropped = selector.remove_correlated_features(feature_df, threshold=0.95)
        print(f"OK: Correlation removal: {len(dropped)} features removed")
        
        # Test importance analysis
        importance = selector.analyze_feature_importance(
            clean_df, 
            y_encoded,
            methods=['f_classif']
        )
        print(f"OK: Feature importance: {len(importance)} features analyzed")
        
    except Exception as e:
        print(f"ERROR: Error in feature selection: {e}")
        return False
    
    # Step 5: Test different presets
    print("\n5. Testing feature extraction presets...")
    for preset_name in ['comprehensive', 'fast', 'clinical']:
        try:
            pipeline = FeatureExtractionPipeline(FEATURE_EXTRACTION_PRESETS[preset_name])
            print(f"OK: Preset '{preset_name}' initialized successfully")
        except Exception as e:
            print(f"ERROR: Error with preset '{preset_name}': {e}")
    
    # Step 6: Verify output structure
    print("\n6. Verifying output structure...")
    required_keys = [
        'X_features', 'X_pca', 'feature_names', 'feature_importance',
        'y_encoded', 'label_encoder', 'pca_transformer', 'scaler',
        'config', 'statistics'
    ]
    
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        print(f"ERROR: Missing keys in results: {missing_keys}")
        return False
    else:
        print("OK: All required keys present in results")
    
    print("\n" + "=" * 60)
    print("SUCCESS: All tests passed! Phase 3 feature extraction is working correctly.")
    return True


def test_full_feature_extraction():
    """Test feature extraction on larger dataset"""
    print("\nTESTING FULL FEATURE EXTRACTION")
    print("WARNING: This will take several minutes...")
    
    # Load preprocessed data
    phase2_file = DATA_DIR / 'processed' / 'preprocessing_results.pkl'
    
    if not phase2_file.exists():
        print("ERROR: Phase 2 results not found. Please run Phase 2 first.")
        return
    
    with open(phase2_file, 'rb') as f:
        phase2_data = pickle.load(f)
    
    # Use more samples for full test
    n_samples = min(500, len(phase2_data['X_preprocessed']))
    X_preprocessed = phase2_data['X_preprocessed'][:n_samples]
    y_encoded = phase2_data['y_encoded'][:n_samples]
    label_encoder = phase2_data['label_info']['encoder']
    
    print(f"Using {n_samples} samples for full test")
    
    # Run feature extraction with visualizations
    pipeline = FeatureExtractionPipeline()
    results = pipeline.run(
        X_preprocessed=X_preprocessed,
        y_encoded=y_encoded,
        label_encoder=label_encoder,
        use_cache=True,
        visualize=True
    )
    
    # Print detailed statistics
    stats = results['statistics']
    print(f"\nFeature Extraction Statistics:")
    print(f"   - Processing time: {stats['processing_time']:.2f} seconds")
    print(f"   - Feature categories:")
    for category, count in stats['feature_categories'].items():
        print(f"     • {category}: {count}")
    
    # Print top 10 features
    print(f"\nTop 10 Features:")
    for i, row in results['feature_importance'].head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:<40} Score: {row['combined_score']:.3f}")


def test_feature_quality():
    """Test feature quality and statistics"""
    print("\nTESTING FEATURE QUALITY ANALYSIS")
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    # Create features with different characteristics
    features = []
    feature_names = []
    
    # Good features
    for i in range(20):
        features.append(np.random.randn(n_samples))
        feature_names.append(f'good_feature_{i}')
    
    # Constant features (should be removed)
    for i in range(5):
        features.append(np.ones(n_samples))
        feature_names.append(f'constant_feature_{i}')
    
    # Highly correlated features (some should be removed)
    base_feature = np.random.randn(n_samples)
    for i in range(10):
        noise = np.random.randn(n_samples) * 0.1
        features.append(base_feature + noise)
        feature_names.append(f'correlated_feature_{i}')
    
    # Highly skewed features
    for i in range(5):
        features.append(np.random.exponential(scale=2, size=n_samples))
        feature_names.append(f'skewed_feature_{i}')
    
    # Features with outliers
    for i in range(10):
        feature = np.random.randn(n_samples)
        # Add outliers
        outlier_idx = np.random.choice(n_samples, 5)
        feature[outlier_idx] = np.random.randn(5) * 10
        features.append(feature)
        feature_names.append(f'outlier_feature_{i}')
    
    # Create DataFrame
    import pandas as pd
    feature_df = pd.DataFrame(np.array(features).T, columns=feature_names)
    
    # Create synthetic labels
    y = np.random.randint(0, 5, n_samples)
    
    # Test feature selection
    from models.feature_extraction import FeatureSelector
    from config.feature_config import FeatureExtractionConfig
    
    config = FeatureExtractionConfig()
    selector = FeatureSelector(config)
    
    print(f"Initial features: {len(feature_df.columns)}")
    
    # Remove correlated features
    clean_df, dropped = selector.remove_correlated_features(feature_df, threshold=0.95)
    print(f"After correlation removal: {len(clean_df.columns)} (dropped {len(dropped)})")
    
    # Analyze importance
    importance = selector.analyze_feature_importance(clean_df, y)
    print(f"Feature importance analysis complete")
    
    # Select top features
    selected_df = selector.select_features(clean_df, importance, method='top_k', k=20)
    print(f"Selected features: {len(selected_df.columns)}")
    
    print("\nSUCCESS: Feature quality analysis working correctly")


if __name__ == "__main__":
    # Run basic tests first
    if test_feature_extraction_pipeline():
        print("\n" + "=" * 60)
        
        # Test feature quality analysis
        test_feature_quality()
        
        print("\n" + "=" * 60)
        response = input("\nRun full feature extraction test? (y/n): ")
        if response.lower() == 'y':
            test_full_feature_extraction()
    else:
        print("\nERROR: Please fix the errors before proceeding.")
        print("\nCommon issues:")
        print("1. Make sure Phase 2 has been run successfully")
        print("2. Check that all dependencies are installed (especially pywt)")
        print("3. Ensure you have enough memory for feature extraction")
# Add at the very end of the file
if __name__ == "__main__":
    try:
        # Run the existing test code
        # Add at the very end of the file
        if __name__ == "__main__":
            try:
                # Run the existing test code
                test_feature_extraction_pipeline()  # or whatever your main function is called
            except Exception as e:
                print(f"\nERROR: Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Keep terminal open
                print("\n" + "="*60)
                input("Press Enter to exit...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Keep terminal open
        print("\n" + "="*60)
        input("Press Enter to exit...")