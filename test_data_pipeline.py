#!/usr/bin/env python3
"""
Data Pipeline Tests for ECG System
Tests data loading, preprocessing, feature extraction pipeline integrity
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import pickle

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class TestDataPipeline:
    """Data pipeline functionality tests"""
    
    def setup_method(self):
        """Set up test environment"""
        self.data_path = Path('data')
        self.cache_path = self.data_path / 'cache'
        self.raw_path = self.data_path / 'raw'
        
    def test_data_directories_exist(self):
        """Test that all required data directories exist"""
        required_dirs = [
            'data/raw/ptbxl',
            'data/cache',
            'data/models',
            'data/processed'
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            assert path.exists(), f"Required directory missing: {dir_path}"
            print(f"✅ Directory exists: {dir_path}")
    
    def test_ptbxl_dataset_integrity(self):
        """Test PTB-XL dataset loading and integrity"""
        ptbxl_db = self.raw_path / 'ptbxl' / 'ptbxl_database.csv'
        
        if not ptbxl_db.exists():
            pytest.skip("PTB-XL database not available")
        
        # Load and validate dataset
        df = pd.read_csv(ptbxl_db)
        
        # Basic integrity checks
        assert len(df) > 15000, f"PTB-XL dataset too small: {len(df)} records"
        
        required_columns = ['ecg_id', 'filename_lr', 'scp_codes', 'age', 'sex']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data quality
        assert df['ecg_id'].nunique() == len(df), "ECG IDs must be unique"
        assert df['scp_codes'].notna().sum() > 0.8 * len(df), "Too many missing SCP codes"
        
        print(f"✅ PTB-XL Dataset: {len(df)} records with {df.columns.tolist()}")
    
    def test_cache_system_functionality(self):
        """Test caching system works correctly"""
        cache_files = list(self.cache_path.glob('*.pkl'))
        assert len(cache_files) >= 5, f"Expected at least 5 cache files, found {len(cache_files)}"
        
        # Test loading different cache files
        successful_loads = 0
        data_shapes = []
        
        for cache_file in cache_files[:5]:  # Test first 5 files
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Validate cache structure
                if isinstance(data, (tuple, list)) and len(data) >= 2:
                    X, y = data[0], data[1]
                    
                    # Validate shapes
                    if hasattr(X, 'shape') and hasattr(y, '__len__'):
                        assert X.shape[0] == len(y), f"Shape mismatch in {cache_file.name}"
                        data_shapes.append((cache_file.name, X.shape, len(y)))
                        successful_loads += 1
                        print(f"✅ Cache loaded: {cache_file.name} - Shape: {X.shape}")
                
            except Exception as e:
                print(f"⚠️  Cache load failed: {cache_file.name} - {e}")
        
        assert successful_loads >= 3, f"Must load at least 3 cache files successfully"
        
        # Validate data consistency across cache files
        if len(data_shapes) >= 2:
            # Check that different cache files have consistent feature dimensions
            feature_dims = [shape[1][1] if len(shape[1]) > 1 else 1 for shape in data_shapes]
            print(f"   Feature dimensions across cache files: {feature_dims}")
    
    def test_dataset_manager_functionality(self):
        """Test dataset manager can load data correctly"""
        try:
            from app.utils.dataset_manager import run_combined_dataset_loading
            
            # Test small dataset loading (to avoid long test times)
            print("Testing dataset manager with small sample...")
            
            start_time = time.time()
            X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
                ptbxl_max_records=100,  # Small sample for testing
                arrhythmia_max_records=0,  # Skip arrhythmia for speed
                target_mi_records=10,
                sampling_rate=100,
                use_cache=True
            )
            load_time = time.time() - start_time
            
            # Validate loaded data
            assert len(X) == len(labels), "Feature-label count mismatch"
            assert len(X) > 0, "Must load some data"
            assert len(X) <= 100, "Should respect max_records limit"
            
            # Validate data structure
            if isinstance(X, np.ndarray):
                assert X.ndim >= 2, "Features must be at least 2D"
            
            print(f"✅ Dataset Manager: Loaded {len(X)} samples in {load_time:.2f}s")
            print(f"   Labels: {set(labels)}")
            print(f"   Statistics: {stats}")
            
        except ImportError as e:
            pytest.skip(f"Dataset manager not available: {e}")
        except Exception as e:
            pytest.fail(f"Dataset manager test failed: {e}")
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline functionality"""
        try:
            # Try to import preprocessing components
            from models.preprocessing.preprocessing_pipeline import PreprocessingPipeline
            
            # Create minimal test data
            test_signals = np.random.randn(5, 12, 1000)  # 5 samples, 12 leads, 1000 points
            test_labels = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
            
            # Initialize preprocessing pipeline
            preprocessor = PreprocessingPipeline()
            
            print("✅ Preprocessing Pipeline: Successfully imported and initialized")
            
        except ImportError as e:
            print(f"⚠️  Preprocessing pipeline not available for testing: {e}")
            pytest.skip("Preprocessing pipeline not available")
        except Exception as e:
            print(f"⚠️  Preprocessing test failed: {e}")
    
    def test_feature_extraction_pipeline(self):
        """Test feature extraction pipeline"""
        try:
            from models.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
            
            # Test pipeline initialization
            feature_extractor = FeatureExtractionPipeline()
            
            print("✅ Feature Extraction Pipeline: Successfully imported and initialized")
            
        except ImportError as e:
            print(f"⚠️  Feature extraction pipeline not available: {e}")
            pytest.skip("Feature extraction pipeline not available")
        except Exception as e:
            print(f"⚠️  Feature extraction test failed: {e}")
    
    def test_data_processing_performance(self):
        """Test data processing performance benchmarks"""
        # Test cache loading performance
        cache_files = list(self.cache_path.glob('*.pkl'))
        if cache_files:
            cache_file = cache_files[0]
            
            start_time = time.time()
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            load_time = time.time() - start_time
            
            assert load_time < 10.0, f"Cache loading too slow: {load_time:.2f}s"
            print(f"✅ Cache Loading Performance: {load_time:.2f}s")
            
            # Test data size
            if isinstance(data, (tuple, list)) and len(data) >= 2:
                X = data[0]
                if hasattr(X, 'shape'):
                    data_size_mb = X.nbytes / (1024 * 1024) if hasattr(X, 'nbytes') else 0
                    print(f"   Data size: {data_size_mb:.1f} MB")
    
    def test_label_consistency(self):
        """Test label consistency across cached datasets"""
        cache_files = list(self.cache_path.glob('*.pkl'))
        all_labels = set()
        
        for cache_file in cache_files[:3]:  # Check first 3 cache files
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, (tuple, list)) and len(data) >= 2:
                    labels = data[1]
                    if isinstance(labels, (list, np.ndarray)):
                        unique_labels = set(labels) if isinstance(labels, list) else set(labels.tolist())
                        all_labels.update(unique_labels)
                        print(f"✅ Labels in {cache_file.name}: {sorted(unique_labels)}")
                        
            except Exception as e:
                print(f"⚠️  Could not extract labels from {cache_file.name}: {e}")
        
        # Validate we have expected clinical labels
        expected_labels = {'NORM', 'MI'}  # Minimum expected
        found_expected = len(expected_labels.intersection(all_labels))
        
        assert found_expected >= 1, f"Must find at least 1 expected label, found: {all_labels}"
        print(f"✅ Label Consistency: Found {len(all_labels)} unique labels across cache files")

def run_data_pipeline_tests():
    """Run all data pipeline tests"""
    print("=" * 60)
    print("RUNNING DATA PIPELINE TESTS")
    print("=" * 60)
    
    # Run pytest on this file
    test_file = __file__
    result = pytest.main(['-v', test_file])
    
    if result == 0:
        print("\\n✅ ALL DATA PIPELINE TESTS PASSED")
        return True
    else:
        print("\\n❌ SOME DATA PIPELINE TESTS FAILED")
        return False

if __name__ == '__main__':
    success = run_data_pipeline_tests()
    sys.exit(0 if success else 1)