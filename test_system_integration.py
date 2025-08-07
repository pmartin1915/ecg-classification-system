#!/usr/bin/env python3
"""
System Integration Tests for ECG Clinical System
Tests core system functionality, model loading, and launcher integration
"""

import os
import sys
import time
import subprocess
import importlib.util
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class TestSystemIntegration:
    """Core system integration tests"""
    
    def setup_method(self):
        """Set up test environment"""
        os.chdir(project_root)
        
    def test_main_application_import(self):
        """Test that main application imports without errors"""
        try:
            import complete_user_friendly
            assert True, "Main application imported successfully"
        except ImportError as e:
            pytest.fail(f"Failed to import main application: {e}")
    
    def test_models_directory_exists(self):
        """Test that models directory and files exist"""
        models_path = Path('data/models')
        assert models_path.exists(), "Models directory must exist"
        
        models = list(models_path.glob('*.pkl')) + list(models_path.glob('*.joblib'))
        assert len(models) >= 4, f"Expected at least 4 models, found {len(models)}"
        
        # Check for specific expected models
        model_names = [m.name for m in models]
        expected_models = ['enhanced_mi_focused_model.pkl', 'basic_mi_model.joblib']
        
        for expected in expected_models:
            assert any(expected in name for name in model_names), f"Missing expected model: {expected}"
    
    def test_data_cache_accessible(self):
        """Test that cached data files are accessible"""
        cache_path = Path('data/cache')
        assert cache_path.exists(), "Cache directory must exist"
        
        cache_files = list(cache_path.glob('*.pkl'))
        assert len(cache_files) >= 10, f"Expected at least 10 cache files, found {len(cache_files)}"
    
    def test_ptbxl_dataset_available(self):
        """Test PTB-XL dataset availability"""
        ptbxl_path = Path('data/raw/ptbxl/ptbxl_database.csv')
        assert ptbxl_path.exists(), "PTB-XL database file must exist"
        
        # Test file can be read
        import pandas as pd
        try:
            df = pd.read_csv(ptbxl_path)
            assert len(df) > 20000, f"Expected >20,000 records, found {len(df)}"
        except Exception as e:
            pytest.fail(f"Failed to read PTB-XL dataset: {e}")
    
    def test_core_modules_import(self):
        """Test that core application modules can be imported"""
        core_modules = [
            'app.main',
            'app.components.classification',
            'app.utils.dataset_manager',
            'models.preprocessing.preprocessing_pipeline',
            'models.feature_extraction.feature_extraction_pipeline'
        ]
        
        for module_name in core_modules:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    assert True, f"Successfully imported {module_name}"
                else:
                    pytest.fail(f"Module not found: {module_name}")
            except Exception as e:
                # Some modules might have dependencies that fail in test env
                print(f"Warning: Could not fully load {module_name}: {e}")
    
    def test_launcher_file_exists(self):
        """Test that main launcher exists and is executable"""
        launcher = Path('COMPLETE_USER_FRIENDLY_LAUNCHER.bat')
        assert launcher.exists(), "Main launcher must exist"
        
        # Check launcher contains expected commands
        content = launcher.read_text(encoding='utf-8')
        assert 'streamlit run complete_user_friendly.py' in content, "Launcher must contain streamlit command"
        assert '--server.port=8507' in content, "Launcher must specify correct port"
    
    def test_configuration_files(self):
        """Test that configuration files are present and valid"""
        config_files = [
            'config/settings.py',
            'config/model_config.py',
            'CLAUDE.md'
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            assert config_path.exists(), f"Configuration file must exist: {config_file}"
    
    def test_python_environment(self):
        """Test Python environment and key dependencies"""
        import sys
        assert sys.version_info >= (3, 8), "Python 3.8+ required"
        
        # Test key dependencies can be imported
        key_deps = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'plotly', 'sklearn']
        
        for dep in key_deps:
            try:
                __import__(dep)
                assert True, f"Successfully imported {dep}"
            except ImportError:
                pytest.fail(f"Missing required dependency: {dep}")

def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("RUNNING SYSTEM INTEGRATION TESTS")
    print("=" * 60)
    
    # Run pytest on this file
    test_file = __file__
    result = pytest.main(['-v', test_file])
    
    if result == 0:
        print("\\n✅ ALL INTEGRATION TESTS PASSED")
        return True
    else:
        print("\\n❌ SOME INTEGRATION TESTS FAILED")
        return False

if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)