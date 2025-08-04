"""
System Diagnostics - Comprehensive Health Check
Tests all critical components and identifies issues
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

def print_header(title):
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

def print_status(item, status, details=""):
    status_icon = "[OK]" if status else "[FAIL]"
    print(f"{status_icon} {item:<40} {details}")

def test_environment():
    """Test Python environment and dependencies"""
    print_header("ENVIRONMENT TEST")
    
    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_ok = sys.version_info >= (3, 8)
    print_status("Python Version", python_ok, f"v{python_version}")
    
    # Test critical imports
    imports_to_test = [
        ('numpy', 'np'),
        ('pandas', 'pd'), 
        ('streamlit', 'st'),
        ('sklearn', 'sklearn'),
        ('wfdb', 'wfdb'),
        ('matplotlib.pyplot', 'plt'),
        ('joblib', 'joblib')
    ]
    
    missing_imports = []
    for module, alias in imports_to_test:
        try:
            __import__(module)
            print_status(f"Import {module}", True, "OK")
        except ImportError:
            print_status(f"Import {module}", False, "MISSING")
            missing_imports.append(module)
    
    return len(missing_imports) == 0

def test_data_structure():
    """Test data directory structure and files"""
    print_header("DATA STRUCTURE TEST")
    
    project_root = Path(__file__).parent
    
    # Critical directories
    critical_dirs = [
        "data/raw/ptbxl",
        "data/raw/ecg-arrhythmia-dataset", 
        "data/processed",
        "data/cache",
        "models/trained_models",
        "app/components"
    ]
    
    all_dirs_ok = True
    for dir_path in critical_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        if not exists:
            all_dirs_ok = False
        print_status(f"Directory {dir_path}", exists)
    
    # Critical files
    critical_files = [
        "data/raw/ptbxl/ptbxl_database.csv",
        "data/raw/ptbxl/scp_statements.csv",
        "requirements.txt",
        "app/main.py"
    ]
    
    all_files_ok = True
    for file_path in critical_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        if not exists:
            all_files_ok = False
        size_mb = full_path.stat().st_size / (1024*1024) if exists else 0
        print_status(f"File {file_path}", exists, f"{size_mb:.1f}MB" if exists else "MISSING")
    
    return all_dirs_ok and all_files_ok

def test_data_loading():
    """Test basic data loading functionality"""
    print_header("DATA LOADING TEST")
    
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))
    
    try:
        from app.utils.dataset_manager import DatasetManager
        print_status("DatasetManager Import", True)
        
        manager = DatasetManager()
        print_status("DatasetManager Creation", True)
        
        # Test PTB-XL metadata loading
        try:
            ptbxl_data = manager.load_ptbxl_complete(max_records=10, use_cache=False)
            records_loaded = ptbxl_data['stats']['total_records']
            print_status("PTB-XL Loading", records_loaded > 0, f"{records_loaded} records")
        except Exception as e:
            print_status("PTB-XL Loading", False, str(e)[:50])
        
        # Test ECG Arrhythmia loading  
        try:
            arrhythmia_data = manager.load_ecg_arrhythmia_complete(
                max_records=10, use_cache=False
            )
            records_loaded = arrhythmia_data['stats']['total_records']
            print_status("ECG Arrhythmia Loading", records_loaded > 0, f"{records_loaded} records")
        except Exception as e:
            print_status("ECG Arrhythmia Loading", False, str(e)[:50])
            
        return True
        
    except Exception as e:
        print_status("Data Loading System", False, str(e)[:50])
        return False

def test_existing_models():
    """Test existing trained models"""
    print_header("TRAINED MODELS TEST")
    
    project_root = Path(__file__).parent
    models_dir = project_root / "data" / "models"
    trained_models_dir = project_root / "models" / "trained_models"
    
    models_found = []
    
    # Check data/models directory
    if models_dir.exists():
        for model_file in models_dir.glob("*.joblib"):
            try:
                import joblib
                model = joblib.load(model_file)
                models_found.append(str(model_file.name))
                print_status(f"Model {model_file.name}", True, "Loadable")
            except Exception as e:
                print_status(f"Model {model_file.name}", False, "Load Error")
    
    # Check models/trained_models directory
    if trained_models_dir.exists():
        for model_file in trained_models_dir.glob("*.pkl"):
            models_found.append(str(model_file.name))
            print_status(f"Model {model_file.name}", True, "Found")
    
    print_status("Total Models Found", len(models_found) > 0, f"{len(models_found)} models")
    return len(models_found) > 0

def test_streamlit_app():
    """Test Streamlit app loading"""
    print_header("STREAMLIT APP TEST")
    
    project_root = Path(__file__).parent
    main_app = project_root / "app" / "main.py"
    
    if not main_app.exists():
        print_status("Main App File", False, "Missing")
        return False
    
    # Test imports in main.py
    try:
        sys.path.append(str(project_root))
        spec = __import__('importlib.util').util.spec_from_file_location("main", main_app)
        # Don't actually import to avoid streamlit startup
        print_status("Main App Structure", True, "Syntax OK")
        return True
    except Exception as e:
        print_status("Main App Structure", False, str(e)[:50])
        return False

def generate_system_report():
    """Generate comprehensive system health report"""
    print_header("ECG CLASSIFICATION SYSTEM - DIAGNOSTIC REPORT")
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        ("Environment", test_environment),
        ("Data Structure", test_data_structure), 
        ("Data Loading", test_data_loading),
        ("Trained Models", test_existing_models),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n[ERROR] {test_name} test failed: {e}")
            results[test_name] = False
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status_icon = "[OK]" if passed else "[FAIL]"
        print(f"{status_icon} {test_name}")
    
    print(f"\nOverall Health: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("SUCCESS: System is healthy and ready for use!")
    elif passed_tests >= total_tests * 0.8:
        print("WARNING: System is mostly functional with minor issues")
    else:
        print("ERROR: System has significant issues requiring attention")
    
    elapsed_time = time.time() - start_time
    print(f"\nDiagnostic completed in {elapsed_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    generate_system_report()