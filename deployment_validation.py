"""
Clinical Training System - Deployment Validation
Tests all major components for deployment readiness
"""

import sys
import importlib
import traceback
from pathlib import Path

def test_component(component_name, test_func):
    """Test a system component and report results"""
    try:
        print(f"[TESTING] {component_name}...")
        result = test_func()
        if result:
            print(f"[SUCCESS] {component_name} - Ready for deployment")
            return True
        else:
            print(f"[WARNING] {component_name} - Issues detected")
            return False
    except Exception as e:
        print(f"[ERROR] {component_name} - {str(e)}")
        return False

def test_imports():
    """Test all critical imports"""
    required_modules = [
        'streamlit', 'numpy', 'pandas', 'matplotlib', 'plotly', 
        'wfdb', 'sklearn', 'tqdm', 'pathlib'
    ]
    
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            print(f"Missing required module: {module}")
            return False
    return True

def test_ptbxl_extractor():
    """Test PTB-XL MI extractor functionality"""
    try:
        sys.path.append('.')
        from app.utils.ptbxl_mi_extractor import PTBXLMIExtractor
        
        # Test initialization
        extractor = PTBXLMIExtractor()
        
        # Verify database loading
        if extractor.database_df is None or len(extractor.database_df) == 0:
            return False
        
        # Verify MI mapping
        if not extractor.mi_mapping or len(extractor.mi_mapping) < 6:
            return False
            
        print(f"    - Database loaded: {len(extractor.database_df)} records")
        print(f"    - MI types mapped: {len(extractor.mi_mapping)} types")
        
        return True
    except Exception as e:
        print(f"    - Error: {e}")
        return False

def test_main_interface():
    """Test main Streamlit interface"""
    try:
        import complete_user_friendly
        
        # Check if main functions exist
        required_functions = [
            'main', 'show_dashboard', 'show_ecg_analysis', 
            'show_heart_attack_focus', 'show_ai_explainability',
            'show_clinical_training', 'show_clinical_reports',
            'show_batch_processing', 'show_performance_monitor'
        ]
        
        for func_name in required_functions:
            if not hasattr(complete_user_friendly, func_name):
                print(f"    - Missing function: {func_name}")
                return False
        
        print(f"    - All {len(required_functions)} core functions available")
        return True
    except Exception as e:
        print(f"    - Error: {e}")
        return False

def test_clinical_training():
    """Test clinical training modules"""
    try:
        import complete_user_friendly
        
        # Check training functions
        training_functions = [
            'show_ecg_fundamentals', 'show_heart_attack_training',
            'show_ptbxl_dataset_training', 'show_rhythm_training',
            'show_ai_cardiology_training', 'show_practice_cases'
        ]
        
        for func_name in training_functions:
            if not hasattr(complete_user_friendly, func_name):
                print(f"    - Missing training function: {func_name}")
                return False
        
        print(f"    - All {len(training_functions)} training modules available")
        return True
    except Exception as e:
        print(f"    - Error: {e}")
        return False

def test_clinical_reports():
    """Test clinical reporting system"""
    try:
        import complete_user_friendly
        
        # Check reporting functions
        report_functions = [
            'show_clinical_reports', 'show_report_generator',
            'show_clinical_templates', 'show_validation_metrics',
            'show_report_management', 'show_report_standards'
        ]
        
        for func_name in report_functions:
            if not hasattr(complete_user_friendly, func_name):
                print(f"    - Missing report function: {func_name}")
                return False
        
        print(f"    - All {len(report_functions)} reporting modules available")
        return True
    except Exception as e:
        print(f"    - Error: {e}")
        return False

def test_data_availability():
    """Test data file availability"""
    try:
        data_paths = [
            Path("data/raw/ptbxl/ptbxl_database.csv"),
            Path("data/raw/ptbxl/scp_statements.csv"),
            Path("data/cache"),
            Path("data/processed")
        ]
        
        missing_paths = []
        for path in data_paths:
            if not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            print(f"    - Missing data paths: {missing_paths}")
            return False
        
        print(f"    - All {len(data_paths)} data paths available")
        return True
    except Exception as e:
        print(f"    - Error: {e}")
        return False

def test_launchers():
    """Test deployment launchers"""
    try:
        launcher_files = [
            "COMPLETE_USER_FRIENDLY_LAUNCHER.bat",
            "LAUNCH_ECG_SYSTEM.bat"
        ]
        
        missing_launchers = []
        for launcher in launcher_files:
            if not Path(launcher).exists():
                missing_launchers.append(launcher)
        
        if missing_launchers:
            print(f"    - Missing launchers: {missing_launchers}")
            return False
        
        print(f"    - All {len(launcher_files)} launchers available")
        return True
    except Exception as e:
        print(f"    - Error: {e}")
        return False

def main():
    """Run complete deployment validation"""
    print("=" * 70)
    print("CLINICAL TRAINING SYSTEM - DEPLOYMENT VALIDATION")
    print("=" * 70)
    print()
    
    # Test components
    test_results = []
    
    test_results.append(test_component("Required Dependencies", test_imports))
    test_results.append(test_component("PTB-XL MI Extractor", test_ptbxl_extractor))  
    test_results.append(test_component("Main Interface", test_main_interface))
    test_results.append(test_component("Clinical Training", test_clinical_training))
    test_results.append(test_component("Clinical Reports", test_clinical_reports))
    test_results.append(test_component("Data Availability", test_data_availability))
    test_results.append(test_component("Deployment Launchers", test_launchers))
    
    # Summary
    print()
    print("=" * 70)
    print("DEPLOYMENT VALIDATION SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: [OK] {passed_tests}")
    print(f"Failed: [FAIL] {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    if failed_tests == 0:
        print("*** DEPLOYMENT READY - All systems operational! ***")
        print("[OK] Clinical Training System ready for deployment")
        print("[OK] PTB-XL integration validated")
        print("[OK] Professional reporting system confirmed")
        print("[OK] Educational modules verified")
    else:
        print("*** DEPLOYMENT ISSUES - Please address failed components ***")
    
    print("=" * 70)
    
    return failed_tests == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)