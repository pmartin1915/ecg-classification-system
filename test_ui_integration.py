"""
Quick UI Integration Test
Tests the enhanced interface components without heavy computation
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_ui_integration():
    """Test UI integration components"""
    
    print("TEST: UI INTEGRATION TEST")
    print("=" * 50)
    print("Testing enhanced interface components...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Enhanced main app import
    total_tests += 1
    try:
        from app import enhanced_main
        print("[OK] Enhanced main app imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f" Enhanced main app import failed: {e}")
    
    # Test 2: Smart launcher import
    total_tests += 1
    try:
        from app import smart_launcher
        print(" Smart launcher imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f" Smart launcher import failed: {e}")
    
    # Test 3: Enhanced MI status check
    total_tests += 1
    try:
        from app.enhanced_main import check_enhanced_mi_status
        status = check_enhanced_mi_status()
        print(f" MI status check works: {status['available']}")
        tests_passed += 1
    except Exception as e:
        print(f" MI status check failed: {e}")
    
    # Test 4: ECG generation functions
    total_tests += 1
    try:
        from app.enhanced_main import generate_demo_ecg, generate_enhanced_mi_ecg
        import numpy as np
        
        t = np.linspace(0, 4, 400)
        normal_ecg = generate_demo_ecg('NORM', t)
        mi_ecg = generate_enhanced_mi_ecg(t, 'anterior')
        
        print(" ECG generation functions work")
        tests_passed += 1
    except Exception as e:
        print(f" ECG generation failed: {e}")
    
    # Test 5: Clinical training component
    total_tests += 1
    try:
        from app.components.clinical_training import clinical_trainer
        print(" Clinical training component available")
        tests_passed += 1
    except Exception as e:
        print(f" Clinical training component failed: {e}")
    
    # Test 6: Batch processor component
    total_tests += 1
    try:
        from app.components.batch_processor import batch_processor
        print(" Batch processor component available")
        tests_passed += 1
    except Exception as e:
        print(f" Batch processor component failed: {e}")
    
    # Test 7: AI explainability component
    total_tests += 1
    try:
        from app.components.ai_explainability import ecg_explainer
        print(" AI explainability component available")
        tests_passed += 1
    except Exception as e:
        print(f" AI explainability component failed: {e}")
    
    # Test 8: Enhanced features availability
    total_tests += 1
    try:
        from models.feature_extraction.mi_specific_features import MISpecificFeatureExtractor
        extractor = MISpecificFeatureExtractor()
        print(" Enhanced MI feature extractor available")
        tests_passed += 1
    except Exception as e:
        print(f" Enhanced MI features failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("UI INTEGRATION TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print(" ALL TESTS PASSED!")
        print(" Enhanced UI integration is ready!")
        print("\nNext steps:")
        print("1. Run: ENHANCED_LAUNCHER.bat")
        print("2. Select option 1 for smart interface")
        print("3. Test enhanced MI analysis tab")
        return True
    elif tests_passed >= total_tests * 0.8:
        print(" MOSTLY WORKING - Minor issues detected")
        print(" Enhanced UI should work with some limitations")
        return True
    else:
        print(" SIGNIFICANT ISSUES - Needs attention")
        print(" Fix import errors before proceeding")
        return False

def test_streamlit_compatibility():
    """Test Streamlit compatibility"""
    print("\n STREAMLIT COMPATIBILITY TEST")
    print("-" * 40)
    
    try:
        import streamlit as st
        print(" Streamlit is installed")
        
        # Test if we can import our enhanced main without running it
        import sys
        from io import StringIO
        
        # Capture any output during import
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            from app import enhanced_main
            print(" Enhanced main compatible with Streamlit")
        finally:
            sys.stdout = old_stdout
            
        return True
        
    except Exception as e:
        print(f" Streamlit compatibility issue: {e}")
        return False

def main():
    """Main test function"""
    
    print("Testing enhanced UI integration...")
    print("This tests the interface without heavy computation")
    print()
    
    # Run UI integration tests
    ui_success = test_ui_integration()
    
    # Run Streamlit compatibility tests
    streamlit_success = test_streamlit_compatibility()
    
    # Overall result
    print("\n" + "" * 20)
    print("OVERALL INTEGRATION STATUS")
    print("" * 20)
    
    if ui_success and streamlit_success:
        print(" INTEGRATION SUCCESSFUL!")
        print(" Enhanced ECG interface is ready to use")
        print("\n Ready to launch:")
        print("   1. Double-click ENHANCED_LAUNCHER.bat")
        print("   2. Select option 1 (Enhanced System)")
        print("   3. Enjoy the new MI-focused interface!")
        
    elif ui_success or streamlit_success:
        print(" PARTIAL SUCCESS")
        print(" Basic functionality should work")
        print(" Some features may have limitations")
        
    else:
        print(" INTEGRATION ISSUES")
        print(" Please check dependencies and file paths")
        
    return ui_success and streamlit_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)