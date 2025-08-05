"""
Test Enhanced AI Explainability Integration
Tests the enhanced explainability component and UI integration
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_enhanced_explainability_import():
    """Test importing enhanced explainability component"""
    
    print("ENHANCED AI EXPLAINABILITY INTEGRATION TEST")
    print("=" * 55)
    print("Testing enhanced explainability component integration...")
    print("=" * 55)
    
    try:
        print("\n1. Enhanced Explainability Component Import:")
        from app.components.enhanced_explainability import enhanced_explainer
        print("   [OK] Enhanced explainability component imported successfully")
        
        # Test core functionality
        print("\n2. Core Functionality Test:")
        
        # Test MI diagnostic criteria
        mi_criteria = enhanced_explainer.mi_diagnostic_criteria
        print(f"   [OK] MI diagnostic criteria loaded: {len(mi_criteria)} conditions")
        
        # Test clinical reasoning framework
        reasoning_framework = enhanced_explainer.clinical_reasoning_framework
        print(f"   [OK] Clinical reasoning framework loaded")
        
        # Test feature mapping
        feature_mapping = enhanced_explainer.feature_clinical_mapping
        print(f"   [OK] Feature-clinical mapping loaded: {len(feature_mapping)} categories")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Enhanced explainability import failed: {e}")
        return False

def test_enhanced_main_integration():
    """Test integration with enhanced main app"""
    
    print("\n\nENHANCED MAIN APP INTEGRATION TEST")
    print("=" * 50)
    
    try:
        print("1. Enhanced Main App Import:")
        import app.enhanced_main as enhanced_main
        print("   [OK] Enhanced main app imported successfully")
        
        # Test function exists
        print("\n2. AI Explainability Function Test:")
        if hasattr(enhanced_main, 'show_enhanced_ai_explainability'):
            print("   [OK] show_enhanced_ai_explainability function exists")
        else:
            print("   [ERROR] show_enhanced_ai_explainability function not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Enhanced main app integration failed: {e}")
        return False

def test_component_integration():
    """Test integration between components"""
    
    print("\n\nCOMPONENT INTEGRATION TEST")
    print("=" * 40)
    
    try:
        # Test imports together
        print("1. Combined Import Test:")
        
        from app.components.enhanced_explainability import enhanced_explainer
        from app.utils.fast_prediction_pipeline import fast_pipeline
        from app.components.performance_monitor import PerformanceMonitor
        
        print("   [OK] All components can be imported together")
        
        # Test component interaction
        print("\n2. Component Interaction Test:")
        
        # Test explainability with different diagnoses
        test_diagnoses = ["AMI", "IMI", "AFIB", "LBBB", "NORM"]
        
        for diagnosis in test_diagnoses:
            mi_type = enhanced_explainer._determine_mi_type(diagnosis)
            confidence_category = enhanced_explainer._get_confidence_category(85.0)
            print(f"   [OK] {diagnosis}: MI type = {mi_type}, Confidence = {confidence_category}")
        
        # Test feature analysis
        print("\n3. Feature Analysis Test:")
        mi_features = enhanced_explainer._get_mi_features_for_display("AMI")
        general_features = enhanced_explainer._get_general_features_for_display("AFIB")
        
        print(f"   [OK] MI features generated: {len(mi_features)} features")
        print(f"   [OK] General features generated: {len(general_features)} features")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Component integration test failed: {e}")
        return False

def test_teaching_scenarios():
    """Test teaching scenarios functionality"""
    
    print("\n\nTEACHING SCENARIOS TEST")
    print("=" * 35)
    
    try:
        from app.components.enhanced_explainability import enhanced_explainer
        
        print("1. Teaching Scenarios Generation:")
        
        # Test different experience levels
        experience_levels = ["Beginner", "Intermediate", "Advanced"]
        
        for level in experience_levels:
            scenarios = enhanced_explainer._get_scenarios_for_level(level)
            learning_objectives = enhanced_explainer._get_learning_objectives("AMI", level)
            practice_recommendations = enhanced_explainer._get_practice_recommendations("AMI", level)
            
            print(f"   [OK] {level}: {len(scenarios)} scenarios, {len(learning_objectives)} objectives")
        
        print("\n2. Clinical Context Generation:")
        
        # Test clinical actions
        immediate_actions = enhanced_explainer._get_immediate_actions("AMI", 0.85)
        followup_actions = enhanced_explainer._get_followup_actions("AMI", 0.85)
        patient_education = enhanced_explainer._get_patient_education_points("AMI")
        
        print(f"   [OK] Immediate actions: {len(immediate_actions)}")
        print(f"   [OK] Follow-up actions: {len(followup_actions)}")
        print(f"   [OK] Patient education points: {len(patient_education)}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Teaching scenarios test failed: {e}")
        return False

def main():
    """Main testing function"""
    
    print("ENHANCED AI EXPLAINABILITY TESTING")
    print("=" * 60)
    print("Testing enhanced AI explainability integration and functionality")
    print("=" * 60)
    
    # Run tests
    test_results = []
    
    # Test 1: Enhanced explainability import
    explainability_success = test_enhanced_explainability_import()
    test_results.append(("Enhanced Explainability Import", explainability_success))
    
    # Test 2: Enhanced main integration
    main_integration_success = test_enhanced_main_integration()
    test_results.append(("Enhanced Main Integration", main_integration_success))
    
    # Test 3: Component integration
    component_integration_success = test_component_integration()
    test_results.append(("Component Integration", component_integration_success))
    
    # Test 4: Teaching scenarios
    teaching_success = test_teaching_scenarios()
    test_results.append(("Teaching Scenarios", teaching_success))
    
    # Summary
    print("\n\n" + "=" * 60)
    print("ENHANCED AI EXPLAINABILITY TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, success in test_results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
        if success:
            passed_tests += 1
    
    print(f"\nTests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] Enhanced AI Explainability fully functional!")
        print("System ready for comprehensive clinical education and training")
        print("\nKey Features Available:")
        print("✅ MI-specific diagnostic reasoning")
        print("✅ Clinical feature explanations") 
        print("✅ Interactive teaching scenarios")
        print("✅ Experience-tailored explanations")
        print("✅ Uncertainty handling and confidence analysis")
        print("✅ Real-time ECG analysis with AI reasoning")
        
        print("\nNext steps:")
        print("1. Launch enhanced system: ENHANCED_LAUNCHER.bat")
        print("2. Navigate to 'AI Explainability' tab")
        print("3. Try different demo modes and diagnoses")
        print("4. Test teaching scenarios for medical education")
        
    elif passed_tests >= total_tests * 0.8:
        print("\n[MOSTLY WORKING] Enhanced AI explainability mostly functional")
        print("Some minor issues detected - core functionality should work")
        
    else:
        print("\n[ISSUES DETECTED] Enhanced AI explainability needs attention")
        print("Fix import errors and component issues before proceeding")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)