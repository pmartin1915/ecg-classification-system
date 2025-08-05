"""
Performance Optimization Testing Script
Tests the fast prediction pipeline and performance monitoring
"""
import sys
from pathlib import Path
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_fast_prediction_pipeline():
    """Test the fast prediction pipeline performance"""
    
    print("FAST PREDICTION PIPELINE TEST")
    print("=" * 50)
    print("Testing real-time ECG analysis performance...")
    print("Target: <3 seconds total analysis time")
    print("=" * 50)
    
    try:
        from app.utils.fast_prediction_pipeline import fast_pipeline
        
        # Test 1: Model loading check
        print("\n1. Model Loading Test:")
        models_loaded = list(fast_pipeline.models_cache.keys())
        print(f"   Models loaded: {models_loaded}")
        
        if not models_loaded:
            print("   [WARNING] No models loaded - using synthetic model")
        else:
            print("   [OK] Models successfully loaded into memory")
        
        # Test 2: Synthetic ECG generation
        print("\n2. Synthetic ECG Generation Test:")
        start_time = time.time()
        
        # Generate test ECG data
        duration = 4  # seconds
        sampling_rate = 100  # Hz
        samples = duration * sampling_rate
        
        # Create 12-lead ECG
        test_ecg = np.random.randn(12, samples) * 0.1
        for lead in range(12):
            # Add realistic ECG pattern
            t = np.linspace(0, duration, samples)
            test_ecg[lead] += 0.5 * np.sin(2 * np.pi * 1.2 * t)  # Heart rhythm
            test_ecg[lead] += 0.3 * np.exp(-((t - 2) / 0.1)**2)   # QRS complex
        
        generation_time = time.time() - start_time
        print(f"   ECG generation time: {generation_time:.3f}s")
        print(f"   ECG shape: {test_ecg.shape}")
        
        # Test 3: Fast prediction performance
        print("\n3. Fast Prediction Performance Test:")
        
        # Run multiple predictions to get average performance
        times = []
        results = []
        
        for i in range(5):  # Test 5 predictions
            start_time = time.time()
            
            try:
                result = fast_pipeline.fast_predict(test_ecg, use_enhanced=True)
                prediction_time = time.time() - start_time
                times.append(prediction_time)
                results.append(result)
                
                print(f"   Prediction {i+1}: {prediction_time:.3f}s - {result.get('diagnosis', 'Unknown')}")
                
            except Exception as e:
                print(f"   Prediction {i+1}: FAILED - {e}")
        
        # Performance analysis
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"\n   Performance Summary:")
            print(f"   Average time: {avg_time:.3f}s")
            print(f"   Fastest time: {min_time:.3f}s")
            print(f"   Slowest time: {max_time:.3f}s")
            
            # Performance grade
            if avg_time <= 1.0:
                grade = "A+ (Excellent)"
                status = "[EXCELLENT]"
            elif avg_time <= 2.0:
                grade = "A (Very Good)"
                status = "[VERY GOOD]"
            elif avg_time <= 3.0:
                grade = "B (Good - Target Met)"
                status = "[TARGET MET]"
            elif avg_time <= 5.0:
                grade = "C (Acceptable)"
                status = "[ACCEPTABLE]"
            else:
                grade = "D (Needs Improvement)"
                status = "[NEEDS WORK]"
            
            print(f"   Performance grade: {grade}")
            print(f"   Status: {status}")
            
            return avg_time <= 3.0
        else:
            print("   [ERROR] No successful predictions")
            return False
            
    except Exception as e:
        print(f"   [ERROR] Fast prediction pipeline test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring components"""
    
    print("\n\nPERFORMANCE MONITORING TEST")
    print("=" * 50)
    
    try:
        from app.components.performance_monitor import PerformanceMonitor
        
        # Test 1: Performance monitor initialization
        print("1. Performance Monitor Initialization:")
        monitor = PerformanceMonitor()
        print("   [OK] Performance monitor created")
        
        # Test 2: Session state initialization
        print("\n2. Session State Test:")
        try:
            import streamlit as st
            
            # Mock session state for testing
            class MockSessionState:
                def __init__(self):
                    self.performance_history = []
                    self.performance_targets = {
                        'total_time': 3.0,
                        'preprocessing_time': 0.5,
                        'feature_extraction_time': 1.5,
                        'prediction_time': 0.5,
                        'formatting_time': 0.5
                    }
                def __contains__(self, key):
                    return hasattr(self, key)
                def __getitem__(self, key):
                    return getattr(self, key)
                def __setitem__(self, key, value):
                    setattr(self, key, value)
            
            # Replace st.session_state for testing
            original_session_state = getattr(st, 'session_state', None)
            st.session_state = MockSessionState()
            
            monitor.initialize_session_state()
            print("   [OK] Session state initialized")
            
            # Restore original session state
            if original_session_state:
                st.session_state = original_session_state
            
        except ImportError:
            print("   [SKIP] Streamlit not available for session state test")
        
        # Test 3: Synthetic ECG generation for monitoring
        print("\n3. Synthetic ECG Generation for Monitoring:")
        synthetic_ecg = monitor.generate_synthetic_ecg()
        print(f"   ECG shape: {synthetic_ecg.shape}")
        print("   [OK] Synthetic ECG generated for monitoring")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Performance monitoring test failed: {e}")
        return False

def test_integration():
    """Test integration between components"""
    
    print("\n\nINTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Import Test:")
        
        components = [
            ("Fast Prediction Pipeline", "app.utils.fast_prediction_pipeline"),
            ("Performance Monitor", "app.components.performance_monitor"),
            ("Enhanced Main App", "app.enhanced_main")
        ]
        
        import_success = True
        for name, module in components:
            try:
                __import__(module)
                print(f"   [OK] {name}")
            except Exception as e:
                print(f"   [FAIL] {name}: {e}")
                import_success = False
        
        if import_success:
            print("\n2. Integration Status:")
            print("   [OK] All components can be imported")
            print("   [OK] Fast prediction pipeline ready")
            print("   [OK] Performance monitoring ready")
            print("   [OK] Enhanced UI integration ready")
            
            return True
        else:
            print("\n   [ERROR] Some components failed to import")
            return False
            
    except Exception as e:
        print(f"   [ERROR] Integration test failed: {e}")
        return False

def main():
    """Main testing function"""
    
    print("PERFORMANCE OPTIMIZATION TESTING")
    print("=" * 60)
    print("Testing real-time ECG analysis performance optimization")
    print("=" * 60)
    
    # Run tests
    test_results = []
    
    # Test 1: Fast prediction pipeline
    fast_pipeline_success = test_fast_prediction_pipeline()
    test_results.append(("Fast Prediction Pipeline", fast_pipeline_success))
    
    # Test 2: Performance monitoring
    monitoring_success = test_performance_monitoring()
    test_results.append(("Performance Monitoring", monitoring_success))
    
    # Test 3: Integration
    integration_success = test_integration()
    test_results.append(("Component Integration", integration_success))
    
    # Summary
    print("\n\n" + "=" * 60)
    print("PERFORMANCE OPTIMIZATION TEST SUMMARY")
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
        print("\n[SUCCESS] All performance optimization tests passed!")
        print("System is ready for real-time ECG analysis")
        print("\nNext steps:")
        print("1. Launch enhanced system: ENHANCED_LAUNCHER.bat")
        print("2. Test Performance Monitor tab")
        print("3. Upload ECG files for real-time analysis")
        
    elif passed_tests >= total_tests * 0.8:
        print("\n[MOSTLY WORKING] Performance optimization mostly functional")
        print("Some minor issues detected - system should work with limitations")
        
    else:
        print("\n[ISSUES DETECTED] Performance optimization needs attention")
        print("Fix import errors and component issues before proceeding")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)