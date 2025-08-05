"""
Quick Test Script for MI Enhancement System
Tests the enhanced MI detection with a small dataset
"""
import sys
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any, Optional
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def quick_mi_enhancement_test():
    """Run a quick test of the MI enhancement system"""
    
    print("QUICK MI ENHANCEMENT TEST")
    print("=" * 50)
    print("Testing enhanced MI detection with small dataset")
    print("=" * 50)
    
    try:
        from enhanced_mi_detection_system import EnhancedMIDetectionSystem
        
        # Initialize system
        system = EnhancedMIDetectionSystem()
        
        # Run with small dataset for testing
        print("Running enhancement with 200 records (test size)...")
        results = system.run_complete_enhancement(max_records=200)
        
        if results['success']:
            print("\n[SUCCESS] TEST PASSED!")
            
            if 'analysis' in results:
                analysis = results['analysis']
                print(f"MI Detection Improvement: {analysis['baseline_sensitivity']:.1%} -> {analysis['enhanced_sensitivity']:.1%}")
                
                if analysis['enhanced_sensitivity'] > analysis['baseline_sensitivity']:
                    print("[OK] Improvement detected!")
                else:
                    print("[WARNING] No improvement - may need larger dataset")
            
            return True
        else:
            print(f"\n[FAIL] TEST FAILED: {results.get('error', 'Unknown error')}")
            return False
            
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"[ERROR] Test error: {e}")
        return False

def main():
    """Main test function"""
    success = quick_mi_enhancement_test()
    
    if success:
        print("\n[SUCCESS] Ready to run full enhancement!")
        print("Run: python enhanced_mi_detection_system.py")
    else:
        print("\n[FIX NEEDED] Fix issues before running full enhancement")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())