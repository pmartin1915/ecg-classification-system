"""
Non-interactive MI Enhancement Script
Automatically runs MI enhancement with Quick Test configuration
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.utils.model_integration import quick_train_and_deploy_mi_model
from app.utils.dataset_manager import run_combined_dataset_loading


def run_quick_mi_enhancement():
    """Run MI enhancement with Quick Test configuration automatically"""
    print("ECG MI DETECTION ENHANCEMENT")
    print("=" * 60)
    print("Running Quick Test Configuration:")
    print("- PTB-XL records: 500")
    print("- ECG Arrhythmia records: 200") 
    print("- Target MI records: 100")
    print("=" * 60)
    
    # Quick Test Configuration
    config = {
        'ptbxl_records': 500,
        'arrhythmia_records': 200,
        'target_mi_records': 100
    }
    
    try:
        print("\nStep 1: Loading combined dataset...")
        X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
            ptbxl_max_records=config['ptbxl_records'],
            arrhythmia_max_records=config['arrhythmia_records'],
            target_mi_records=config['target_mi_records'],
            sampling_rate=100
        )
        
        print(f"OK: Loaded {len(X)} total records")
        print(f"   PTB-XL records: {stats.get('ptbxl_records', 0)}")
        print(f"   Arrhythmia records: {stats.get('arrhythmia_records', 0)}")
        print(f"   MI records: {stats.get('mi_records', 0)}")
        print(f"   Target conditions: {list(target_conditions.keys())}")
        
        # Check if we have sufficient MI records
        mi_count = stats.get('mi_records', 0)
        if mi_count < 10:
            print(f"WARNING: Only {mi_count} MI records found. Minimum 10 recommended.")
            print("Training will proceed but MI detection may be limited.")
        
        print("\nStep 2: Training MI-enhanced model...")
        result = quick_train_and_deploy_mi_model(
            X, labels, ids, metadata, target_conditions,
            model_name='quick_mi_enhanced',
            training_mode='quick'
        )
        
        if result['success']:
            print("\nOK: MI Enhancement Complete!")
            print("=" * 60)
            print("ENHANCEMENT SUMMARY:")
            print(f"Model saved as: {result['model_name']}")
            print(f"MI Sensitivity: {result['performance']['mi_sensitivity']:.3f}")
            print(f"Overall Accuracy: {result['performance']['accuracy']:.3f}")
            print(f"Model location: {result['model_path']}")
            print("=" * 60)
            
            if result['performance']['mi_sensitivity'] > 0.5:
                print("SUCCESS: MI detection significantly improved!")
                print("Your clinical app will now use this enhanced model automatically.")
            else:
                print("WARNING: MI sensitivity still low. Consider:")
                print("1. Running with Standard or Full configuration")
                print("2. Checking dataset quality")
                print("3. Adjusting training parameters")
        
        return result
        
    except Exception as e:
        print(f"ERROR: MI enhancement failed: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    result = run_quick_mi_enhancement()
    
    if result['success']:
        print("\nNext Steps:")
        print("1. Test your enhanced model: streamlit run app/main.py")
        print("2. Upload ECG files to verify improved MI detection")
        print("3. For better performance, run with Standard configuration")
    else:
        print("\nTroubleshooting:")
        print("1. Verify dataset extraction: python test_combined_dataset.py")
        print("2. Check error logs above")
        print("3. Ensure all dependencies are installed")