"""
Quick MI Enhancement Script
Run this after extracting the ECG Arrhythmia dataset to immediately improve MI detection
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


def main():
    """Main workflow for quick MI enhancement"""
    print("🫀 ECG MI DETECTION ENHANCEMENT")
    print("=" * 60)
    print("This script will:")
    print("1. Load combined PTB-XL + ECG Arrhythmia datasets")
    print("2. Train an MI-enhanced classification model")
    print("3. Validate and deploy the model for clinical use")
    print("4. Fix the MI sensitivity = 0.000 issue")
    print("=" * 60)
    
    # Configuration options
    print("\n📋 Configuration Options:")
    print("A. Quick Test (fast, small dataset)")
    print("B. Standard Training (balanced performance/time)")  
    print("C. Full Training (maximum performance, slow)")
    print("D. Custom Configuration")
    
    choice = input("\nSelect option (A/B/C/D): ").upper().strip()
    
    if choice == 'A':
        config = {
            'ptbxl_records': 500,
            'arrhythmia_records': 200,
            'target_mi_records': 100
        }
        print("🏃 Quick Test Configuration Selected")
    elif choice == 'B':
        config = {
            'ptbxl_records': 2000,
            'arrhythmia_records': 1000,
            'target_mi_records': 500
        }
        print("⚖️ Standard Training Configuration Selected")
    elif choice == 'C':
        config = {
            'ptbxl_records': 5000,
            'arrhythmia_records': 2000,
            'target_mi_records': 1000
        }
        print("🚀 Full Training Configuration Selected")
    elif choice == 'D':
        print("🔧 Custom Configuration:")
        config = {
            'ptbxl_records': int(input("PTB-XL records (100-10000): ")),
            'arrhythmia_records': int(input("ECG Arrhythmia records (50-5000): ")),
            'target_mi_records': int(input("Target MI records (50-2000): "))
        }
    else:
        print("❌ Invalid choice. Using Standard Configuration.")
        config = {
            'ptbxl_records': 2000,
            'arrhythmia_records': 1000,
            'target_mi_records': 500
        }
    
    print(f"\n📊 Training with:")
    print(f"   PTB-XL Records: {config['ptbxl_records']:,}")
    print(f"   ECG Arrhythmia Records: {config['arrhythmia_records']:,}")
    print(f"   Target MI Records: {config['target_mi_records']:,}")
    
    confirm = input("\nProceed with training? (y/N): ").lower().strip()
    if confirm != 'y':
        print("❌ Training cancelled.")
        return
    
    # Start training
    print("\n🚀 Starting MI Enhancement Training...")
    print("⏳ This may take 5-30 minutes depending on configuration...")
    
    try:
        results = quick_train_and_deploy_mi_model(
            ptbxl_records=config['ptbxl_records'],
            arrhythmia_records=config['arrhythmia_records'],
            target_mi_records=config['target_mi_records']
        )
        
        if results.get('training_successful'):
            print("\n✅ MI ENHANCEMENT SUCCESSFUL!")
            print("=" * 60)
            
            # Show improvement metrics
            dataset_stats = results['dataset_stats']
            mi_records = dataset_stats.get('mi_records', 0)
            total_records = dataset_stats.get('total_records', 0)
            
            print(f"📊 Dataset Statistics:")
            print(f"   Total Records: {total_records:,}")
            print(f"   MI Records: {mi_records:,}")
            print(f"   MI Percentage: {(mi_records/total_records*100):.1f}%")
            
            # Show validation results
            validation = results['validation_results']
            print(f"\n🎯 Model Validation:")
            print(f"   Overall Accuracy: {validation['overall_accuracy']:.1%}")
            print(f"   MI Sensitivity: {validation['mi_sensitivity']:.1%}")
            print(f"   Validation Status: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}")
            
            # Show deployment status
            deployment = results.get('deployment_results', {})
            if deployment.get('deployment_successful'):
                print(f"   Deployment: ✅ DEPLOYED TO PRODUCTION")
                print(f"   Model Path: {results['model_path']}")
            else:
                print(f"   Deployment: ⚠️ Manual deployment required")
            
            print(f"\n🎉 BEFORE vs AFTER:")
            print(f"   Previous MI Sensitivity: 0.000 (0.0%)")
            print(f"   New MI Sensitivity: {validation['mi_sensitivity']:.3f} ({validation['mi_sensitivity']:.1%})")
            
            improvement = validation['mi_sensitivity'] * 100
            print(f"   🎯 IMPROVEMENT: +{improvement:.1f}% MI detection capability!")
            
            print(f"\n🏥 Clinical Impact:")
            if validation['mi_sensitivity'] > 0.8:
                print("   🟢 EXCELLENT: Model can reliably detect heart attacks")
            elif validation['mi_sensitivity'] > 0.6:
                print("   🟡 GOOD: Model has strong MI detection capability")
            elif validation['mi_sensitivity'] > 0.3:
                print("   🟠 MODERATE: Significant improvement, consider more training")
            else:
                print("   🔴 POOR: Needs more MI training data")
            
            print(f"\n🚀 Next Steps:")
            print("1. Test the new model in your Streamlit app")
            print("2. Run Phase 4 testing to confirm improvement")
            print("3. Monitor clinical performance in production")
            print("4. Consider additional training with more data if needed")
            
            print(f"\n💻 To use in your app:")
            print("```python")
            print("from app.components.classification import ECGClassifier")
            print("classifier = ECGClassifier('combined_mi_enhanced')")
            print("classifier.load_model()")
            print("result = classifier.predict(your_ecg_data)")
            print("```")
            
        else:
            print("\n❌ MI ENHANCEMENT FAILED!")
            print("=" * 60)
            error = results.get('error', 'Unknown error')
            print(f"Error: {error}")
            
            print(f"\n🔧 Troubleshooting:")
            print("1. Ensure ECG Arrhythmia dataset is properly extracted")
            print("2. Check that WFDBRecords folder exists in data/raw/ecg-arrhythmia-dataset/")
            print("3. Run: python test_combined_dataset.py")
            print("4. Verify sufficient disk space and memory")
            print("5. Try with smaller dataset configuration first")
            
            if 'traceback' in results:
                print(f"\nDetailed error:")
                print(results['traceback'])
    
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def quick_test_only():
    """Quick test without full training"""
    print("🧪 QUICK DATASET TEST")
    print("=" * 40)
    
    try:
        # Test combined dataset loading
        print("Testing combined dataset loading...")
        X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
            ptbxl_max_records=50,
            arrhythmia_max_records=20,
            target_mi_records=10,
            sampling_rate=100
        )
        
        if len(X) > 0:
            print("✅ Combined dataset loading: SUCCESS")
            print(f"   Total records: {len(X)}")
            print(f"   MI records: {stats.get('mi_records', 0)}")
            print(f"   Dataset sources: {stats.get('dataset_sources', 'Unknown')}")
            
            from collections import Counter
            label_dist = Counter(labels)
            print(f"   Label distribution: {dict(label_dist)}")
            
            if stats.get('mi_records', 0) > 0:
                print("🎯 MI records found! Ready for full training.")
                return True
            else:
                print("⚠️ No MI records found in test sample.")
                return False
        else:
            print("❌ Combined dataset loading: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("Welcome to the ECG MI Detection Enhancement Tool!")
    print("=" * 60)
    
    # Check if user wants quick test first
    test_first = input("Run quick dataset test first? (Y/n): ").lower().strip()
    
    if test_first != 'n':
        print("\n" + "="*40)
        if quick_test_only():
            print("\n" + "="*40)
            proceed = input("Test successful! Proceed with full training? (Y/n): ").lower().strip()
            if proceed != 'n':
                main()
        else:
            print("\n❌ Quick test failed. Please check your dataset setup.")
            print("Run: python test_combined_dataset.py for detailed diagnostics")
    else:
        main()