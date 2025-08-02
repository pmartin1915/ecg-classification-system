"""
Test script for combined PTB-XL + ECG Arrhythmia dataset integration
This script tests the new functionality for improved MI detection
"""
import sys
import warnings
from pathlib import Path
import numpy as np
from collections import Counter

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.utils.dataset_manager import DatasetManager, run_combined_dataset_loading


def test_ecg_arrhythmia_dataset_scan():
    """Test scanning the ECG Arrhythmia dataset structure"""
    print("TESTING ECG ARRHYTHMIA DATASET STRUCTURE SCAN")
    print("=" * 60)
    
    from app.utils.data_loader import ECGArrhythmiaDataLoader
    
    loader = ECGArrhythmiaDataLoader()
    structure = loader.scan_dataset_structure()
    
    print(f"Total estimated records: {structure['total_records']:,}")
    print(f"Top-level folders found: {len(structure['folders'])}")
    
    if structure['total_records'] > 0:
        print("✅ ECG Arrhythmia dataset structure detected successfully")
        
        # Test loading a few sample records
        print("\nTesting sample record loading...")
        sample_data = loader.load_records_batch(max_records=10, target_mi_records=5)
        
        if len(sample_data['X']) > 0:
            print(f"✅ Loaded {len(sample_data['X'])} sample records")
            print(f"   MI records found: {sample_data['stats']['mi_records']}")
            print(f"   Label distribution: {sample_data['stats']['label_distribution']}")
            return True
        else:
            print("❌ No sample records loaded")
            return False
    else:
        print("❌ ECG Arrhythmia dataset not found or inaccessible")
        print("\nTo use this functionality, you need to:")
        print("1. Download the ECG Arrhythmia dataset from PhysioNet")
        print("2. Place it in: data/raw/ecg-arrhythmia-dataset/")
        print("3. Ensure the WFDBRecords folder structure is preserved")
        return False


def test_combined_dataset_loading():
    """Test the combined dataset loading functionality"""
    print("\nTESTING COMBINED DATASET LOADING")
    print("=" * 60)
    
    try:
        # Test with small subset first
        print("Loading combined dataset (small test)...")
        X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
            ptbxl_max_records=50,    # Small PTB-XL subset
            arrhythmia_max_records=20,  # Small ECG Arrhythmia subset
            target_mi_records=10,     # Target at least 10 MI records
            sampling_rate=100,
            use_cache=True
        )
        
        if len(X) > 0:
            print(f"✅ Combined dataset loaded successfully!")
            print(f"   Total records: {len(X):,}")
            print(f"   Signal shape: {X.shape}")
            print(f"   Memory usage: {stats['memory_gb']:.3f} GB")
            
            # Analyze label distribution
            label_counts = Counter(labels)
            print(f"   Label distribution: {dict(label_counts)}")
            
            mi_count = label_counts.get('MI', 0)
            total_count = len(labels)
            mi_percentage = (mi_count / total_count * 100) if total_count > 0 else 0
            
            print(f"   MI Detection Improvement:")
            print(f"     MI records: {mi_count}")
            print(f"     MI percentage: {mi_percentage:.1f}%")
            
            if mi_count > 0:
                print("   🎯 SUCCESS: MI records found in combined dataset!")
                print("   This should significantly improve MI detection sensitivity.")
            else:
                print("   ⚠️  No MI records found in this small test subset")
                print("   Try increasing target_mi_records or arrhythmia_max_records")
            
            return True, stats
            
        else:
            print("❌ Combined dataset loading failed - no records loaded")
            return False, None
            
    except Exception as e:
        print(f"❌ Error during combined dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_phase4_with_combined_dataset():
    """Test Phase 4 training with the combined dataset"""
    print("\nTESTING PHASE 4 WITH COMBINED DATASET")
    print("=" * 60)
    
    # Check if we can load the combined dataset first
    success, stats = test_combined_dataset_loading()
    
    if not success:
        print("❌ Cannot test Phase 4 - combined dataset loading failed")
        return False
    
    try:
        # Import Phase 4 components
        from models.training.phase4_model_training import Phase4Pipeline
        
        print("\nRunning Phase 4 training with combined dataset...")
        
        # Load combined dataset for training
        X, labels, ids, metadata, target_conditions, load_stats = run_combined_dataset_loading(
            ptbxl_max_records=200,   # Moderate subset for training test
            arrhythmia_max_records=100,
            target_mi_records=50,    # Ensure good MI representation
            sampling_rate=100,
            use_cache=True
        )
        
        if len(X) == 0:
            print("❌ No data loaded for training")
            return False
        
        # Run feature extraction (minimal)
        from models.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
        from models.preprocessing.preprocessing_pipeline import PreprocessingPipeline
        
        print("Running preprocessing...")
        preprocessing = PreprocessingPipeline()
        prep_results = preprocessing.run(X, labels, max_records=len(X))
        
        print("Running feature extraction...")
        feature_pipeline = FeatureExtractionPipeline()
        feature_results = feature_pipeline.run(
            prep_results['X_preprocessed'],
            prep_results['y_encoded'],
            prep_results['label_encoder'],
            max_features=50  # Reduced for testing
        )
        
        # Configure training pipeline for testing
        training_config = {
            'test_size': 0.3,
            'val_size': 0.2,
            'use_smote': True,  # Important for MI detection
            'use_hyperparameter_tuning': False,  # Skip for faster testing
            'model_keys': ['random_forest', 'logistic_regression'],
            'create_visualizations': False,
            'save_models': False
        }
        
        print("Running model training...")
        pipeline = Phase4Pipeline(training_config)
        training_results = pipeline.run(
            X=feature_results['X_features'],
            y=feature_results['y_encoded'],
            feature_names=feature_results['feature_names']
        )
        
        # Check MI detection performance
        test_metrics = training_results['test_results']['metrics']
        print(f"\n🎯 TRAINING RESULTS WITH COMBINED DATASET:")
        print(f"   Best model: {training_results['best_model_name']}")
        print(f"   Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"   F1-Score: {test_metrics['f1_weighted']:.3f}")
        
        # Look for MI-specific metrics
        if 'classification_report' in test_metrics:
            print(f"   Classification Report Available: Yes")
        
        # Check if MI sensitivity improved
        mi_sensitivity = test_metrics.get('mi_sensitivity', 'N/A')
        print(f"   MI Sensitivity: {mi_sensitivity}")
        
        if isinstance(mi_sensitivity, (int, float)) and mi_sensitivity > 0:
            print("   🎉 SUCCESS: MI detection working with combined dataset!")
        else:
            print("   ⚠️  MI sensitivity still low - may need more MI samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Phase 4 testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main testing function"""
    print("ECG ARRHYTHMIA DATASET INTEGRATION TEST")
    print("=" * 80)
    
    # Test 1: Dataset structure scan
    structure_test = test_ecg_arrhythmia_dataset_scan()
    
    # Test 2: Combined dataset loading
    if structure_test:
        loading_test, _ = test_combined_dataset_loading()
        
        # Test 3: Phase 4 training with combined dataset
        if loading_test:
            print("\n" + "=" * 60)
            response = input("\nRun Phase 4 training test with combined dataset? (y/n): ")
            if response.lower() == 'y':
                test_phase4_with_combined_dataset()
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    if structure_test:
        print("✅ ECG Arrhythmia dataset integration: READY")
        print("\nNEXT STEPS:")
        print("1. Download full ECG Arrhythmia dataset if not already done")
        print("2. Run with larger record counts for production training")
        print("3. Monitor MI detection sensitivity improvements")
        print("4. Consider adjusting label mapping if needed")
    else:
        print("❌ ECG Arrhythmia dataset integration: NOT READY")
        print("\nREQUIRED ACTIONS:")
        print("1. Download ECG Arrhythmia dataset from PhysioNet")
        print("2. Extract to data/raw/ecg-arrhythmia-dataset/")
        print("3. Verify WFDBRecords folder structure")
        print("4. Re-run this test")
    
    print("\nTo use combined dataset in your training:")
    print("```python")
    print("from app.utils.dataset_manager import run_combined_dataset_loading")
    print("X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(")
    print("    ptbxl_max_records=5000,")
    print("    arrhythmia_max_records=2000,")
    print("    target_mi_records=1000,")
    print("    sampling_rate=100")
    print(")")
    print("```")


if __name__ == "__main__":
    main()