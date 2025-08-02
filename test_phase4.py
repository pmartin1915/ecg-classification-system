"""
Test script for Phase 4 model training pipeline - FIXED VERSION
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.training.phase4_model_training import Phase4Pipeline as TrainingPipeline
from config.model_config import ModelTrainingConfig
from config.settings import DATA_DIR


def test_model_training_pipeline():
    """Test the model training pipeline with feature data"""
    print("TESTING PHASE 4 MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load feature data from Phase 3
    print("\n1. Loading feature data from Phase 3...")
    try:
        # Try to load from saved Phase 3 results
        phase3_file = DATA_DIR / 'processed' / 'feature_extraction_results.pkl'
        
        if not phase3_file.exists():
            # Try alternative file name
            phase3_file = DATA_DIR / 'processed' / 'phase3_features.pkl'
        
        if phase3_file.exists():
            print(f"Loading from: {phase3_file}")
            with open(phase3_file, 'rb') as f:
                phase3_data = pickle.load(f)
            
            # Extract data
            if 'X_features' in phase3_data:
                X_features = phase3_data['X_features'][:100]  # Use 100 samples for testing
                y_encoded = phase3_data['y_encoded'][:100]
                feature_names = phase3_data['feature_names']
            else:
                # Handle different data structure
                X_features = phase3_data.get('X', phase3_data.get('features'))[:100]
                y_encoded = phase3_data.get('y', phase3_data.get('labels'))[:100]
                feature_names = phase3_data.get('feature_names', 
                                               [f'feature_{i}' for i in range(X_features.shape[1])])
            
            print(f"OK: Loaded {len(X_features)} samples")
            print(f"   Feature shape: {X_features.shape}")
            print(f"   Classes: {len(np.unique(y_encoded))}")
        else:
            print("ERROR: Phase 3 results not found. Creating synthetic data for testing...")
            
            # Create synthetic data
            np.random.seed(42)
            n_samples = 100
            n_features = 50
            n_classes = 5
            
            X_features = np.random.randn(n_samples, n_features)
            y_encoded = np.random.randint(0, n_classes, n_samples)
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            print(f"OK: Created synthetic data: {X_features.shape}")
            
    except Exception as e:
        print(f"ERROR: Error loading feature data: {e}")
        return False
    
    # Step 2: Test configuration
    print("\n2. Testing model configuration...")
    try:
        config = ModelTrainingConfig(
            test_size=0.3,
            val_size=0.2,
            use_smote=True,
            use_hyperparameter_tuning=False,  # Disable for faster testing
            create_visualizations=False  # Disable for testing
        )
        
        print("OK: Configuration created successfully")
        print(f"   Test size: {config.test_size}")
        print(f"   Validation size: {config.val_size}")
        print(f"   SMOTE: {config.use_smote}")
        
    except Exception as e:
        print(f"ERROR: Error creating configuration: {e}")
        return False
    
    # Step 3: Test individual model training
    print("\n3. Testing individual model training...")
    try:
        from models.training import ModelTrainer
        
        trainer = ModelTrainer(config)
        trainer.load_data(X_features, y_encoded, feature_names)
        trainer.prepare_data_splits()
        trainer.scale_features()
        
        # Setup a single model
        trainer.setup_models(['logistic_regression'])
        
        # Train the model
        results = trainer.train_model('Logistic Regression', use_smote=False)
        
        print(f"OK: Model trained successfully")
        print(f"   Accuracy: {results['metrics']['accuracy']:.3f}")
        print(f"   F1-Score: {results['metrics']['f1_weighted']:.3f}")
        
    except Exception as e:
        print(f"ERROR: Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test complete pipeline - FIXED VERSION
    print("\n4. Testing complete training pipeline...")
    try:
        # Convert config object to dictionary - THIS IS THE FIX
        pipeline_config = {
            'test_size': config.test_size,
            'val_size': config.val_size,
            'use_smote': config.use_smote,
            'use_hyperparameter_tuning': config.use_hyperparameter_tuning,
            'model_keys': ['random_forest', 'logistic_regression'],  # Just 2 models for speed
            'create_visualizations': config.create_visualizations,
            'save_models': False
        }
        
        # Use minimal models for testing
        pipeline = TrainingPipeline(pipeline_config)
        
        results = pipeline.run(
            X=X_features,
            y=y_encoded,
            feature_names=feature_names
        )
        
        print(f"OK: Pipeline completed successfully!")
        print(f"   Best model: {results['best_model_name']}")
        print(f"   Test accuracy: {results['test_results']['metrics']['accuracy']:.3f}")
        print(f"   Models trained: {len(results['trainer'].results)}")
        
    except Exception as e:
        print(f"ERROR: Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test model evaluation
    print("\n5. Testing model evaluation...")
    try:
        from models.training import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        # Test confusion matrix calculation
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(trainer.y_test, results['test_results']['predictions'])
        print(f"OK: Confusion matrix calculated: shape {cm.shape}")
        
        # Test metric calculation
        metrics = trainer.evaluate_model(
            trainer.y_test, 
            results['test_results']['predictions']
        )
        print(f"OK: Metrics calculated: {len(metrics)} metrics")
        
    except Exception as e:
        print(f"ERROR: Error in model evaluation: {e}")
        return False
    
    # Step 6: Test model saving
    print("\n6. Testing model saving...")
    try:
        save_dir = DATA_DIR / 'test_models'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = trainer.save_model(
            results['best_model'],
            results['best_model_name'],
            save_dir
        )
        
        print(f"OK: Model saved successfully: {model_path}")
        
        # Test loading
        import joblib
        loaded_model = joblib.load(model_path)
        print("OK: Model loaded successfully")
        
        # Clean up
        import shutil
        shutil.rmtree(save_dir)
        
    except Exception as e:
        print(f"ERROR: Error in model saving: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("SUCCESS: All tests passed! Phase 4 model training is working correctly.")
    return True


def test_full_model_training():
    """Test model training on larger dataset - FIXED VERSION"""
    print("\nTESTING FULL MODEL TRAINING")
    print("WARNING: This will take several minutes...")
    
    # Load feature data
    phase3_file = DATA_DIR / 'processed' / 'feature_extraction_results.pkl'
    
    if not phase3_file.exists():
        phase3_file = DATA_DIR / 'processed' / 'phase3_features.pkl'
    
    if not phase3_file.exists():
        print("ERROR: Phase 3 results not found. Please run Phase 3 first.")
        return
    
    with open(phase3_file, 'rb') as f:
        phase3_data = pickle.load(f)
    
    # Extract data
    if 'X_features' in phase3_data:
        X_features = phase3_data['X_features']
        y_encoded = phase3_data['y_encoded']
        feature_names = phase3_data['feature_names']
    else:
        X_features = phase3_data.get('X', phase3_data.get('features'))
        y_encoded = phase3_data.get('y', phase3_data.get('labels'))
        feature_names = phase3_data.get('feature_names', 
                                       [f'feature_{i}' for i in range(X_features.shape[1])])
    
    # Limit samples if needed
    n_samples = min(1000, len(X_features))
    X_features = X_features[:n_samples]
    y_encoded = y_encoded[:n_samples]
    
    print(f"Using {n_samples} samples for full test")
    
    # Create configuration object first
    config = ModelTrainingConfig(
        use_hyperparameter_tuning=True,
        create_visualizations=True
    )
    
    # Use the helper function to safely convert config to dictionary
    pipeline_config = create_config_dict_from_object(config)
    
    # Run pipeline
    pipeline = TrainingPipeline(pipeline_config)
    results = pipeline.run(
        X=X_features,
        y=y_encoded,
        feature_names=feature_names
    )
    
    # Print detailed results
    print(f"\nModel Training Results:")
    print(results['comparison_df'].to_string(index=False, float_format='%.3f'))
    
    # Print clinical metrics
    test_metrics = results['test_results']['metrics']
    print(f"\nClinical Performance:")
    print(f"   MI Sensitivity:   {test_metrics.get('MI_sensitivity', 0):.3f}")
    print(f"   MI Specificity:   {test_metrics.get('MI_specificity', 0):.3f}")
    print(f"   NORM Specificity: {test_metrics.get('NORM_specificity', 0):.3f}")
    
    # Print top features if available
    best_results = results['trainer'].results[results['best_model_name']]
    if best_results.get('feature_importance') is not None:
        importance = best_results['feature_importance']
        top_indices = np.argsort(importance)[-10:][::-1]
        
        print(f"\nTop 10 Important Features:")
        for i, idx in enumerate(top_indices):
            print(f"   {i+1:2d}. {feature_names[idx]:<40} {importance[idx]:.4f}")


def test_model_comparison():
    """Test comparing different model configurations - FIXED VERSION"""
    print("\nTESTING MODEL COMPARISON")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    n_classes = 5
    
    # Create features with some structure
    X = np.random.randn(n_samples, n_features)
    
    # Create labels with some pattern
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_classes):
        mask = slice(i * n_samples // n_classes, (i + 1) * n_samples // n_classes)
        y[mask] = i
        # Add some class-specific patterns
        X[mask, i] += 1.5
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Test different configurations
    configs = {
        'Standard': ModelTrainingConfig(use_smote=True, use_hyperparameter_tuning=False),
        'No SMOTE': ModelTrainingConfig(use_smote=False, use_hyperparameter_tuning=False),
        'With Tuning': ModelTrainingConfig(use_smote=True, use_hyperparameter_tuning=True),
    }
    
    results_summary = []
    
    for config_name, config in configs.items():
        print(f"\nTesting configuration: {config_name}")
        
        # Convert config object to dictionary - THIS IS THE FIX
        pipeline_config = {
            'test_size': config.test_size,
            'val_size': config.val_size,
            'use_smote': config.use_smote,
            'use_hyperparameter_tuning': config.use_hyperparameter_tuning,
            'model_keys': ['random_forest', 'logistic_regression'],
            'create_visualizations': False,
            'save_models': False
        }
        
        pipeline = TrainingPipeline(pipeline_config)
        results = pipeline.run(
            X=X,
            y=y,
            feature_names=feature_names
        )
        
        test_metrics = results['test_results']['metrics']
        results_summary.append({
            'Configuration': config_name,
            'Best Model': results['best_model_name'],
            'Accuracy': test_metrics['accuracy'],
            'F1-Score': test_metrics['f1_weighted']
        })
    
    # Display comparison
    comparison_df = pd.DataFrame(results_summary)
    print("\nConfiguration Comparison:")
    print(comparison_df.to_string(index=False, float_format='%.3f'))


def create_config_dict_from_object(config_obj):
    """
    Helper function to convert ModelTrainingConfig object to dictionary
    This addresses the subscriptable error and missing attributes
    """
    return {
        'test_size': getattr(config_obj, 'test_size', 0.2),
        'val_size': getattr(config_obj, 'val_size', 0.15),
        'use_smote': getattr(config_obj, 'use_smote', True),
        'use_hyperparameter_tuning': getattr(config_obj, 'use_hyperparameter_tuning', False),
        'model_keys': getattr(config_obj, 'model_keys', ['random_forest', 'logistic_regression', 'gradient_boosting']),
        'create_visualizations': getattr(config_obj, 'create_visualizations', True),
        'save_models': getattr(config_obj, 'save_models', True),
        'save_dir': getattr(config_obj, 'save_dir', DATA_DIR / 'models'),
        'class_weights': getattr(config_obj, 'class_weights', 'balanced'),
        'random_state': getattr(config_obj, 'random_state', 42),
        'n_jobs': getattr(config_obj, 'n_jobs', -1)
    }


if __name__ == "__main__":
    # Run basic tests first
    if test_model_training_pipeline():
        print("\n" + "=" * 60)
        
        # Test model comparison
        test_model_comparison()
        
        print("\n" + "=" * 60)
        response = input("\nRun full model training test? (y/n): ")
        if response.lower() == 'y':
            test_full_model_training()
    else:
        print("\nERROR: Please fix the errors before proceeding.")
        print("\nCommon issues:")
        print("1. Make sure Phase 3 has been run successfully")
        print("2. Check that all dependencies are installed")
        print("3. Ensure you have scikit-learn, imbalanced-learn installed")
        print("4. For XGBoost/LightGBM support, install them separately")
        print("\nThe main fix applied:")
        print("5. Convert ModelTrainingConfig object to dictionary before passing to pipeline")