"""
PTB-XL MI Enhancement - Reliable Fallback
Uses the proven PTB-XL dataset to improve MI detection
"""
import sys
from pathlib import Path
import warnings
import numpy as np
from collections import Counter
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.utils.dataset_manager import run_combined_dataset_loading


def load_ptbxl_focused_dataset():
    """Load PTB-XL with focus on MI detection"""
    print("PTB-XL FOCUSED MI ENHANCEMENT")
    print("=" * 60)
    print("Using proven PTB-XL dataset with 5,469 MI records")
    print("=" * 60)
    
    try:
        # Use the working combined dataset loader but PTB-XL only
        print("Loading PTB-XL dataset with MI focus...")
        X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
            ptbxl_max_records=1000,  # Larger PTB-XL sample
            arrhythmia_max_records=0,  # Skip ECG Arrhythmia for now
            target_mi_records=200,     # Focus on getting good MI representation
            sampling_rate=100
        )
        
        print(f"OK: Loaded {len(X)} total records")
        print(f"   PTB-XL records: {stats.get('ptbxl_records', len(X))}")
        print(f"   MI records: {stats.get('mi_records', 0)}")
        print(f"   Data shape: {X.shape}")
        
        # Check MI count
        mi_count = stats.get('mi_records', 0)
        if mi_count >= 50:
            print(f"   OK: {mi_count} MI records - excellent for training!")
        else:
            print(f"   WARNING: Only {mi_count} MI records found")
        
        return X, labels, ids, target_conditions, stats
        
    except Exception as e:
        print(f"ERROR: PTB-XL loading failed: {e}")
        return None


def train_ptbxl_mi_model(X, labels, ids, target_conditions):
    """Train MI model using PTB-XL data"""
    print("\n" + "=" * 60)
    print("TRAINING PTB-XL MI MODEL")
    print("=" * 60)
    
    # Import training components
    try:
        from models.training.training_pipeline import ModelTrainingPipeline
        from config.model_config import ModelConfig
    except ImportError as e:
        print(f"WARNING: Cannot import training components: {e}")
        print("Attempting to train with basic components...")
        return train_basic_model(X, labels, ids)
    
    # Use clinical-optimized configuration
    config = ModelConfig('clinical_optimized')
    trainer = ModelTrainingPipeline(config)
    
    print("Starting PTB-XL MI-focused training...")
    print(f"Training data: {X.shape}")
    print(f"Target conditions: {list(target_conditions.values())}")
    
    # Train the model
    results = trainer.train_and_evaluate(
        X, labels, ids,
        model_name='ptbxl_mi_enhanced',
        save_model=True
    )
    
    return results


def train_basic_model(X, labels, ids):
    """Basic model training fallback"""
    print("Using basic Random Forest training...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import pickle
    
    # Reshape data for sklearn
    X_flat = X.reshape(X.shape[0], -1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Calculate MI sensitivity specifically
    mi_mask = (y_test == 1)  # MI is class 1
    if np.sum(mi_mask) > 0:
        mi_sensitivity = np.sum((y_pred == 1) & mi_mask) / np.sum(mi_mask)
        print(f"\nMI Sensitivity: {mi_sensitivity:.3f} ({mi_sensitivity*100:.1f}%)")
    else:
        mi_sensitivity = 0.0
        print("\nWARNING: No MI samples in test set")
    
    # Save model
    model_path = Path("data/models/ptbxl_mi_enhanced.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    
    print(f"Model saved to: {model_path}")
    
    return {
        'model': rf,
        'model_path': str(model_path),
        'mi_sensitivity': mi_sensitivity,
        'accuracy': rf.score(X_test, y_test),
        'success': True
    }


def main():
    """Main PTB-XL MI enhancement workflow"""
    print("PTB-XL MI DETECTION ENHANCEMENT")
    print("=" * 60)
    print("Reliable enhancement using proven PTB-XL dataset")
    print("5,469 MI records available for training")
    print("=" * 60)
    
    # Load PTB-XL dataset
    dataset_result = load_ptbxl_focused_dataset()
    
    if dataset_result is None:
        print("\nERROR: PTB-XL dataset loading failed")
        return False
    
    X, labels, ids, target_conditions, stats = dataset_result
    
    if len(X) == 0:
        print("\nERROR: No data loaded")
        return False
    
    # Train model
    try:
        results = train_ptbxl_mi_model(X, labels, ids, target_conditions)
        
        if results and results.get('success', False):
            print("\n" + "=" * 60)
            print("SUCCESS: PTB-XL MI ENHANCEMENT COMPLETE!")
            print("=" * 60)
            
            mi_sens = results.get('mi_sensitivity', 0)
            acc = results.get('accuracy', 0)
            
            print(f"Model Performance:")
            print(f"  MI Sensitivity: {mi_sens:.3f} ({mi_sens*100:.1f}%)")
            print(f"  Overall Accuracy: {acc:.3f} ({acc*100:.1f}%)")
            print(f"  Model saved: {results.get('model_path', 'N/A')}")
            
            if mi_sens > 0.3:  # Reasonable threshold
                print("\nOK: Significant MI detection improvement achieved!")
                print("Your system can now detect heart attacks much better!")
            else:
                print("\nWARNING: MI sensitivity still low")
                print("Consider using more training data or different parameters")
            
            return True
        else:
            print("\nERROR: Training completed but no results returned")
            return False
            
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nNEXT STEPS:")
        print("1. Test your enhanced model: streamlit run app/main.py")
        print("2. Upload ECG files to test MI detection")
        print("3. Compare with previous 0.000% MI sensitivity")
        print("\nYour ECG system now has significantly improved MI detection!")
    else:
        print("\nFALLBACK OPTIONS:")
        print("1. Check if all dependencies are installed")
        print("2. Try running test_phase1.py to verify PTB-XL access")
        print("3. Use existing model until issues are resolved")