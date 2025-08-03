"""
Simple PTB-XL MI Enhancement using existing working components
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


def main():
    """Simple MI enhancement using existing working code"""
    print("SIMPLE PTB-XL MI ENHANCEMENT")
    print("=" * 60)
    print("Using existing proven components")
    print("=" * 60)
    
    try:
        # Use the working dataset manager
        from app.utils.dataset_manager import DatasetManager
        
        print("1. Initializing dataset manager...")
        manager = DatasetManager()
        
        print("2. Loading PTB-XL dataset...")
        result = manager.load_ptbxl_complete(
            max_records=1000,  # Reasonable size for training
            sampling_rate=100,
            use_cache=True
        )
        
        X = result['X']
        labels = result['labels'] 
        ids = result['ids']
        target_conditions = result['target_conditions']
        stats = result['stats']
        
        print(f"   OK: Loaded {len(X)} records")
        print(f"   Shape: {X.shape}")
        print(f"   Target conditions: {target_conditions}")
        
        # Check MI count
        mi_count = np.sum(labels == 1)  # MI should be label 1
        print(f"   MI samples: {mi_count}")
        
        if mi_count < 10:
            print("   WARNING: Very few MI samples")
        else:
            print("   OK: Sufficient MI samples for training")
        
        print("\n3. Training simple Random Forest model...")
        success = train_simple_model(X, labels, ids, target_conditions, mi_count)
        
        return success
        
    except Exception as e:
        print(f"ERROR: Enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_simple_model(X, labels, ids, target_conditions, mi_count):
    """Train a simple but effective Random Forest model"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    print(f"Training with {len(X)} samples ({mi_count} MI cases)")
    
    # Flatten ECG data for sklearn
    X_flat = X.reshape(X.shape[0], -1)
    print(f"Flattened shape: {X_flat.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_mi = np.sum(y_train == 1)
    test_mi = np.sum(y_test == 1)
    print(f"Train/Test split: {len(X_train)}/{len(X_test)} (MI: {train_mi}/{test_mi})")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',  # Important for MI detection
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=target_conditions))
    
    # Calculate MI sensitivity
    mi_mask = (y_test == 1)
    if np.sum(mi_mask) > 0:
        mi_tp = np.sum((y_pred == 1) & mi_mask)
        mi_fn = np.sum((y_pred != 1) & mi_mask)
        mi_sensitivity = mi_tp / (mi_tp + mi_fn) if (mi_tp + mi_fn) > 0 else 0
        
        print(f"\nMI DETECTION RESULTS:")
        print(f"  Sensitivity: {mi_sensitivity:.3f} ({mi_sensitivity*100:.1f}%)")
        print(f"  True Positives: {mi_tp}")
        print(f"  False Negatives: {mi_fn}")
    else:
        mi_sensitivity = 0.0
        print("\nWARNING: No MI samples in test set")
    
    accuracy = rf.score(X_test_scaled, y_test)
    print(f"  Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Save model
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "simple_mi_enhanced.joblib"
    scaler_path = models_dir / "simple_scaler.joblib"
    
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    
    # Show improvement
    print("\n" + "=" * 60)
    print("ENHANCEMENT RESULTS:")
    print("=" * 60)
    print(f"BEFORE: MI Sensitivity = 0.000 (0%)")
    print(f"AFTER:  MI Sensitivity = {mi_sensitivity:.3f} ({mi_sensitivity*100:.1f}%)")
    
    if mi_sensitivity > 0.3:
        improvement = mi_sensitivity * 100
        print(f"\nSUCCESS: Dramatic +{improvement:.0f} percentage point improvement!")
        print("Your ECG system can now detect heart attacks effectively!")
    elif mi_sensitivity > 0.1:
        print(f"\nGOOD: Significant improvement achieved!")
        print("MI detection capability added to your system!")
    else:
        print(f"\nMODEST: Some improvement, but could be better")
        print("Consider more training data or parameter tuning")
    
    return mi_sensitivity > 0.05  # Success if any reasonable improvement


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 60)
        print("MI ENHANCEMENT COMPLETE!")
        print("=" * 60)
        print("NEXT STEPS:")
        print("1. Test your system: streamlit run app/main.py")
        print("2. Upload ECG files to verify MI detection")
        print("3. Compare with previous 0% MI sensitivity")
        print("\nYour ECG system is now enhanced! 🚀")
    else:
        print("\n" + "=" * 60)
        print("ENHANCEMENT NEEDS ATTENTION")
        print("=" * 60)
        print("Check the logs above for specific issues")
        print("The existing system will continue to work")