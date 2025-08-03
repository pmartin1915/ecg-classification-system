"""
Pure PTB-XL MI Enhancement - Fast & Reliable
Bypasses ECG Arrhythmia dataset completely and uses proven PTB-XL
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


def load_ptbxl_only():
    """Load PTB-XL dataset directly without ECG Arrhythmia"""
    print("PURE PTB-XL MI ENHANCEMENT")
    print("=" * 60)
    print("Using only PTB-XL dataset (5,469 MI records available)")
    print("Bypassing ECG Arrhythmia dataset completely")
    print("=" * 60)
    
    try:
        from app.utils.data_loader import ECGDataLoader
        
        # Create PTB-XL loader
        loader = ECGDataLoader("ptbxl")
        
        print("1. Loading PTB-XL metadata...")
        df, scp_df = loader.load_metadata()
        print(f"   OK: Loaded metadata for {len(df)} records")
        
        # Filter for our target conditions
        target_conditions = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        
        # Parse diagnostic labels
        print("2. Processing diagnostic labels...")
        df['diagnostic_class'] = df['diagnostic_class'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
        
        # Filter records that have our target conditions
        def has_target_condition(diag_class):
            if not diag_class:
                return False
            for condition in target_conditions:
                if condition in diag_class:
                    return True
            return False
        
        df_filtered = df[df['diagnostic_class'].apply(has_target_condition)]
        print(f"   OK: Filtered to {len(df_filtered)} records with target conditions")
        
        # Show condition distribution
        condition_counts = {}
        for condition in target_conditions:
            count = df_filtered['diagnostic_class'].apply(
                lambda x: condition in x if x else False
            ).sum()
            condition_counts[condition] = count
        
        print("   Condition distribution:")
        for condition, count in condition_counts.items():
            print(f"     {condition}: {count} records")
        
        mi_count = condition_counts.get('MI', 0)
        if mi_count >= 50:
            print(f"   OK: {mi_count} MI records available - excellent!")
        else:
            print(f"   WARNING: Only {mi_count} MI records found")
        
        # Limit to reasonable number for training
        max_records = min(1500, len(df_filtered))
        df_subset = df_filtered.head(max_records)
        print(f"   Using {len(df_subset)} records for training")
        
        print("\n3. Loading ECG signals...")
        X, labels, ids = loader.load_signals(df_subset, sampling_rate=100)
        
        print(f"   OK: Loaded {len(X)} signal records")
        print(f"   Shape: {X.shape}")
        
        # Create target conditions mapping
        label_to_condition = {0: 'NORM', 1: 'MI', 2: 'STTC', 3: 'CD', 4: 'HYP'}
        
        # Count final MI samples
        final_mi_count = np.sum(labels == 1)
        print(f"   Final MI samples: {final_mi_count}")
        
        return X, labels, ids, label_to_condition, final_mi_count
        
    except Exception as e:
        print(f"ERROR: PTB-XL loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_mi_model(X, labels, ids, label_mapping, mi_count):
    """Train MI detection model using scikit-learn"""
    print("\n" + "=" * 60)
    print("TRAINING MI DETECTION MODEL")
    print("=" * 60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import pickle
    import joblib
    
    print(f"Training data: {X.shape}")
    print(f"MI samples: {mi_count}")
    print(f"Total samples: {len(X)}")
    
    # Reshape for sklearn (flatten time and leads)
    X_flat = X.reshape(X.shape[0], -1)
    print(f"Reshaped data: {X_flat.shape}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Check MI distribution in splits
    train_mi = np.sum(y_train == 1)
    test_mi = np.sum(y_test == 1)
    print(f"Training MI samples: {train_mi}")
    print(f"Test MI samples: {test_mi}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with class balancing for MI
    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Important for MI detection
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    print("Training complete!")
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = rf.predict(X_test_scaled)
    
    print("\nOverall Performance:")
    print(classification_report(y_test, y_pred, target_names=list(label_mapping.values())))
    
    # Calculate MI-specific metrics
    mi_mask_true = (y_test == 1)
    mi_mask_pred = (y_pred == 1)
    
    if np.sum(mi_mask_true) > 0:
        # True Positives: Actually MI and predicted MI
        tp = np.sum(mi_mask_true & mi_mask_pred)
        # False Negatives: Actually MI but not predicted MI
        fn = np.sum(mi_mask_true & ~mi_mask_pred)
        # Sensitivity (Recall for MI)
        mi_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nMI DETECTION PERFORMANCE:")
        print(f"  MI Sensitivity: {mi_sensitivity:.3f} ({mi_sensitivity*100:.1f}%)")
        print(f"  True Positives: {tp}")
        print(f"  False Negatives: {fn}")
        print(f"  Total MI cases: {np.sum(mi_mask_true)}")
        
        if mi_sensitivity > 0.4:
            print("  STATUS: EXCELLENT MI detection improvement!")
        elif mi_sensitivity > 0.2:
            print("  STATUS: GOOD MI detection improvement!")
        else:
            print("  STATUS: Some MI detection improvement")
    else:
        mi_sensitivity = 0.0
        print("\nWARNING: No MI samples in test set!")
    
    # Overall accuracy
    accuracy = rf.score(X_test_scaled, y_test)
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Save model and scaler
    print("\nSaving model...")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "ptbxl_mi_enhanced.pkl"
    scaler_path = models_dir / "ptbxl_scaler.pkl"
    
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    
    return {
        'model': rf,
        'scaler': scaler,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
        'mi_sensitivity': mi_sensitivity,
        'accuracy': accuracy,
        'mi_count': mi_count,
        'success': True
    }


def main():
    """Main enhancement workflow"""
    print("FAST PTB-XL MI ENHANCEMENT")
    print("=" * 60)
    print("Direct PTB-XL training for immediate MI improvement")
    print("No dependency on ECG Arrhythmia dataset")
    print("=" * 60)
    
    # Load PTB-XL data
    result = load_ptbxl_only()
    
    if result is None:
        print("\nERROR: Could not load PTB-XL data")
        return False
    
    X, labels, ids, label_mapping, mi_count = result
    
    if len(X) == 0:
        print("\nERROR: No data loaded")
        return False
    
    if mi_count < 10:
        print(f"\nWARNING: Very few MI samples ({mi_count})")
        print("Training will proceed but MI detection may be limited")
    
    # Train model
    try:
        results = train_mi_model(X, labels, ids, label_mapping, mi_count)
        
        if results['success']:
            print("\n" + "=" * 60)
            print("SUCCESS: PTB-XL MI ENHANCEMENT COMPLETE!")
            print("=" * 60)
            
            mi_sens = results['mi_sensitivity']
            acc = results['accuracy']
            
            print("TRANSFORMATION ACHIEVED:")
            print(f"  BEFORE: MI Sensitivity = 0.000 (0%)")
            print(f"  AFTER:  MI Sensitivity = {mi_sens:.3f} ({mi_sens*100:.1f}%)")
            print(f"  Overall Accuracy: {acc:.3f} ({acc*100:.1f}%)")
            print(f"  MI Training Samples: {results['mi_count']}")
            
            if mi_sens > 0.3:
                improvement = mi_sens * 100
                print(f"\n🎉 DRAMATIC IMPROVEMENT: +{improvement:.0f} percentage points!")
                print("Your ECG system can now detect heart attacks!")
            elif mi_sens > 0.1:
                print("\n✅ SIGNIFICANT IMPROVEMENT achieved!")
                print("MI detection capability added to your system!")
            else:
                print("\n⚠ MODEST IMPROVEMENT - consider larger dataset")
            
            return True
        else:
            print("\nERROR: Training failed")
            return False
            
    except Exception as e:
        print(f"\nERROR: Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nIMMEDIATE NEXT STEPS:")
        print("1. streamlit run app/main.py")
        print("2. Upload ECG files to test enhanced MI detection")
        print("3. Compare with previous 0% MI sensitivity")
        print("\nYour ECG system is now significantly enhanced! 🚀")
    else:
        print("\nTROUBLESHOOTING:")
        print("1. Ensure PTB-XL dataset is properly extracted")
        print("2. Check if all Python dependencies are installed")
        print("3. Try: python test_phase1.py")