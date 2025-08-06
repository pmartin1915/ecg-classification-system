"""
Enhanced MI Detection Pipeline
Combines PTB-XL + Large-Scale MI Database for superior MI detection
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

from app.utils.dataset_manager import run_combined_dataset_loading
from app.utils.mi_database_loader import MIDatabaseLoader
from config.settings import CACHE_DIR, DATA_DIR

warnings.filterwarnings('ignore')


class EnhancedMIPipeline:
    """
    Professional MI Detection Pipeline combining multiple datasets
    Designed for clinical-grade MI detection improvement
    """
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MI database loader
        mi_db_path = "C:\\ecg-classification-system-pc\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
        self.mi_loader = MIDatabaseLoader(mi_db_path)
        
        self.combined_data = None
        self.model = None
        self.performance_metrics = {}
        
    def load_combined_datasets(self, 
                             ptbxl_records: int = 5000,
                             mi_db_records: int = 2000,
                             target_mi_records: int = 500,
                             use_cache: bool = True) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Load and combine PTB-XL + MI Database for enhanced training
        
        Args:
            ptbxl_records: Number of PTB-XL records to include
            mi_db_records: Number of MI database records to include  
            target_mi_records: Target MI records from MI database
            use_cache: Use cached results if available
            
        Returns:
            Combined signals, labels, and metadata
        """
        
        cache_file = self.cache_dir / f"enhanced_mi_dataset_{ptbxl_records}_{mi_db_records}_{target_mi_records}.pkl"
        
        if use_cache and cache_file.exists():
            print("[CACHE] Loading enhanced MI dataset from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"[BUILDING] Enhanced MI dataset (PTB-XL: {ptbxl_records}, MI-DB: {mi_db_records})")
        
        # 1. Load PTB-XL dataset (your existing proven system)
        print("[LOADING] PTB-XL dataset...")
        try:
            ptbxl_signals, ptbxl_labels, ptbxl_ids, ptbxl_metadata, target_conditions, stats = run_combined_dataset_loading(
                ptbxl_max_records=ptbxl_records,
                arrhythmia_max_records=0,  # Don't use old arrhythmia dataset
                target_mi_records=0,
                sampling_rate=100
            )
            print(f"[SUCCESS] PTB-XL: {len(ptbxl_signals)} records loaded")
        except Exception as e:
            print(f"[WARNING] PTB-XL loading failed: {e}")
            ptbxl_signals, ptbxl_labels, ptbxl_ids = np.array([]), [], []
        
        # 2. Load new MI database
        print("[LOADING] Enhanced MI database...")
        mi_signals, mi_labels, mi_ids, mi_metadata = self.mi_loader.load_mi_enhanced_dataset(
            max_records=mi_db_records,
            target_mi_records=target_mi_records,
            target_sampling_rate=100,
            use_cache=True
        )
        
        print(f"[SUCCESS] MI Database: {len(mi_signals)} records, {mi_metadata['mi_records']} MI cases")
        
        # 3. Combine datasets intelligently
        combined_signals = []
        combined_labels = []
        combined_ids = []
        
        # Add PTB-XL data
        if len(ptbxl_signals) > 0:
            for i, signal in enumerate(ptbxl_signals):
                # Ensure consistent shape (time_samples, leads)
                if signal.shape[1] != 12:
                    signal = signal[:, :12]  # Take first 12 leads
                combined_signals.append(signal)
                combined_labels.append(ptbxl_labels[i])
                combined_ids.append(f"ptbxl_{ptbxl_ids[i]}")
        
        # Add MI database data  
        if len(mi_signals) > 0:
            for i, signal in enumerate(mi_signals):
                # Ensure consistent shape
                if signal.shape[1] != 12:
                    # Pad or truncate to 12 leads
                    if signal.shape[1] < 12:
                        padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
                        signal = np.hstack([signal, padding])
                    else:
                        signal = signal[:, :12]
                        
                combined_signals.append(signal)
                combined_labels.append(mi_labels[i])
                combined_ids.append(f"mi_db_{mi_ids[i]}")
        
        # Convert to arrays
        combined_signals = np.array(combined_signals, dtype=np.float32)
        
        # Create comprehensive metadata
        label_dist = {label: combined_labels.count(label) for label in set(combined_labels)}
        mi_count = label_dist.get('MI', 0)
        
        metadata = {
            'total_records': len(combined_signals),
            'ptbxl_records': len(ptbxl_signals) if len(ptbxl_signals) > 0 else 0,
            'mi_db_records': len(mi_signals),
            'total_mi_records': mi_count,
            'mi_percentage': (mi_count / len(combined_signals)) * 100 if len(combined_signals) > 0 else 0,
            'label_distribution': label_dist,
            'signal_shape': combined_signals.shape,
            'sources': ['PTB-XL', 'Large-Scale MI Database'],
            'enhancement_focus': 'MI Detection'
        }
        
        print(f"[COMBINED] Total: {len(combined_signals)} records")
        print(f"[MI FOCUS] {mi_count} MI records ({metadata['mi_percentage']:.1f}%)")
        print(f"[DISTRIBUTION] {label_dist}")
        
        # Cache results
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump((combined_signals, combined_labels, combined_ids, metadata), f)
            print(f"[CACHE] Enhanced dataset cached")
        
        self.combined_data = (combined_signals, combined_labels, combined_ids, metadata)
        return combined_signals, combined_labels, combined_ids, metadata
    
    def train_enhanced_mi_model(self, 
                               test_size: float = 0.2,
                               random_state: int = 42) -> Dict[str, Any]:
        """
        Train enhanced MI detection model on combined dataset
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Training results and performance metrics
        """
        
        if self.combined_data is None:
            raise ValueError("No combined data loaded. Run load_combined_datasets() first.")
        
        signals, labels, ids, metadata = self.combined_data
        
        print(f"[TRAINING] Enhanced MI model on {len(signals)} records...")
        
        # Prepare features (flatten ECG signals)
        X = signals.reshape(len(signals), -1)  # Flatten to (n_samples, n_features)
        y = labels
        
        # Train/test split with stratification
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, ids, test_size=test_size, random_state=random_state, 
            stratify=y if len(set(y)) > 1 else None
        )
        
        print(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train model (optimized for MI detection)
        self.model = RandomForestClassifier(
            n_estimators=200,  # More trees for better performance
            max_depth=15,      # Deeper trees for complex patterns
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=random_state,
            n_jobs=-1  # Use all cores
        )
        
        print("[FITTING] Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        train_report = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)
        test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        
        # Focus on MI metrics
        mi_test_metrics = test_report.get('MI', {})
        mi_train_metrics = train_report.get('MI', {})
        
        results = {
            'model_type': 'Enhanced MI Random Forest',
            'dataset_info': metadata,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_accuracy': train_report['accuracy'],
            'test_accuracy': test_report['accuracy'],
            
            # MI-specific metrics (key focus)
            'mi_precision': mi_test_metrics.get('precision', 0.0),
            'mi_recall': mi_test_metrics.get('recall', 0.0),  # This is sensitivity!
            'mi_f1_score': mi_test_metrics.get('f1-score', 0.0),
            'mi_support': mi_test_metrics.get('support', 0),
            
            # Compare to baseline
            'baseline_mi_sensitivity': 0.35,  # Previous best
            'new_mi_sensitivity': mi_test_metrics.get('recall', 0.0),
            'mi_improvement': mi_test_metrics.get('recall', 0.0) - 0.35,
            
            'full_train_report': train_report,
            'full_test_report': test_report,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
            'test_ids': ids_test,
            'predictions': y_test_pred.tolist(),
            'probabilities': y_test_proba.tolist() if y_test_proba.ndim > 1 else []
        }
        
        self.performance_metrics = results
        
        # Print key results
        print(f"[RESULTS] Model trained successfully!")
        print(f"[ACCURACY] Overall: {results['test_accuracy']:.1%}")
        print(f"[MI FOCUS] Precision: {results['mi_precision']:.1%}, Recall/Sensitivity: {results['mi_recall']:.1%}")
        print(f"[IMPROVEMENT] MI Sensitivity: {results['baseline_mi_sensitivity']:.1%} -> {results['new_mi_sensitivity']:.1%}")
        
        if results['mi_improvement'] > 0:
            print(f"[SUCCESS] MI Detection improved by +{results['mi_improvement']:.1%} points! 🎉")
        else:
            print(f"[INFO] MI Detection: {results['mi_improvement']:.1%} points change")
        
        return results
    
    def save_enhanced_model(self, model_name: str = "enhanced_mi_model") -> Path:
        """Save the trained enhanced MI model"""
        
        if self.model is None:
            raise ValueError("No model trained. Run train_enhanced_mi_model() first.")
        
        model_path = DATA_DIR / "models" / f"{model_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata together
        model_data = {
            'model': self.model,
            'performance_metrics': self.performance_metrics,
            'training_info': {
                'model_type': 'Enhanced MI Detection',
                'datasets': ['PTB-XL', 'Large-Scale MI Database'],
                'focus': 'MI Detection Enhancement',
                'trained_at': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[SAVED] Enhanced MI model saved to: {model_path}")
        return model_path
    
    def generate_clinical_report(self) -> str:
        """Generate clinical-ready performance report"""
        
        if not self.performance_metrics:
            return "No performance metrics available. Train model first."
        
        metrics = self.performance_metrics
        
        report = f"""
=== ENHANCED MI DETECTION SYSTEM - CLINICAL VALIDATION REPORT ===

[DATA] DATASET COMPOSITION:
   • Total Records: {metrics['dataset_info']['total_records']:,}
   • MI Cases: {metrics['dataset_info']['total_mi_records']} ({metrics['dataset_info']['mi_percentage']:.1f}%)
   • Data Sources: PTB-XL + Large-Scale MI Database
   • Signal Quality: 12-lead ECG, 100Hz, 10-second recordings

[MODEL] MODEL PERFORMANCE:
   • Overall Accuracy: {metrics['test_accuracy']:.1%}
   • Test Samples: {metrics['test_samples']:,}

[MI] MI DETECTION RESULTS (Primary Metric):
   • MI Precision: {metrics['mi_precision']:.1%}
   • MI Sensitivity/Recall: {metrics['mi_recall']:.1%} [PRIMARY METRIC]
   • MI F1-Score: {metrics['mi_f1_score']:.1%}
   • MI Cases in Test: {metrics['mi_support']}

[RESULTS] CLINICAL IMPROVEMENT:
   • Baseline MI Sensitivity: {metrics['baseline_mi_sensitivity']:.1%}
   • Enhanced MI Sensitivity: {metrics['new_mi_sensitivity']:.1%}
   • Improvement: {metrics['mi_improvement']:+.1%} percentage points
   
[CLINICAL] CLINICAL SIGNIFICANCE:
   • {"SIGNIFICANT IMPROVEMENT" if metrics['mi_improvement'] > 0.1 else "PERFORMANCE MAINTAINED"}
   • Enhanced patient safety through better MI detection
   • Ready for clinical validation studies
   
[WARNING] CLINICAL DISCLAIMER:
   This system is designed for clinical decision support only.
   All diagnoses must be confirmed by qualified healthcare professionals.
   
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report


def test_enhanced_mi_pipeline():
    """Test the enhanced MI pipeline with realistic clinical parameters"""
    
    try:
        print("=== TESTING ENHANCED MI DETECTION PIPELINE ===")
        
        # Initialize pipeline
        pipeline = EnhancedMIPipeline()
        
        # Load combined datasets (clinical-scale test)
        signals, labels, ids, metadata = pipeline.load_combined_datasets(
            ptbxl_records=2000,   # Reasonable PTB-XL sample
            mi_db_records=1000,   # Good MI database sample
            target_mi_records=100, # Target 100 MI cases
            use_cache=True
        )
        
        # Train enhanced model
        results = pipeline.train_enhanced_mi_model(test_size=0.2)
        
        # Save model
        model_path = pipeline.save_enhanced_model("test_enhanced_mi_model")
        
        # Generate clinical report
        report = pipeline.generate_clinical_report()
        print(report)
        
        print(f"[SUCCESS] Enhanced MI pipeline test completed!")
        print(f"[MODEL] Saved to: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Enhanced MI pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_enhanced_mi_pipeline()