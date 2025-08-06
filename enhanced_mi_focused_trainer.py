"""
Focused MI Enhancement Training System
Specifically targets MI detection improvement using the new large-scale database
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

from app.utils.mi_database_loader import MIDatabaseLoader
from config.settings import CACHE_DIR, DATA_DIR

warnings.filterwarnings('ignore')


class FocusedMITrainer:
    """
    Focused MI Detection Enhancement System
    Designed to maximize MI detection improvement for clinical presentation
    """
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.data_dir = DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MI database loader
        mi_db_path = "C:\\ecg-classification-system-pc\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
        self.mi_loader = MIDatabaseLoader(mi_db_path)
        
        self.model = None
        self.training_data = None
        self.performance_results = {}
        
    def load_mi_focused_dataset(self, 
                               target_mi_records: int = 50,
                               total_records: int = 3000,
                               use_cache: bool = True) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Load dataset with maximum MI focus for clinical improvement
        
        Args:
            target_mi_records: Target number of MI records to find
            total_records: Total records to load for balanced training
            use_cache: Use cached results if available
            
        Returns:
            Signals, labels, and metadata optimized for MI detection
        """
        
        cache_file = self.cache_dir / f"focused_mi_dataset_{target_mi_records}_{total_records}.pkl"
        
        if use_cache and cache_file.exists():
            print("[CACHE] Loading focused MI dataset...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"[BUILDING] MI-Focused Dataset (Target: {target_mi_records} MI records)")
        
        # Load from MI database with aggressive MI targeting
        signals, labels, record_ids, metadata = self.mi_loader.load_mi_enhanced_dataset(
            max_records=total_records,
            target_mi_records=target_mi_records,
            target_sampling_rate=100,
            use_cache=True
        )
        
        # Analyze what we got
        label_counts = {label: labels.count(label) for label in set(labels)}
        mi_count = label_counts.get('MI', 0)
        
        enhanced_metadata = {
            'total_records': len(signals),
            'mi_records': mi_count,
            'mi_percentage': (mi_count / len(signals)) * 100 if len(signals) > 0 else 0,
            'label_distribution': label_counts,
            'signal_shape': signals.shape if len(signals) > 0 else (0, 0, 0),
            'data_source': 'Large-Scale MI Database',
            'focus': 'MI Detection Enhancement',
            'baseline_comparison': {
                'previous_mi_sensitivity': 35.0,  # Your current best
                'target_improvement': 10.0  # Target +10% points
            }
        }
        
        print(f"[SUCCESS] Focused dataset loaded:")
        print(f"[STATS] Total: {len(signals)}, MI: {mi_count} ({enhanced_metadata['mi_percentage']:.1f}%)")
        print(f"[DISTRIBUTION] {label_counts}")
        
        # Cache the focused dataset
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump((signals, labels, record_ids, enhanced_metadata), f)
            print(f"[CACHE] Focused dataset cached")
        
        self.training_data = (signals, labels, record_ids, enhanced_metadata)
        return signals, labels, record_ids, enhanced_metadata
    
    def train_focused_mi_model(self, 
                              test_size: float = 0.2,
                              random_state: int = 42) -> Dict[str, Any]:
        """
        Train MI-focused model optimized for clinical improvement
        
        Args:
            test_size: Test set proportion
            random_state: Random seed for reproducibility
            
        Returns:
            Comprehensive training results for clinical presentation
        """
        
        if self.training_data is None:
            raise ValueError("No training data loaded. Run load_mi_focused_dataset() first.")
        
        signals, labels, record_ids, metadata = self.training_data
        
        if len(signals) == 0:
            raise ValueError("No training signals available.")
        
        print(f"[TRAINING] MI-Focused Model on {len(signals)} records...")
        
        # Prepare features (flatten ECG signals for Random Forest)
        X = signals.reshape(len(signals), -1)  # Shape: (n_samples, time_points * leads)
        y = labels
        
        # Check if we have enough data for train/test split
        if len(X) < 4:
            print(f"[WARNING] Very few samples ({len(X)}). Training on all data, no test set.")
            X_train, X_test = X, X
            y_train, y_test = y, y
            ids_train, ids_test = record_ids, record_ids
        else:
            # Stratified split to ensure MI cases in both train/test
            try:
                X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
                    X, y, record_ids, test_size=test_size, random_state=random_state,
                    stratify=y if len(set(y)) > 1 and min([y.count(label) for label in set(y)]) > 1 else None
                )
            except ValueError:
                # If stratification fails (not enough samples of each class), do simple split
                X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
                    X, y, record_ids, test_size=test_size, random_state=random_state
                )
        
        print(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"[TRAIN LABELS] {[y_train.count(label) for label in set(y_train)]}")
        print(f"[TEST LABELS] {[y_test.count(label) for label in set(y_test)]}")
        
        # Train optimized Random Forest for MI detection
        self.model = RandomForestClassifier(
            n_estimators=300,      # More trees for stability
            max_depth=20,          # Deeper for complex ECG patterns
            min_samples_split=2,   # Lower for small datasets
            min_samples_leaf=1,    # Lower for small datasets
            class_weight='balanced',  # Critical for MI imbalance
            random_state=random_state,
            n_jobs=-1,             # Use all CPU cores
            bootstrap=True,
            oob_score=True         # Out-of-bag score for validation
        )
        
        print("[TRAINING] Fitting Random Forest with MI optimization...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Get prediction probabilities if possible
        try:
            y_test_proba = self.model.predict_proba(X_test)
        except:
            y_test_proba = None
        
        # Calculate comprehensive metrics
        train_report = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)
        test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        
        # Extract MI-specific metrics (primary focus)
        mi_test_metrics = test_report.get('MI', {})
        mi_train_metrics = train_report.get('MI', {})
        
        # Calculate improvement vs baseline
        baseline_mi_sensitivity = 35.0  # Your current best from PTB-XL
        new_mi_sensitivity = mi_test_metrics.get('recall', 0.0) * 100  # Convert to percentage
        mi_improvement = new_mi_sensitivity - baseline_mi_sensitivity
        
        # Compile comprehensive results
        results = {
            'model_info': {
                'type': 'MI-Focused Random Forest',
                'n_estimators': 300,
                'optimization': 'MI Detection Enhancement',
                'data_source': 'Large-Scale MI Database'
            },
            'dataset_info': metadata,
            'training_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'mi_cases_train': y_train.count('MI') if 'MI' in y_train else 0,
                'mi_cases_test': y_test.count('MI') if 'MI' in y_test else 0
            },
            
            # Overall model performance
            'overall_metrics': {
                'train_accuracy': train_report.get('accuracy', 0.0),
                'test_accuracy': test_report.get('accuracy', 0.0),
                'oob_score': getattr(self.model, 'oob_score_', 'Not available')
            },
            
            # MI-specific results (most important)
            'mi_metrics': {
                'precision': mi_test_metrics.get('precision', 0.0),
                'recall_sensitivity': mi_test_metrics.get('recall', 0.0),  # Key clinical metric
                'f1_score': mi_test_metrics.get('f1-score', 0.0),
                'support': mi_test_metrics.get('support', 0)
            },
            
            # Clinical improvement analysis
            'clinical_improvement': {
                'baseline_mi_sensitivity_pct': baseline_mi_sensitivity,
                'enhanced_mi_sensitivity_pct': new_mi_sensitivity,
                'improvement_points': mi_improvement,
                'improvement_percentage': (mi_improvement / baseline_mi_sensitivity) * 100 if baseline_mi_sensitivity > 0 else 0,
                'clinical_significance': 'SIGNIFICANT' if mi_improvement > 5.0 else 'MODEST' if mi_improvement > 0 else 'NEEDS_IMPROVEMENT'
            },
            
            # Detailed reports for analysis
            'detailed_reports': {
                'train_classification_report': train_report,
                'test_classification_report': test_report,
                'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
                'test_predictions': y_test_pred.tolist(),
                'test_probabilities': y_test_proba.tolist() if y_test_proba is not None else []
            }
        }
        
        self.performance_results = results
        
        # Print clinical-focused summary
        print(f"\n[RESULTS] MI-Focused Model Training Complete!")
        print(f"[ACCURACY] Overall Test Accuracy: {results['overall_metrics']['test_accuracy']:.1%}")
        print(f"[MI FOCUS] MI Precision: {results['mi_metrics']['precision']:.1%}")
        print(f"[MI FOCUS] MI Sensitivity: {results['mi_metrics']['recall_sensitivity']:.1%}")
        print(f"[BASELINE] Previous MI Sensitivity: {baseline_mi_sensitivity}%")
        print(f"[ENHANCED] New MI Sensitivity: {new_mi_sensitivity:.1f}%")
        print(f"[IMPROVEMENT] Change: {mi_improvement:+.1f} percentage points")
        print(f"[CLINICAL] Significance: {results['clinical_improvement']['clinical_significance']}")
        
        return results
    
    def save_focused_model(self, filename: str = "enhanced_mi_focused_model") -> Path:
        """Save the focused MI model with all metadata"""
        
        if self.model is None:
            raise ValueError("No model trained yet.")
        
        models_dir = self.data_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{filename}.pkl"
        
        model_package = {
            'model': self.model,
            'performance_results': self.performance_results,
            'training_data_info': self.training_data[3] if self.training_data else None,
            'model_metadata': {
                'focus': 'MI Detection Enhancement',
                'training_approach': 'Focused MI Database Training',
                'clinical_target': 'Improved MI Sensitivity',
                'created_at': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"[SAVED] MI-focused model saved to: {model_path}")
        return model_path
    
    def generate_clinical_summary(self) -> str:
        """Generate clinical presentation summary"""
        
        if not self.performance_results:
            return "No results available. Train model first."
        
        results = self.performance_results
        
        summary = f"""
=== MI DETECTION ENHANCEMENT - CLINICAL SUMMARY ===

[OBJECTIVE] Improve MI detection sensitivity using new large-scale ECG database

[DATASET]
  - Source: Large-Scale 12-Lead ECG Database (45,152 total records)
  - Training Set: {results['dataset_info']['total_records']} records
  - MI Cases: {results['dataset_info']['mi_records']} ({results['dataset_info']['mi_percentage']:.1f}%)
  - Signal Quality: 12-lead ECG, 100Hz, 10-second recordings

[MODEL PERFORMANCE]
  - Algorithm: Random Forest (300 estimators, MI-optimized)
  - Overall Accuracy: {results['overall_metrics']['test_accuracy']:.1%}
  - Test Samples: {results['training_info']['test_samples']}

[MI DETECTION RESULTS]
  - MI Precision: {results['mi_metrics']['precision']:.1%}
  - MI Sensitivity: {results['mi_metrics']['recall_sensitivity']:.1%} [KEY METRIC]
  - MI F1-Score: {results['mi_metrics']['f1_score']:.1%}
  - MI Cases in Test: {results['mi_metrics']['support']}

[CLINICAL IMPROVEMENT]
  - Baseline MI Sensitivity: {results['clinical_improvement']['baseline_mi_sensitivity_pct']}%
  - Enhanced MI Sensitivity: {results['clinical_improvement']['enhanced_mi_sensitivity_pct']:.1f}%
  - Improvement: {results['clinical_improvement']['improvement_points']:+.1f} percentage points
  - Clinical Significance: {results['clinical_improvement']['clinical_significance']}

[RECOMMENDATION]
  {"Ready for clinical validation - significant improvement achieved" if results['clinical_improvement']['improvement_points'] > 5 
   else "Further optimization recommended" if results['clinical_improvement']['improvement_points'] > 0 
   else "Dataset expansion needed for MI detection improvement"}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return summary


def run_focused_mi_training():
    """Run the complete focused MI training pipeline"""
    
    try:
        print("=== FOCUSED MI ENHANCEMENT TRAINING ===")
        
        # Initialize trainer
        trainer = FocusedMITrainer()
        
        # Load MI-focused dataset with aggressive targeting
        print("\n[PHASE 1] Loading MI-focused dataset...")
        signals, labels, ids, metadata = trainer.load_mi_focused_dataset(
            target_mi_records=20,   # Realistic target based on database analysis
            total_records=2000,     # Reasonable total for training
            use_cache=True
        )
        
        # Train the focused model
        print("\n[PHASE 2] Training MI-focused model...")
        results = trainer.train_focused_mi_model(test_size=0.2)
        
        # Save the model
        print("\n[PHASE 3] Saving enhanced model...")
        model_path = trainer.save_focused_model()
        
        # Generate clinical summary
        print("\n[PHASE 4] Generating clinical summary...")
        summary = trainer.generate_clinical_summary()
        print(summary)
        
        # Save summary to file
        summary_path = trainer.data_dir / "results" / "mi_enhancement_clinical_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n[SUCCESS] MI Enhancement Training Complete!")
        print(f"[MODEL] Saved to: {model_path}")
        print(f"[SUMMARY] Saved to: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] MI training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_focused_mi_training()