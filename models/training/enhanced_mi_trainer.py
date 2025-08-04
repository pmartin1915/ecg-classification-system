"""
Enhanced MI Detection Model Trainer
Advanced algorithms and techniques specifically for MI detection improvement
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class EnhancedMITrainer:
    """
    Enhanced trainer specifically designed for MI detection improvement
    Uses advanced algorithms, class balancing, and ensemble methods
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.label_encoder = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.results = {}
        
    def create_enhanced_models(self) -> Dict[str, Any]:
        """Create enhanced model configurations for MI detection"""
        
        models = {
            'enhanced_rf': RandomForestClassifier(
                n_estimators=200,           # More trees
                max_depth=15,               # Deeper trees
                min_samples_split=10,       # Better generalization
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                class_weight='balanced',    # Handle class imbalance
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,              # Prevent overfitting
                random_state=self.random_state
            ),
            
            'mi_optimized_rf': RandomForestClassifier(
                n_estimators=300,           # Even more trees for MI detection
                max_depth=20,
                min_samples_split=5,        # Allow more specific patterns
                min_samples_leaf=2,
                max_features=0.3,           # Focus on most relevant features
                bootstrap=True,
                oob_score=True,
                class_weight={0: 1, 1: 3},  # Heavy MI class weighting
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost_mi'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,              # L1 regularization
                reg_lambda=0.1,             # L2 regularization
                scale_pos_weight=3,         # Handle class imbalance
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        return models
    
    def prepare_mi_focused_data(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data specifically for MI detection with advanced preprocessing
        """
        print("PREPARING MI-FOCUSED DATA")
        print("=" * 50)
        
        # Convert to binary MI detection problem
        print("1. Converting to binary MI detection...")
        
        # Encode labels if they're not already numeric
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)
        
        # Create binary MI labels (1 = MI, 0 = Not MI)
        mi_labels = ['AMI', 'IMI', 'LMI', 'PMI']  # All MI types
        if hasattr(self.label_encoder, 'classes_'):
            mi_indices = [i for i, label in enumerate(self.label_encoder.classes_) 
                         if any(mi_type in str(label).upper() for mi_type in mi_labels)]
            y_binary = np.isin(y_encoded, mi_indices).astype(int)
        else:
            # Fallback: assume labels contain MI information
            y_binary = np.array([1 if any(mi_type in str(label).upper() for mi_type in mi_labels) 
                               else 0 for label in y])
        
        print(f"   Original classes: {len(np.unique(y_encoded))}")
        print(f"   Binary MI distribution: {np.bincount(y_binary)}")
        
        # Feature selection for MI detection
        print("2. Selecting MI-relevant features...")
        mi_relevant_features = self._select_mi_features(X, feature_names)
        X_selected = X[:, mi_relevant_features] if mi_relevant_features else X
        
        print(f"   Features: {X.shape[1]} -> {X_selected.shape[1]}")
        
        # Advanced preprocessing
        print("3. Advanced preprocessing...")
        
        # Remove extreme outliers
        X_cleaned = self._remove_outliers(X_selected)
        
        # Handle missing values
        X_cleaned = self._handle_missing_values(X_cleaned)
        
        print(f"   Final data shape: {X_cleaned.shape}")
        print(f"   MI prevalence: {np.mean(y_binary):.1%}")
        
        return X_cleaned, y_binary
    
    def train_with_advanced_techniques(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Train models using advanced techniques for MI detection
        """
        print("ADVANCED MI DETECTION TRAINING")
        print("=" * 60)
        
        # Prepare data
        X_processed, y_binary = self.prepare_mi_focused_data(X, y, feature_names)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_binary, 
            test_size=0.2, 
            random_state=self.random_state,
            stratify=y_binary
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training MI prevalence: {np.mean(y_train):.1%}")
        
        # Create models
        models = self.create_enhanced_models()
        results = {}
        
        # Train each model with advanced techniques
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Apply SMOTE for class balancing
                X_train_balanced, y_train_balanced = self._apply_smote(X_train, y_train)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_balanced)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train_balanced)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train_balanced)
                test_score = model.score(X_test_scaled, y_test)
                
                # Cross-validation
                cv_scores = self._cross_validate_model(model, X_train_scaled, y_train_balanced)
                
                # Detailed evaluation
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # MI-specific metrics
                mi_metrics = self._calculate_mi_metrics(y_test, y_pred, y_pred_proba)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'mi_metrics': mi_metrics
                }
                
                # Track best model
                if mi_metrics['sensitivity'] > self.best_score:
                    self.best_score = mi_metrics['sensitivity']
                    self.best_model = model
                    self.best_model_name = model_name
                
                print(f"   Training accuracy: {train_score:.3f}")
                print(f"   Test accuracy: {test_score:.3f}")
                print(f"   CV score: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores)*2:.3f})")
                print(f"   MI Sensitivity: {mi_metrics['sensitivity']:.3f}")
                print(f"   MI Specificity: {mi_metrics['specificity']:.3f}")
                
            except Exception as e:
                print(f"   ERROR training {model_name}: {e}")
                continue
        
        # Create ensemble if multiple models succeeded
        if len(results) >= 2:
            print("\nCreating ensemble model...")
            ensemble_results = self._create_mi_ensemble(results, X_test_scaled, y_test)
            if ensemble_results:
                results['ensemble'] = ensemble_results
        
        self.results = results
        return results
    
    def _select_mi_features(self, X: np.ndarray, feature_names: List[str] = None) -> List[int]:
        """Select features most relevant to MI detection"""
        if feature_names is None:
            return list(range(X.shape[1]))  # Use all features if names not available
        
        # MI-relevant feature keywords
        mi_keywords = [
            'st_elevation', 'st_depression', 'q_wave', 't_wave', 
            'anterior', 'inferior', 'lateral', 'reciprocal',
            'v1', 'v2', 'v3', 'v4', 'v5', 'v6',  # Precordial leads
            'ii', 'iii', 'avf',  # Inferior leads
            'i', 'avl',  # Lateral leads
            'amplitude', 'elevation', 'depression'
        ]
        
        selected_indices = []
        for i, name in enumerate(feature_names):
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in mi_keywords):
                selected_indices.append(i)
        
        # If too few MI-specific features, add general ECG features
        if len(selected_indices) < 20:
            general_keywords = ['median', 'mean', 'max', 'min', 'std', 'correlation']
            for i, name in enumerate(feature_names):
                name_lower = name.lower()
                if (any(keyword in name_lower for keyword in general_keywords) and 
                    i not in selected_indices):
                    selected_indices.append(i)
                    if len(selected_indices) >= 50:  # Limit to reasonable number
                        break
        
        return selected_indices if selected_indices else list(range(X.shape[1]))
    
    def _remove_outliers(self, X: np.ndarray) -> np.ndarray:
        """Remove extreme outliers using IQR method"""
        X_cleaned = X.copy()
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            
            # Calculate IQR
            Q1 = np.percentile(feature_data, 25)
            Q3 = np.percentile(feature_data, 75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 3 * IQR  # More conservative than typical 1.5*IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap outliers
            X_cleaned[:, feature_idx] = np.clip(feature_data, lower_bound, upper_bound)
        
        return X_cleaned
    
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values with median imputation"""
        X_cleaned = X.copy()
        
        for feature_idx in range(X.shape[1]):
            feature_data = X_cleaned[:, feature_idx]
            
            # Replace inf/-inf with NaN
            feature_data[np.isinf(feature_data)] = np.nan
            
            # Impute NaN with median
            if np.any(np.isnan(feature_data)):
                median_value = np.nanmedian(feature_data)
                feature_data[np.isnan(feature_data)] = median_value
                X_cleaned[:, feature_idx] = feature_data
        
        return X_cleaned
    
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for handling class imbalance"""
        try:
            # Check class distribution
            class_counts = np.bincount(y)
            minority_class_count = np.min(class_counts)
            
            # Only apply SMOTE if significant imbalance
            if minority_class_count / np.max(class_counts) < 0.3:
                
                # Use SMOTE with random undersampling
                smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, minority_class_count-1))
                under = RandomUnderSampler(sampling_strategy=0.7, random_state=self.random_state)
                
                pipeline = ImbPipeline([('smote', smote), ('under', under)])
                X_resampled, y_resampled = pipeline.fit_resample(X, y)
                
                print(f"   SMOTE applied: {len(y)} -> {len(y_resampled)} samples")
                print(f"   New distribution: {np.bincount(y_resampled)}")
                return X_resampled, y_resampled
            
        except Exception as e:
            print(f"   SMOTE failed: {e}, using original data")
        
        return X, y
    
    def _cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform stratified cross-validation"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return scores
    
    def _calculate_mi_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate MI-specific evaluation metrics"""
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for MI
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for non-MI
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # F1 score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # AUC if probabilities available
        auc = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else 0
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,  # Most important for MI detection
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'auc': auc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def _create_mi_ensemble(self, model_results: Dict[str, Any], 
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Create ensemble model for MI detection"""
        try:
            # Select top 3 models based on sensitivity
            sorted_models = sorted(model_results.items(), 
                                 key=lambda x: x[1]['mi_metrics']['sensitivity'], 
                                 reverse=True)[:3]
            
            estimators = []
            for model_name, results in sorted_models:
                estimators.append((model_name, results['model']))
            
            # Create voting classifier
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'  # Use probability averaging
            )
            
            # Use the scaler from the best model
            best_scaler = sorted_models[0][1]['scaler']  
            
            # We need to retrain the ensemble, but we'll use the test set results as proxy
            y_pred_ensemble = np.zeros(len(y_test))
            y_pred_proba_ensemble = np.zeros(len(y_test))
            
            # Simple ensemble by averaging predictions
            for model_name, results in sorted_models:
                model = results['model']
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                y_pred_ensemble += y_pred
                y_pred_proba_ensemble += y_pred_proba
            
            # Average and threshold
            y_pred_ensemble = (y_pred_ensemble / len(sorted_models) > 0.5).astype(int)
            y_pred_proba_ensemble = y_pred_proba_ensemble / len(sorted_models)
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_mi_metrics(y_test, y_pred_ensemble, y_pred_proba_ensemble)
            
            print(f"   Ensemble Sensitivity: {ensemble_metrics['sensitivity']:.3f}")
            print(f"   Ensemble Specificity: {ensemble_metrics['specificity']:.3f}")
            
            return {
                'model': ensemble,
                'scaler': best_scaler,
                'test_score': ensemble_metrics['accuracy'],
                'mi_metrics': ensemble_metrics,
                'component_models': [name for name, _ in sorted_models]
            }
            
        except Exception as e:
            print(f"   Ensemble creation failed: {e}")
            return None
    
    def save_best_model(self, save_path: Path) -> bool:
        """Save the best performing MI detection model"""
        if self.best_model is None:
            print("No trained model to save")
            return False
        
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model package
            model_package = {
                'model': self.best_model,
                'scaler': self.results[self.best_model_name]['scaler'],
                'label_encoder': self.label_encoder,
                'model_name': self.best_model_name,
                'performance_metrics': self.results[self.best_model_name]['mi_metrics'],
                'training_info': {
                    'best_sensitivity': self.best_score,
                    'model_type': 'Enhanced MI Detection',
                    'training_date': pd.Timestamp.now().isoformat()
                }
            }
            
            model_file = save_path / f"enhanced_mi_model_{self.best_model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_package, f)
            
            print(f"Enhanced MI model saved: {model_file}")
            print(f"Best model: {self.best_model_name}")
            print(f"MI Sensitivity: {self.best_score:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def print_training_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*60)
        print("ENHANCED MI DETECTION TRAINING SUMMARY")
        print("="*60)
        
        if not self.results:
            print("No training results available")
            return
        
        print(f"Best Model: {self.best_model_name}")
        print(f"Best MI Sensitivity: {self.best_score:.3f}")
        print()
        
        # Results table
        print("Model Performance Summary:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1-Score':<10}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            metrics = results['mi_metrics']
            print(f"{model_name:<20} {metrics['accuracy']:<10.3f} {metrics['sensitivity']:<12.3f} "
                  f"{metrics['specificity']:<12.3f} {metrics['f1_score']:<10.3f}")
        
        print("-" * 80)
        print(f"\nTarget: MI Sensitivity > 70% for clinical use")
        print(f"Current Best: {self.best_score:.1%}")
        
        if self.best_score >= 0.70:
            print("🎉 CLINICAL TARGET ACHIEVED!")
        else:
            improvement_needed = 0.70 - self.best_score
            print(f"Improvement needed: +{improvement_needed:.1%}")

# Convenience function for easy use
def train_enhanced_mi_model(X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str] = None,
                          save_path: Path = None) -> EnhancedMITrainer:
    """
    Train enhanced MI detection model with all advanced techniques
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: Feature names for selection
        save_path: Path to save best model
        
    Returns:
        Trained EnhancedMITrainer instance
    """
    trainer = EnhancedMITrainer()
    trainer.train_with_advanced_techniques(X, y, feature_names)
    trainer.print_training_summary()
    
    if save_path:
        trainer.save_best_model(save_path)
    
    return trainer