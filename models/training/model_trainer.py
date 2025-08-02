"""
Main model trainer for ECG classification
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Additional models (optional imports)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from config.model_config import ModelTrainingConfig, MODEL_CONFIGS
from models.training.base_trainer import BaseTrainer


class ModelTrainer(BaseTrainer):
    """Main model trainer with support for multiple algorithms"""
    
    def __init__(self, config: ModelTrainingConfig):
        super().__init__(config)
        self.model_configs = self._initialize_model_configs()
        
    def _initialize_model_configs(self) -> Dict[str, Dict]:
        """Initialize available model configurations"""
        available_models = {}
        
        # Standard sklearn models
        model_classes = {
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression,
            'SVC': SVC,
            'KNeighborsClassifier': KNeighborsClassifier,
            'MLPClassifier': MLPClassifier,
            'AdaBoostClassifier': AdaBoostClassifier
        }
        
        # Optional models
        if HAS_XGBOOST:
            model_classes['XGBClassifier'] = xgb.XGBClassifier
        if HAS_LIGHTGBM:
            model_classes['LGBMClassifier'] = lgb.LGBMClassifier
        
        # Build available models from config
        for model_key, model_config in MODEL_CONFIGS.items():
            if model_config.model_class in model_classes:
                available_models[model_key] = {
                    'config': model_config,
                    'class': model_classes[model_config.model_class]
                }
        
        return available_models
    
    def setup_models(self, model_keys: Optional[List[str]] = None) -> None:
        """Setup models for training"""
        if model_keys is None:
            model_keys = list(self.model_configs.keys())
        
        self.logger.info(f"Setting up {len(model_keys)} models")
        
        for model_key in model_keys:
            if model_key not in self.model_configs:
                self.logger.warning(f"Model {model_key} not available, skipping")
                continue
            
            model_info = self.model_configs[model_key]
            model_config = model_info['config']
            model_class = model_info['class']
            
            # Create model instance with default parameters
            params = model_config.default_params.copy()
            
            # Add random state if applicable
            if 'random_state' in model_class.__init__.__code__.co_varnames:
                params['random_state'] = self.config.random_state
            
            # Handle class weights
            if self.config.use_class_weights and hasattr(self, 'class_weight_dict'):
                if 'class_weight' in model_class.__init__.__code__.co_varnames:
                    params['class_weight'] = self.class_weight_dict
            
            # Create model
            model = model_class(**params)
            
            self.models[model_config.name] = {
                'model': model,
                'config': model_config,
                'param_grid': model_config.param_grid
            }
        
        self.logger.info(f"Models setup complete:")
        for model_name in self.models.keys():
            self.logger.info(f"  - {model_name}")
    
    def train_model(self, model_name: str, use_smote: bool = None,
                   use_hyperparameter_tuning: bool = None) -> Dict[str, Any]:
        """Train a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in configured models")
        
        use_smote = use_smote if use_smote is not None else self.config.use_smote
        use_hp_tuning = (use_hyperparameter_tuning if use_hyperparameter_tuning is not None 
                        else self.config.use_hyperparameter_tuning)
        
        self.logger.info(f"Training {model_name}...")
        
        model_info = self.models[model_name]
        model = model_info['model']
        model_config = model_info['config']
        
        # Get training data
        X_train = self.X_train_scaled if model_config.requires_scaling else self.X_train
        X_val = self.X_val_scaled if model_config.requires_scaling else self.X_val
        
        # Apply SMOTE if requested
        X_train_final = X_train.copy()
        y_train_final = self.y_train.copy()
        
        if use_smote:
            self.logger.info(f"  Applying SMOTE...")
            smote = SMOTE(random_state=self.config.random_state)
            X_train_final, y_train_final = smote.fit_resample(X_train_final, y_train_final)
            self.logger.info(f"  Training samples after SMOTE: {len(X_train_final):,}")
        
        # Hyperparameter tuning
        if use_hp_tuning and model_info['param_grid']:
            model = self._tune_hyperparameters(
                model, model_info['param_grid'], 
                X_train_final, y_train_final
            )
        
        # Train model
        start_time = time.time()
        model.fit(X_train_final, y_train_final)
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        y_val_proba = None
        if model_config.supports_probability:
            try:
                y_val_proba = model.predict_proba(X_val)
            except:
                pass
        
        # Calculate metrics
        metrics = self.evaluate_model(self.y_val, y_val_pred, y_val_proba)
        
        # Get feature importance if available
        feature_importance = self.get_feature_importance(model, model_name)
        
        # Store results
        results = {
            'model': model,
            'metrics': metrics,
            'training_time': training_time,
            'used_smote': use_smote,
            'used_hp_tuning': use_hp_tuning,
            'predictions': y_val_pred,
            'probabilities': y_val_proba,
            'feature_importance': feature_importance
        }
        
        self.results[model_name] = results
        
        self.logger.info(f"  Training complete - F1: {metrics['f1_weighted']:.3f}, "
                        f"Accuracy: {metrics['accuracy']:.3f}")
        
        return results
    
    def _tune_hyperparameters(self, model: Any, param_grid: Dict,
                            X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Perform hyperparameter tuning"""
        self.logger.info("  Performing hyperparameter tuning...")
        
        if self.config.tuning_method == 'grid':
            search = GridSearchCV(
                model, param_grid,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
                n_jobs=self.config.n_jobs,
                verbose=0
            )
        elif self.config.tuning_method == 'random':
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=self.config.n_iter_random,
                cv=self.config.cv_folds,
                scoring=self.config.scoring_metric,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown tuning method: {self.config.tuning_method}")
        
        search.fit(X_train, y_train)
        
        self.logger.info(f"  Best parameters: {search.best_params_}")
        self.logger.info(f"  Best CV score: {search.best_score_:.3f}")
        
        return search.best_estimator_
    
    def train_all_models(self, model_keys: Optional[List[str]] = None,
                        use_smote: bool = None) -> None:
        """Train all configured models"""
        if model_keys is None:
            model_keys = list(self.models.keys())
        
        self.logger.info(f"Training {len(model_keys)} models...")
        
        for model_name in tqdm(model_keys, desc="Training models"):
            try:
                self.train_model(model_name, use_smote=use_smote)
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """Evaluate and compare all trained models"""
        self.logger.info("Evaluating all models...")
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            
            row_data = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision_weighted'],
                'Recall': metrics['recall_weighted'],
                'F1-Score': metrics['f1_weighted'],
                'Cohen Kappa': metrics['cohen_kappa'],
                'Matthews Corr': metrics['matthews_corrcoef'],
                'Training Time': results['training_time'],
                'SMOTE': results['used_smote']
            }
            
            # Add ROC AUC if available
            if 'roc_auc_ovr' in metrics:
                row_data['ROC AUC'] = metrics['roc_auc_ovr']
            
            comparison_data.append(row_data)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values('F1-Score', ascending=False)
        
        # Find best model
        self.best_model_name = self.comparison_df.iloc[0]['Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        self.logger.info("\nModel Comparison:")
        self.logger.info(self.comparison_df.to_string(index=False, float_format='%.3f'))
        self.logger.info(f"\nBest model: {self.best_model_name}")
        
        return self.comparison_df
    
    def cross_validate_model(self, model_name: str, cv_folds: int = None) -> Dict[str, Any]:
        """Perform cross-validation for a model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        cv_folds = cv_folds or self.config.cv_folds
        
        self.logger.info(f"Cross-validating {model_name} with {cv_folds} folds...")
        
        model = self.models[model_name]['model']
        model_config = self.models[model_name]['config']
        
        # Get appropriate data
        X = self.X_train_scaled if model_config.requires_scaling else self.X_train
        y = self.y_train
        
        # Perform cross-validation
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                model, X, y,
                cv=cv_folds,
                scoring=metric,
                n_jobs=self.config.n_jobs
            )
            
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            
            self.logger.info(f"  {metric}: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        return cv_results
    
    def test_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Test model on test set"""
        model_name = model_name or self.best_model_name
        
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not trained yet")
        
        self.logger.info(f"Testing {model_name} on test set...")
        
        model = self.results[model_name]['model']
        model_config = self.models[model_name]['config']
        
        # Get test data
        X_test = self.X_test_scaled if model_config.requires_scaling else self.X_test
        
        # Predict
        y_test_pred = model.predict(X_test)
        y_test_proba = None
        if model_config.supports_probability:
            try:
                y_test_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Evaluate
        test_metrics = self.evaluate_model(self.y_test, y_test_pred, y_test_proba)
        
        # Store test results
        test_results = {
            'model_name': model_name,
            'metrics': test_metrics,
            'predictions': y_test_pred,
            'probabilities': y_test_proba
        }
        
        self.test_results = test_results
        
        # Log results
        self.logger.info(f"Test Results for {model_name}:")
        self.logger.info(f"  Accuracy: {test_metrics['accuracy']:.3f}")
        self.logger.info(f"  Precision: {test_metrics['precision_weighted']:.3f}")
        self.logger.info(f"  Recall: {test_metrics['recall_weighted']:.3f}")
        self.logger.info(f"  F1-Score: {test_metrics['f1_weighted']:.3f}")
        
        # Check clinical thresholds
        self._check_clinical_thresholds(test_metrics)
        
        return test_results
    
    def _check_clinical_thresholds(self, metrics: Dict[str, float]) -> None:
        """Check if model meets clinical thresholds"""
        self.logger.info("\nClinical Threshold Check:")
        
        # Check MI sensitivity
        mi_sensitivity = metrics.get('MI_sensitivity', 0)
        mi_threshold = self.config.clinical_thresholds.get('mi_sensitivity', 0.9)
        status = "✓" if mi_sensitivity >= mi_threshold else "✗"
        self.logger.info(f"  MI Sensitivity: {mi_sensitivity:.3f} {status} (threshold: {mi_threshold})")
        
        # Check NORM specificity
        norm_specificity = metrics.get('NORM_specificity', 0)
        norm_threshold = self.config.clinical_thresholds.get('norm_specificity', 0.95)
        status = "✓" if norm_specificity >= norm_threshold else "✗"
        self.logger.info(f"  NORM Specificity: {norm_specificity:.3f} {status} (threshold: {norm_threshold})")
        
        # Check overall accuracy
        accuracy = metrics.get('accuracy', 0)
        acc_threshold = self.config.clinical_thresholds.get('overall_accuracy', 0.85)
        status = "✓" if accuracy >= acc_threshold else "✗"
        self.logger.info(f"  Overall Accuracy: {accuracy:.3f} {status} (threshold: {acc_threshold})")
    
    def create_ensemble(self, base_models: List[str], 
                       voting: str = 'soft',
                       weights: Optional[List[float]] = None) -> None:
        """Create ensemble model from trained models"""
        self.logger.info(f"Creating {voting} voting ensemble from {len(base_models)} models")
        
        # Get trained models
        estimators = []
        for model_name in base_models:
            if model_name not in self.results:
                self.logger.warning(f"Model {model_name} not trained, skipping")
                continue
            
            model = self.results[model_name]['model']
            estimators.append((model_name, model))
        
        if len(estimators) < 2:
            raise ValueError("Need at least 2 models for ensemble")
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=self.config.n_jobs
        )
        
        # Add to models
        ensemble_name = f"Ensemble_{voting}"
        self.models[ensemble_name] = {
            'model': ensemble,
            'config': type('obj', (object,), {
                'name': ensemble_name,
                'requires_scaling': True,
                'supports_probability': voting == 'soft',
                'supports_feature_importance': False
            }),
            'param_grid': {}
        }
        
        # Train ensemble
        self.train_model(ensemble_name, use_smote=False, use_hyperparameter_tuning=False)
        
        self.logger.info(f"Ensemble created and trained successfully")