"""
Base trainer class for model training
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, cohen_kappa_score, matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight
import joblib
import time
from datetime import datetime
import logging

from config.model_config import ModelTrainingConfig


class BaseTrainer:
    """Base class for model training"""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # Data attributes
        self.X = None
        self.y = None
        self.feature_names = None
        
        # Split data
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Scaled data
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        
        # Preprocessing
        self.scaler = None
        self.class_weights = None
        
        # Results storage
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Set random seed
        np.random.seed(self.config.random_state)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def load_data(self, X: np.ndarray, y: np.ndarray, 
                  feature_names: Optional[List[str]] = None) -> None:
        """Load data into trainer"""
        self.logger.info("Loading data into trainer")
        
        self.X = X
        self.y = y
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Validate data
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same number of samples")
        
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("Number of feature names must match number of features")
        
        # Log data info
        self.logger.info(f"Data loaded successfully:")
        self.logger.info(f"  - Shape: {self.X.shape}")
        self.logger.info(f"  - Features: {len(self.feature_names)}")
        self.logger.info(f"  - Classes: {len(np.unique(self.y))}")
        
        # Class distribution
        class_counts = pd.Series(self.y).value_counts().sort_index()
        self.logger.info("Class distribution:")
        for class_idx, count in class_counts.items():
            percentage = (count / len(self.y)) * 100
            class_name = self.config.class_names.get(class_idx, f'Class_{class_idx}')
            self.logger.info(f"  - {class_name}: {count:,} ({percentage:.1f}%)")
    
    def prepare_data_splits(self) -> None:
        """Create train/validation/test splits"""
        self.logger.info("Creating data splits")
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.config.test_size,
            stratify=self.y if self.config.stratify else None,
            random_state=self.config.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp if self.config.stratify else None,
            random_state=self.config.random_state
        )
        
        # Log split info
        self.logger.info(f"Data splits created:")
        self.logger.info(f"  - Train: {len(self.X_train):,} ({len(self.X_train)/len(self.X)*100:.1f}%)")
        self.logger.info(f"  - Val: {len(self.X_val):,} ({len(self.X_val)/len(self.X)*100:.1f}%)")
        self.logger.info(f"  - Test: {len(self.X_test):,} ({len(self.X_test)/len(self.X)*100:.1f}%)")
        
        # Calculate class weights
        if self.config.use_class_weights:
            self._calculate_class_weights()
    
    def _calculate_class_weights(self) -> None:
        """Calculate class weights for imbalanced data"""
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        
        self.class_weight_dict = dict(zip(np.unique(self.y_train), self.class_weights))
        
        self.logger.info("Class weights calculated:")
        for class_idx, weight in self.class_weight_dict.items():
            class_name = self.config.class_names.get(class_idx, f'Class_{class_idx}')
            self.logger.info(f"  - {class_name}: {weight:.3f}")
    
    def scale_features(self, scaler_type: str = 'standard') -> None:
        """Scale features"""
        self.logger.info(f"Scaling features using {scaler_type} scaler")
        
        # Select scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit on training data only
        self.scaler.fit(self.X_train)
        
        # Transform all splits
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.logger.info("Features scaled successfully")
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate model performance"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        metrics['precision_weighted'] = precision
        metrics['recall_weighted'] = recall
        metrics['f1_weighted'] = f1
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        
        # Additional metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                )
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['class_report'] = class_report
        
        # Clinical metrics if enabled
        if self.config.clinical_metrics:
            clinical_metrics = self._calculate_clinical_metrics(y_true, y_pred)
            metrics.update(clinical_metrics)
        
        return metrics
    
    def _calculate_clinical_metrics(self, y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate clinical performance metrics"""
        metrics = {}
        cm = confusion_matrix(y_true, y_pred)
        
        # Get unique classes present in the data
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        for class_idx in range(len(self.config.class_names)):
            class_name = self.config.class_names.get(class_idx, f'Class_{class_idx}')
            
            # Check if this class is present in the test data
            if class_idx not in unique_classes or class_idx >= cm.shape[0] or class_idx >= cm.shape[1]:
                # Class not present in test data, set metrics to 0
                metrics[f'{class_name}_sensitivity'] = 0.0
                metrics[f'{class_name}_specificity'] = 0.0
                
                if self.config.calculate_ppv_npv:
                    metrics[f'{class_name}_ppv'] = 0.0
                    metrics[f'{class_name}_npv'] = 0.0
                
                self.logger.warning(f"Class {class_name} (index {class_idx}) not present in test data")
                continue
            
            # True/False positives/negatives
            tp = cm[class_idx, class_idx]
            fp = cm[:, class_idx].sum() - tp
            fn = cm[class_idx, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            # Sensitivity (Recall)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f'{class_name}_sensitivity'] = sensitivity
            
            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'{class_name}_specificity'] = specificity
            
            if self.config.calculate_ppv_npv:
                # Positive Predictive Value (Precision)
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                metrics[f'{class_name}_ppv'] = ppv
                
                # Negative Predictive Value
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                metrics[f'{class_name}_npv'] = npv
        
        return metrics
    
    def save_model(self, model: Any, model_name: str, save_dir: Path) -> Path:
        """Save a trained model"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean model name for filename
        clean_name = model_name.lower().replace(' ', '_')
        model_path = save_dir / f'model_{clean_name}.pkl'
        
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved: {model_path}")
        
        return model_path
    
    def save_results(self, save_dir: Path) -> None:
        """Save training results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = save_dir / 'scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Scaler saved: {scaler_path}")
        
        # Save results dictionary
        results_data = {
            'results': self.results,
            'best_model_name': self.best_model_name,
            'config': self.config,
            'feature_names': self.feature_names,
            'class_weights': self.class_weight_dict if hasattr(self, 'class_weight_dict') else None,
            'training_date': datetime.now().isoformat()
        }
        
        results_path = save_dir / 'training_results.pkl'
        joblib.dump(results_data, results_path)
        self.logger.info(f"Results saved: {results_path}")
    
    def get_feature_importance(self, model: Any, model_name: str) -> Optional[np.ndarray]:
        """Extract feature importance from model if available"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficient values
            return np.abs(model.coef_).mean(axis=0)
        else:
            return None