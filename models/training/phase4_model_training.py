"""
Phase 4: Model Training Pipeline
================================
This module implements a comprehensive model training pipeline for ECG classification.
Includes support for multiple models, hyperparameter tuning, and large dataset handling.

Author: ECG AI Project
Date: 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import gc
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Optional advanced models (install separately if needed)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

warnings.filterwarnings('ignore')


class MemoryOptimizer:
    """Handle memory optimization for large datasets"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        return df
    
    @staticmethod
    def clear_memory():
        """Force garbage collection"""
        gc.collect()


class DataHandler:
    """Handle data loading and preprocessing for large datasets"""
    
    def __init__(self, chunk_size: int = 5000):
        self.chunk_size = chunk_size
        self.memory_optimizer = MemoryOptimizer()
    
    def load_data_in_chunks(self, data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load large dataset in chunks to manage memory"""
        print(f"Loading data from {data_path} in chunks of {self.chunk_size}...")
        
        # Determine file type and load accordingly
        if data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            return self._extract_features_labels(data)
        
        elif data_path.suffix == '.csv':
            # Read CSV in chunks
            chunks = []
            for chunk in pd.read_csv(data_path, chunksize=self.chunk_size):
                # Optimize memory usage
                chunk = self.memory_optimizer.optimize_dtypes(chunk)
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            X = df.drop('label', axis=1).values
            y = df['label'].values
            return X, y
        
        else:
            raise ValueError(f"Unsupported file type: {data_path.suffix}")
    
    def _extract_features_labels(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from loaded data"""
        if 'X_features' in data:
            X = data['X_features']
            y = data['y_encoded']
        elif 'X' in data:
            X = data['X']
            y = data['y']
        else:
            X = data.get('features')
            y = data.get('labels')
        
        return X, y
    
    def prepare_batches(self, X: np.ndarray, y: np.ndarray, 
                       batch_size: int = 1000) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Prepare data in batches for processing"""
        n_samples = len(X)
        batches = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batches.append((X[i:end_idx], y[i:end_idx]))
        
        return batches


class ModelTrainer:
    """Enhanced model trainer with support for large datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = self._initialize_models()
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.memory_optimizer = MemoryOptimizer()
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all available models"""
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'decision_tree': DecisionTreeClassifier(
                class_weight='balanced',
                random_state=42
            ),
            'naive_bayes': GaussianNB(),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGB_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1
            )
        
        # Add LightGBM if available
        if LGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        return models
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, 
                    val_size: float = 0.1) -> Tuple:
        """Prepare train/val/test splits with stratification"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            stratify=y_temp, random_state=42
        )
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_imbalanced_data(self, X: np.ndarray, y: np.ndarray, 
                              method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance using various techniques"""
        print(f"Applying {method.upper()} for class balancing...")
        
        # Check if we have enough samples for SMOTE
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_samples = min(class_counts)
        
        if min_samples < 6 and method in ['smote', 'adasyn', 'smotetomek']:
            print(f"Warning: Minimum class has only {min_samples} samples. Skipping {method.upper()} and using class weights instead.")
            return X, y
        
        try:
            if method == 'smote':
                sampler = SMOTE(random_state=42)
            elif method == 'adasyn':
                sampler = ADASYN(random_state=42)
            elif method == 'smotetomek':
                sampler = SMOTETomek(random_state=42)
            elif method == 'undersample':
                sampler = RandomUnderSampler(random_state=42)
            else:
                return X, y
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            print(f"Resampled shape: {X_resampled.shape}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"Warning: {method.upper()} failed ({str(e)}). Using original data with class weights.")
            return X, y
    
    def get_hyperparameter_grid(self, model_name: str) -> Dict:
        """Get hyperparameter grid for each model"""
        grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        
        if XGB_AVAILABLE and model_name == 'xgboost':
            grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
        
        if LGBM_AVAILABLE and model_name == 'lightgbm':
            grids['lightgbm'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 100]
            }
        
        return grids.get(model_name, {})
    
    def train_model(self, model_name: str, model: Any,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   use_hyperparameter_tuning: bool = False) -> Dict:
        """Train a single model with optional hyperparameter tuning"""
        print(f"\nTraining {model_name}...")
        start_time = datetime.now()
        
        try:
            if use_hyperparameter_tuning and model_name not in ['naive_bayes']:
                param_grid = self.get_hyperparameter_grid(model_name)
                if param_grid:
                    # Use RandomizedSearchCV for faster results
                    model = RandomizedSearchCV(
                        model, param_grid, 
                        n_iter=20,  # Limit iterations
                        cv=3,  # 3-fold CV
                        scoring='f1_weighted',
                        n_jobs=-1,
                        random_state=42
                    )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get best model if using hyperparameter tuning
            if hasattr(model, 'best_estimator_'):
                best_model = model.best_estimator_
                best_params = model.best_params_
            else:
                best_model = model
                best_params = {}
            
            # Predictions
            y_val_pred = best_model.predict(X_val)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_val_pred)
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                feature_importance = np.abs(best_model.coef_).mean(axis=0)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Clear memory
            self.memory_optimizer.clear_memory()
            
            return {
                'model': best_model,
                'metrics': metrics,
                'best_params': best_params,
                'feature_importance': feature_importance,
                'training_time': training_time,
                'predictions': y_val_pred
            }
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            return None
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Get unique classes present in the data
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        # Per-class metrics - only for classes that exist in the data
        class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        for i in unique_classes:
            if i < len(class_names):  # Make sure index is valid
                class_name = class_names[i]
                class_mask = y_true == i
                if class_mask.sum() > 0:
                    class_pred = y_pred == i
                    metrics[f'{class_name}_precision'] = precision_score(class_mask, class_pred, zero_division=0)
                    metrics[f'{class_name}_recall'] = recall_score(class_mask, class_pred, zero_division=0)
                    metrics[f'{class_name}_f1'] = f1_score(class_mask, class_pred, zero_division=0)
        
        # Clinical metrics - only if classes exist
        # MI sensitivity (ability to detect MI)
        if 1 in unique_classes:  # MI is class 1
            mi_mask = y_true == 1
            if mi_mask.sum() > 0:
                metrics['MI_sensitivity'] = recall_score(mi_mask, y_pred == 1, zero_division=0)
                metrics['MI_specificity'] = recall_score(~mi_mask, y_pred != 1, zero_division=0)
        
        # NORM specificity (ability to correctly identify normal ECGs)
        if 0 in unique_classes:  # NORM is class 0
            norm_mask = y_true == 0
            if norm_mask.sum() > 0:
                metrics['NORM_specificity'] = recall_score(norm_mask, y_pred == 0, zero_division=0)
        
        return metrics
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        model_keys: Optional[List[str]] = None,
                        use_smote: bool = True,
                        use_hyperparameter_tuning: bool = False) -> Dict:
        """Train all specified models"""
        if model_keys is None:
            model_keys = list(self.models.keys())
        
        # Apply SMOTE if requested
        if use_smote:
            X_train_balanced, y_train_balanced = self.handle_imbalanced_data(
                X_train, y_train, method='smote'
            )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train each model
        for model_name in model_keys:
            if model_name not in self.models:
                print(f"Model {model_name} not available, skipping...")
                continue
            
            model = self.models[model_name]
            result = self.train_model(
                model_name, model,
                X_train_balanced, y_train_balanced,
                X_val, y_val,
                use_hyperparameter_tuning
            )
            
            if result:
                self.results[model_name] = result
                print(f"OK: {model_name} - Accuracy: {result['metrics']['accuracy']:.3f}, "
                      f"F1: {result['metrics']['f1_weighted']:.3f}")
        
        # Find best model
        self._select_best_model()
        
        return self.results
    
    def _select_best_model(self):
        """Select best model based on F1 score"""
        best_f1 = 0
        for model_name, result in self.results.items():
            f1 = result['metrics']['f1_weighted']
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = result['model']
                self.best_model_name = model_name
        
        print(f"\nBest model: {self.best_model_name} (F1: {best_f1:.3f})")
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate best model on test set"""
        if self.best_model is None:
            raise ValueError("No best model selected. Train models first.")
        
        y_pred = self.best_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get classification report with proper label handling
        class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        
        # Get unique classes present in test set
        unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
        
        # Create labels parameter to ensure all classes are included
        all_labels = list(range(len(class_names)))  # [0, 1, 2, 3, 4]
        
        report = classification_report(y_test, y_pred, 
                                     labels=all_labels,
                                     target_names=class_names,
                                     output_dict=True,
                                     zero_division=0)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'unique_test_classes': unique_test_classes
        }
    
    def save_model(self, model_path: Path):
        """Save the best model and associated data"""
        if self.best_model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_path / f'{self.best_model_name}_model.pkl'
        joblib.dump(self.best_model, model_file)
        
        # Save scaler
        scaler_file = model_path / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_file)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name]['metrics'],
            'feature_importance': self.results[self.best_model_name]['feature_importance'],
            'training_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        metadata_file = model_path / 'model_metadata.pkl'
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {model_path}")
        
        return model_file


class Phase4Pipeline:
    """Main pipeline for Phase 4 model training"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize pipeline with configuration"""
        self.config = config or self._get_default_config()
        self.data_handler = DataHandler()
        self.trainer = None
        self.results = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'test_size': 0.2,
            'val_size': 0.1,
            'use_smote': True,
            'smote_method': 'smote',
            'use_hyperparameter_tuning': False,
            'model_keys': ['logistic_regression', 'random_forest', 
                          'gradient_boosting', 'svm', 'neural_network'],
            'create_visualizations': True,
            'save_models': True,
            'output_dir': Path('models/trained_models')
        }
    
    def run(self, data_path: Optional[Path] = None,
            X: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> Dict:
        """Run the complete Phase 4 pipeline"""
        print("=" * 60)
        print("PHASE 4: MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        if data_path:
            X, y = self.data_handler.load_data_in_chunks(data_path)
        elif X is None or y is None:
            raise ValueError("Either data_path or X and y must be provided")
        
        print(f"\nData shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Step 2: Initialize trainer
        self.trainer = ModelTrainer(self.config)
        
        # Step 3: Prepare data splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_data(
            X, y, 
            test_size=self.config['test_size'],
            val_size=self.config['val_size']
        )
        
        # Step 4: Train models
        print("\n" + "="*40)
        print("Training models...")
        print("="*40)
        
        self.trainer.train_all_models(
            X_train, y_train, X_val, y_val,
            model_keys=self.config['model_keys'],
            use_smote=self.config['use_smote'],
            use_hyperparameter_tuning=self.config['use_hyperparameter_tuning']
        )
        
        # Step 5: Evaluate on test set
        print("\n" + "="*40)
        print("Evaluating best model on test set...")
        print("="*40)
        
        test_results = self.trainer.evaluate_on_test_set(X_test, y_test)
        
        print(f"\nTest Set Performance:")
        print(f"Accuracy: {test_results['metrics']['accuracy']:.3f}")
        print(f"F1-Score: {test_results['metrics']['f1_weighted']:.3f}")
        print(f"MI Sensitivity: {test_results['metrics'].get('MI_sensitivity', 0):.3f}")
        print(f"NORM Specificity: {test_results['metrics'].get('NORM_specificity', 0):.3f}")
        
        # Step 6: Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.trainer.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['metrics']['accuracy'],
                'F1-Score': result['metrics']['f1_weighted'],
                'Precision': result['metrics']['precision_weighted'],
                'Recall': result['metrics']['recall_weighted'],
                'Training Time (s)': result['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*40)
        print("Model Comparison:")
        print("="*40)
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Step 7: Save models if requested
        if self.config['save_models']:
            output_dir = self.config['output_dir']
            self.trainer.save_model(output_dir)
        
        # Step 8: Prepare results
        self.results = {
            'trainer': self.trainer,
            'test_results': test_results,
            'comparison_df': comparison_df,
            'best_model': self.trainer.best_model,
            'best_model_name': self.trainer.best_model_name,
            'feature_names': feature_names,
            'config': self.config
        }
        
        # Step 9: Create visualizations if requested
        if self.config['create_visualizations']:
            self._create_visualizations(test_results, feature_names)
        
        print("\nSUCCESS: Phase 4 completed successfully!")
        
        return self.results
    
    def _create_visualizations(self, test_results: Dict, 
                         feature_names: Optional[List[str]] = None):
        """Create and save visualizations"""
        try:
            # For now, we'll create basic visualizations inline
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create output directory for visualizations
            viz_dir = Path('visualizations/phase4')
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Confusion Matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(test_results['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Blues',
                       xticklabels=['NORM', 'MI', 'STTC', 'CD', 'HYP'],
                       yticklabels=['NORM', 'MI', 'STTC', 'CD', 'HYP'])
            plt.title(f'Confusion Matrix - {self.trainer.best_model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(viz_dir / 'confusion_matrix.png')
            plt.close()
            
            # 2. Feature Importance (if available)
            if feature_names and self.trainer.best_model_name in self.trainer.results:
                importance = self.trainer.results[self.trainer.best_model_name]['feature_importance']
                if importance is not None:
                    # Get top 20 features
                    indices = np.argsort(importance)[-20:]
                    plt.figure(figsize=(10, 8))
                    plt.barh(range(len(indices)), importance[indices])
                    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                    plt.xlabel('Importance')
                    plt.title(f'Top 20 Feature Importances - {self.trainer.best_model_name}')
                    plt.tight_layout()
                    plt.savefig(viz_dir / 'feature_importance.png')
                    plt.close()
            
            print(f"✅ Visualizations saved to {viz_dir}")
            
        except Exception as e:
            print(f"Could not create visualizations: {e}")


# Convenience function for running Phase 4
def run_phase4(data_path: Optional[Path] = None,
               X: Optional[np.ndarray] = None,
               y: Optional[np.ndarray] = None,
               feature_names: Optional[List[str]] = None,
               config: Optional[Dict] = None) -> Dict:
    """
    Convenience function to run Phase 4 pipeline
    
    Args:
        data_path: Path to data file (pkl or csv)
        X: Feature matrix (if data_path not provided)
        y: Labels (if data_path not provided)
        feature_names: List of feature names
        config: Configuration dictionary
    
    Returns:
        Dictionary with results
    """
    pipeline = Phase4Pipeline(config)
    return pipeline.run(data_path, X, y, feature_names)


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Load from Phase 3 results
    phase3_path = Path('data/processed/feature_extraction_results.pkl')
    
    if phase3_path.exists():
        with open(phase3_path, 'rb') as f:
            phase3_data = pickle.load(f)
        
        # Extract data
        X = phase3_data.get('X_features', phase3_data.get('X'))
        y = phase3_data.get('y_encoded', phase3_data.get('y'))
        feature_names = phase3_data.get('feature_names')
        
        # Run pipeline
        results = run_phase4(X=X, y=y, feature_names=feature_names)
    else:
        print("Phase 3 results not found. Please run Phase 3 first.")
