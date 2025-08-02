"""
Model training configuration for ECG classification
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
import numpy as np


@dataclass
class ModelTrainingConfig:
    """Configuration for model training"""
    
    # Random state for reproducibility
    random_state: int = 42
    
    # Data split configuration
    test_size: float = 0.2
    val_size: float = 0.2
    stratify: bool = True
    
    # Class balancing
    use_class_weights: bool = True
    use_smote: bool = True
    smote_strategy: Literal['auto', 'minority', 'not minority', 'not majority', 'all'] = 'auto'
    
    # Training configuration
    n_jobs: int = -1
    cv_folds: int = 5
    scoring_metric: str = 'f1_weighted'
    
    # Hyperparameter tuning
    use_hyperparameter_tuning: bool = True
    tuning_method: Literal['grid', 'random', 'bayesian'] = 'grid'
    n_iter_random: int = 50
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    
    # Model saving
    save_all_models: bool = False
    save_best_only: bool = True
    
    # Evaluation metrics
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
        'precision_macro', 'recall_macro', 'f1_macro',
        'roc_auc_ovr', 'cohen_kappa', 'matthews_corrcoef'
    ])
    
    # Clinical metrics
    clinical_metrics: bool = True
    calculate_sensitivity_specificity: bool = True
    calculate_ppv_npv: bool = True
    
    # Visualization
    create_visualizations: bool = True
    plot_confusion_matrix: bool = True
    plot_roc_curves: bool = True
    plot_learning_curves: bool = True
    plot_feature_importance: bool = True
    
    # Class names mapping
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: 'CD',
        1: 'HYP', 
        2: 'MI',
        3: 'NORM',
        4: 'STTC'
    })
    
    # Clinical significance thresholds
    clinical_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'mi_sensitivity': 0.90,  # High sensitivity for MI detection
        'norm_specificity': 0.95,  # High specificity for normal
        'overall_accuracy': 0.85
    })


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    
    name: str
    model_class: Any
    default_params: Dict[str, Any]
    param_grid: Dict[str, List[Any]]
    supports_probability: bool = True
    supports_feature_importance: bool = False
    is_ensemble: bool = False
    requires_scaling: bool = True


# Model configurations
MODEL_CONFIGS = {
    'random_forest': ModelConfig(
        name='Random Forest',
        model_class='RandomForestClassifier',
        default_params={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'class_weight': 'balanced',
            'n_jobs': -1
        },
        param_grid={
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 10, 12, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        supports_feature_importance=True,
        is_ensemble=True,
        requires_scaling=False
    ),
    
    'gradient_boosting': ModelConfig(
        name='Gradient Boosting',
        model_class='GradientBoostingClassifier',
        default_params={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        param_grid={
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10]
        },
        supports_feature_importance=True,
        is_ensemble=True
    ),
    
    'xgboost': ModelConfig(
        name='XGBoost',
        model_class='XGBClassifier',
        default_params={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softproba',
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        },
        param_grid={
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [4, 6, 8, 10],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1]
        },
        supports_feature_importance=True,
        is_ensemble=True
    ),
    
    'logistic_regression': ModelConfig(
        name='Logistic Regression',
        model_class='LogisticRegression',
        default_params={
            'class_weight': 'balanced',
            'max_iter': 1000,
            'multi_class': 'ovr',
            'solver': 'liblinear'
        },
        param_grid={
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        supports_feature_importance=True  # Via coefficients
    ),
    
    'svm': ModelConfig(
        name='Support Vector Machine',
        model_class='SVC',
        default_params={
            'class_weight': 'balanced',
            'probability': True,
            'kernel': 'rbf'
        },
        param_grid={
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4]  # Only for poly kernel
        }
    ),
    
    'knn': ModelConfig(
        name='K-Nearest Neighbors',
        model_class='KNeighborsClassifier',
        default_params={
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'minkowski',
            'p': 2
        },
        param_grid={
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }
    ),
    
    'neural_network': ModelConfig(
        name='Neural Network',
        model_class='MLPClassifier',
        default_params={
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1
        },
        param_grid={
            'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }
    ),
    
    'lightgbm': ModelConfig(
        name='LightGBM',
        model_class='LGBMClassifier',
        default_params={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multiclass',
            'class_weight': 'balanced',
            'n_jobs': -1
        },
        param_grid={
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [20, 31, 50, 100],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        },
        supports_feature_importance=True,
        is_ensemble=True
    )
}


# Ensemble configurations
ENSEMBLE_CONFIGS = {
    'voting_soft': {
        'name': 'Voting Classifier (Soft)',
        'base_models': ['random_forest', 'gradient_boosting', 'logistic_regression'],
        'voting': 'soft',
        'weights': None  # Equal weights, or specify [1, 1, 1]
    },
    
    'voting_hard': {
        'name': 'Voting Classifier (Hard)',
        'base_models': ['random_forest', 'gradient_boosting', 'svm'],
        'voting': 'hard',
        'weights': None
    },
    
    'stacking': {
        'name': 'Stacking Classifier',
        'base_models': ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm'],
        'meta_model': 'logistic_regression',
        'cv': 5,
        'use_probas': True
    }
}


# Preprocessing presets
PREPROCESSING_PRESETS = {
    'standard': {
        'scaler': 'StandardScaler',
        'handle_imbalance': 'smote',
        'feature_selection': None
    },
    
    'robust': {
        'scaler': 'RobustScaler',
        'handle_imbalance': 'smote_enn',
        'feature_selection': 'mutual_info'
    },
    
    'minimal': {
        'scaler': 'MinMaxScaler',
        'handle_imbalance': 'class_weight',
        'feature_selection': None
    }
}