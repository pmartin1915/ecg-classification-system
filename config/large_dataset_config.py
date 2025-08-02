"""
Configuration and utilities for handling large ECG datasets (e.g., arrhythmia dataset)
This module provides optimized settings and helper functions for memory-efficient processing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import psutil
import os
import gc
from datetime import datetime


class LargeDatasetConfig:
    """Configuration optimized for large datasets"""
    
    # Memory management
    MAX_MEMORY_MB = 4096  # Maximum memory usage in MB
    CHUNK_SIZE = 1000  # Process data in chunks
    BATCH_SIZE = 500   # Batch size for model training
    
    # Data processing
    USE_FLOAT32 = True  # Use float32 instead of float64 to save memory
    SPARSE_FEATURES = True  # Use sparse matrices where possible
    
    # Model training optimizations
    SUBSAMPLE_FOR_TUNING = True  # Use subset for hyperparameter tuning
    TUNING_SAMPLE_SIZE = 5000  # Samples to use for hyperparameter tuning
    
    # Progressive training
    PROGRESSIVE_TRAINING = True  # Train on increasing data sizes
    PROGRESSIVE_SIZES = [1000, 5000, 10000, 50000, 'all']  # Sample sizes
    
    # Model selection for large datasets
    LARGE_DATASET_MODELS = [
        'logistic_regression',  # Scales well
        'random_forest',        # Can handle large data with proper settings
        'lightgbm',            # Excellent for large datasets
        'sgd_classifier'       # Online learning capability
    ]
    
    # Reduced hyperparameter grids for faster tuning
    FAST_PARAM_GRIDS = {
        'logistic_regression': {
            'C': [0.1, 1, 10],
            'solver': ['saga']  # Supports all penalties and large datasets
        },
        'random_forest': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'max_samples': [0.5, 0.8]  # Subsample for each tree
        },
        'lightgbm': {
            'n_estimators': [50, 100],
            'num_leaves': [31, 50],
            'learning_rate': [0.05, 0.1]
        }
    }
    
    # Validation strategy
    VALIDATION_STRATEGY = 'time_series'  # or 'stratified'
    N_SPLITS = 3  # Fewer splits for large data
    
    # Feature selection for dimensionality reduction
    USE_FEATURE_SELECTION = True
    TOP_FEATURES = 100  # Keep top N features
    FEATURE_SELECTION_METHOD = 'mutual_info'  # or 'chi2', 'anova'
    
    # Monitoring
    MONITOR_MEMORY = True
    MEMORY_CHECKPOINT_INTERVAL = 1000  # Check memory every N samples
    
    # Output settings
    SAVE_INTERMEDIATE = True  # Save intermediate results
    COMPRESSION = 'gzip'  # Compress saved files


class MemoryEfficientDataLoader:
    """Memory-efficient data loading for large datasets"""
    
    def __init__(self, config: LargeDatasetConfig):
        self.config = config
        self.memory_limit = config.MAX_MEMORY_MB * 1024 * 1024  # Convert to bytes
        
    def load_arrhythmia_dataset(self, data_path: Path, 
                               target_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load large arrhythmia dataset efficiently
        
        Args:
            data_path: Path to dataset file
            target_column: Name of the target column
            
        Returns:
            X: Feature matrix
            y: Labels
        """
        print(f"Loading large dataset from {data_path}...")
        
        # Check file size
        file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        
        if data_path.suffix == '.csv':
            return self._load_csv_in_chunks(data_path, target_column)
        elif data_path.suffix in ['.pkl', '.pickle']:
            return self._load_pickle_efficiently(data_path)
        elif data_path.suffix == '.parquet':
            return self._load_parquet(data_path, target_column)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def _load_csv_in_chunks(self, file_path: Path, 
                           target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSV file in chunks to manage memory"""
        chunks = []
        labels = []
        
        # First pass: get column info
        first_chunk = pd.read_csv(file_path, nrows=5)
        feature_columns = [col for col in first_chunk.columns if col != target_column]
        
        # Load data in chunks
        chunk_iter = pd.read_csv(file_path, chunksize=self.config.CHUNK_SIZE)
        
        for i, chunk in enumerate(chunk_iter):
            # Monitor memory
            if self.config.MONITOR_MEMORY and i % 10 == 0:
                current_memory = psutil.virtual_memory().used
                if current_memory > self.memory_limit:
                    print(f"⚠️  Memory limit approaching, processed {i * self.config.CHUNK_SIZE} samples")
                    break
            
            # Separate features and labels
            if target_column in chunk.columns:
                y_chunk = chunk[target_column].values
                X_chunk = chunk[feature_columns]
            else:
                # If no label column, assume last column is label
                y_chunk = chunk.iloc[:, -1].values
                X_chunk = chunk.iloc[:, :-1]
            
            # Optimize data types
            X_chunk = self._optimize_dataframe(X_chunk)
            
            # Convert to numpy and use float32
            if self.config.USE_FLOAT32:
                X_chunk = X_chunk.astype(np.float32).values
            else:
                X_chunk = X_chunk.values
            
            chunks.append(X_chunk)
            labels.append(y_chunk)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Loaded {(i + 1) * self.config.CHUNK_SIZE:,} samples...")
        
        # Combine chunks
        X = np.vstack(chunks)
        y = np.concatenate(labels)
        
        print(f"✅ Loaded {X.shape[0]:,} samples with {X.shape[1]} features")
        
        # Clear memory
        del chunks, labels
        gc.collect()
        
        return X, y
    
    def _load_pickle_efficiently(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load pickle file with memory optimization"""
        import pickle
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract features and labels
        if isinstance(data, dict):
            X = data.get('X', data.get('features', data.get('X_features')))
            y = data.get('y', data.get('labels', data.get('y_encoded')))
        elif isinstance(data, tuple) and len(data) == 2:
            X, y = data
        else:
            raise ValueError("Unknown data format in pickle file")
        
        # Convert to float32 if needed
        if self.config.USE_FLOAT32 and X.dtype == np.float64:
            X = X.astype(np.float32)
        
        return X, y
    
    def _load_parquet(self, file_path: Path, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load parquet file (memory efficient format)"""
        df = pd.read_parquet(file_path)
        
        # Separate features and labels
        if target_column in df.columns:
            y = df[target_column].values
            X = df.drop(columns=[target_column])
        else:
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1]
        
        # Optimize memory
        X = self._optimize_dataframe(X)
        
        if self.config.USE_FLOAT32:
            X = X.astype(np.float32).values
        else:
            X = X.values
        
        return X, y
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
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
                    if self.config.USE_FLOAT32:
                        df[col] = df[col].astype(np.float32)
        
        return df


class IncrementalModelTrainer:
    """Train models incrementally on large datasets"""
    
    def __init__(self, config: LargeDatasetConfig):
        self.config = config
        self.models = self._initialize_incremental_models()
        
    def _initialize_incremental_models(self) -> Dict[str, Any]:
        """Initialize models that support incremental learning"""
        from sklearn.linear_model import SGDClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.neural_network import MLPClassifier
        
        models = {
            'sgd_classifier': SGDClassifier(
                loss='log',  # Logistic regression
                penalty='l2',
                alpha=0.001,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'sgd_svm': SGDClassifier(
                loss='hinge',  # Linear SVM
                penalty='l2',
                alpha=0.001,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'naive_bayes': MultinomialNB(alpha=1.0),
            'mlp_incremental': MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=1,  # Single iteration per batch
                warm_start=True,  # Continue from previous state
                random_state=42
            )
        }
        
        return models
    
    def train_incrementally(self, X: np.ndarray, y: np.ndarray, 
                          model_name: str, batch_size: int = 1000) -> Dict:
        """Train model incrementally in batches"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available for incremental training")
        
        model = self.models[model_name]
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"Training {model_name} incrementally on {n_batches} batches...")
        
        start_time = datetime.now()
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X[i:batch_end]
            y_batch = y[i:batch_end]
            
            # Partial fit
            if hasattr(model, 'partial_fit'):
                if i == 0:
                    # First batch: specify classes
                    model.partial_fit(X_batch, y_batch, classes=np.unique(y))
                else:
                    model.partial_fit(X_batch, y_batch)
            else:
                # For models without partial_fit but with warm_start
                model.fit(X_batch, y_batch)
            
            # Progress update
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + batch_size:,} / {n_samples:,} samples...")
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Incremental training completed in {training_time:.1f}s")
        
        return {
            'model': model,
            'training_time': training_time,
            'n_samples_trained': n_samples
        }


class FeatureSelector:
    """Feature selection for large datasets"""
    
    def __init__(self, config: LargeDatasetConfig):
        self.config = config
        self.selected_features = None
        self.selector = None
        
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Select top features to reduce dimensionality"""
        print(f"Selecting top {self.config.TOP_FEATURES} features...")
        
        if self.config.FEATURE_SELECTION_METHOD == 'mutual_info':
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            self.selector = SelectKBest(mutual_info_classif, k=self.config.TOP_FEATURES)
        elif self.config.FEATURE_SELECTION_METHOD == 'chi2':
            from sklearn.feature_selection import SelectKBest, chi2
            # Ensure non-negative values for chi2
            X_min = X.min()
            if X_min < 0:
                X = X - X_min
            self.selector = SelectKBest(chi2, k=self.config.TOP_FEATURES)
        elif self.config.FEATURE_SELECTION_METHOD == 'anova':
            from sklearn.feature_selection import SelectKBest, f_classif
            self.selector = SelectKBest(f_classif, k=self.config.TOP_FEATURES)
        
        # Fit on a subset if data is too large
        if len(X) > 10000:
            subset_idx = np.random.choice(len(X), 10000, replace=False)
            self.selector.fit(X[subset_idx], y[subset_idx])
        else:
            self.selector.fit(X, y)
        
        # Transform data
        X_selected = self.selector.transform(X)
        
        # Get selected feature names
        selected_indices = self.selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
        print(f"✅ Selected {len(selected_feature_names)} features")
        print(f"   Original shape: {X.shape}")
        print(f"   New shape: {X_selected.shape}")
        
        self.selected_features = selected_feature_names
        
        return X_selected, selected_feature_names


def optimize_training_for_large_dataset(X: np.ndarray, y: np.ndarray,
                                      feature_names: List[str],
                                      arrhythmia: bool = True) -> Dict:
    """
    Optimized training pipeline for large datasets
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: Feature names
        arrhythmia: Whether this is the arrhythmia dataset
        
    Returns:
        Dictionary with results
    """
    config = LargeDatasetConfig()
    
    # Step 1: Feature selection (if enabled)
    if config.USE_FEATURE_SELECTION and X.shape[1] > config.TOP_FEATURES:
        selector = FeatureSelector(config)
        X, feature_names = selector.select_features(X, y, feature_names)
        gc.collect()
    
    # Step 2: Progressive training (if enabled)
    if config.PROGRESSIVE_TRAINING:
        results = {}
        
        for size in config.PROGRESSIVE_SIZES:
            if size == 'all':
                n_samples = len(X)
            else:
                n_samples = min(size, len(X))
            
            print(f"\n{'='*60}")
            print(f"Training on {n_samples:,} samples...")
            print('='*60)
            
            # Subsample data
            if n_samples < len(X):
                idx = np.random.choice(len(X), n_samples, replace=False)
                X_subset = X[idx]
                y_subset = y[idx]
            else:
                X_subset = X
                y_subset = y
            
            # Train models
            # Import the main Phase4Pipeline from phase4_model_training.py
            from models.training import Phase4Pipeline
            
            # Configure for subset
            subset_config = {
                'test_size': 0.2,
                'val_size': 0.1,
                'use_smote': True,
                'use_hyperparameter_tuning': n_samples <= 5000,  # Only tune on smaller sets
                'model_keys': config.LARGE_DATASET_MODELS,
                'create_visualizations': False,
                'save_models': n_samples == len(X)  # Only save final model
            }
            
            pipeline = Phase4Pipeline(subset_config)
            subset_results = pipeline.run(X=X_subset, y=y_subset, feature_names=feature_names)
            
            results[f'n_samples_{n_samples}'] = {
                'accuracy': subset_results['test_results']['metrics']['accuracy'],
                'f1_score': subset_results['test_results']['metrics']['f1_weighted'],
                'best_model': subset_results['best_model_name'],
                'training_time': sum([r['training_time'] for r in subset_results['trainer'].results.values()])
            }
            
            # Clear memory
            del X_subset, y_subset, subset_results
            gc.collect()
        
        # Print progressive results
        print("\n📊 Progressive Training Results:")
        print(f"{'Samples':<15} {'Best Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Time (s)':<10}")
        print("-" * 75)
        
        for key, res in results.items():
            n_samples = key.split('_')[-1]
            print(f"{n_samples:<15} {res['best_model']:<20} "
                  f"{res['accuracy']:<10.3f} {res['f1_score']:<10.3f} "
                  f"{res['training_time']:<10.1f}")
        
        return results
    
    else:
        # Standard training with memory optimization
        from models.training import Phase4Pipeline
        
        config_dict = {
            'test_size': 0.2,
            'val_size': 0.1,
            'use_smote': True,
            'use_hyperparameter_tuning': False,
            'model_keys': config.LARGE_DATASET_MODELS,
            'create_visualizations': True,
            'save_models': True
        }
        
        pipeline = Phase4Pipeline(config_dict)
        return pipeline.run(X=X, y=y, feature_names=feature_names)


# Example usage for arrhythmia dataset
if __name__ == "__main__":
    from pathlib import Path
    
    # Example paths
    arrhythmia_path = Path('data/raw/arrhythmia_dataset.csv')
    
    if arrhythmia_path.exists():
        # Initialize components
        config = LargeDatasetConfig()
        loader = MemoryEfficientDataLoader(config)
        
        # Load data
        X, y = loader.load_arrhythmia_dataset(arrhythmia_path)
        
        # Assume feature names are column names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Run optimized training
        results = optimize_training_for_large_dataset(X, y, feature_names, arrhythmia=True)
        
        print("\n✅ Training completed successfully!")
    else:
        print(f"Dataset not found at {arrhythmia_path}")
        print("Please update the path to your arrhythmia dataset")