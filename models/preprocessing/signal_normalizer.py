"""
Signal normalization utilities for ECG preprocessing
"""
import numpy as np
from typing import Dict, Tuple, Optional
import pickle
from pathlib import Path

from config.preprocessing_config import PreprocessingConfig


class SignalNormalizer:
    """Normalize ECG signals using various methods"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.normalization_params = {}
        
    def normalize(self, 
                 X: np.ndarray, 
                 method: Optional[str] = None,
                 fit: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Normalize ECG signals
        
        Args:
            X: Input signals (n_samples, time_points, leads)
            method: Normalization method (if None, uses config)
            fit: Whether to fit normalization parameters
            
        Returns:
            Normalized signals and parameters
        """
        method = method or self.config.normalization_method
        
        print(f"=== SIGNAL NORMALIZATION ({method.upper()}) ===")
        
        if method == 'z-score':
            X_normalized, params = self._zscore_normalize(X, fit)
        elif method == 'min-max':
            X_normalized, params = self._minmax_normalize(X, fit)
        elif method == 'robust':
            X_normalized, params = self._robust_normalize(X, fit)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Store parameters
        if fit:
            self.normalization_params = params
        
        self._print_normalization_summary(X, X_normalized, method)
        
        return X_normalized, params
    
    def _zscore_normalize(self, X: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, Dict]:
        """Z-score normalization (mean=0, std=1)"""
        X_normalized = np.zeros_like(X, dtype=self.config.output_dtype)
        
        if fit:
            # Calculate parameters
            params = {
                'method': 'z-score',
                'per_lead': True,
                'lead_params': {}
            }
            
            for lead_idx in range(X.shape[2]):
                lead_data = X[:, :, lead_idx].flatten()
                mean_val = np.mean(lead_data)
                std_val = np.std(lead_data)
                
                params['lead_params'][lead_idx] = {
                    'mean': float(mean_val),
                    'std': float(std_val)
                }
        else:
            # Use existing parameters
            params = self.normalization_params
        
        # Apply normalization
        for i in range(X.shape[0]):
            for lead_idx in range(X.shape[2]):
                lead_signal = X[i, :, lead_idx]
                
                if fit or lead_idx not in params['lead_params']:
                    # Per-signal normalization
                    mean_val = np.mean(lead_signal)
                    std_val = np.std(lead_signal)
                else:
                    # Use fitted parameters
                    mean_val = params['lead_params'][lead_idx]['mean']
                    std_val = params['lead_params'][lead_idx]['std']
                
                if std_val > 1e-8:
                    X_normalized[i, :, lead_idx] = (lead_signal - mean_val) / std_val
                else:
                    X_normalized[i, :, lead_idx] = lead_signal - mean_val
        
        return X_normalized, params
    
    def _minmax_normalize(self, X: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, Dict]:
        """Min-max normalization (0 to 1)"""
        X_normalized = np.zeros_like(X, dtype=self.config.output_dtype)
        
        if fit:
            params = {
                'method': 'min-max',
                'per_lead': True,
                'lead_params': {}
            }
            
            for lead_idx in range(X.shape[2]):
                lead_data = X[:, :, lead_idx].flatten()
                min_val = np.min(lead_data)
                max_val = np.max(lead_data)
                
                params['lead_params'][lead_idx] = {
                    'min': float(min_val),
                    'max': float(max_val)
                }
        else:
            params = self.normalization_params
        
        # Apply normalization
        for i in range(X.shape[0]):
            for lead_idx in range(X.shape[2]):
                lead_signal = X[i, :, lead_idx]
                
                if fit or lead_idx not in params['lead_params']:
                    min_val = np.min(lead_signal)
                    max_val = np.max(lead_signal)
                else:
                    min_val = params['lead_params'][lead_idx]['min']
                    max_val = params['lead_params'][lead_idx]['max']
                
                if max_val > min_val:
                    X_normalized[i, :, lead_idx] = (lead_signal - min_val) / (max_val - min_val)
                else:
                    X_normalized[i, :, lead_idx] = 0
        
        return X_normalized, params
    
    def _robust_normalize(self, X: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, Dict]:
        """Robust normalization using median and IQR"""
        X_normalized = np.zeros_like(X, dtype=self.config.output_dtype)
        
        if fit:
            params = {
                'method': 'robust',
                'per_lead': True,
                'lead_params': {}
            }
            
            for lead_idx in range(X.shape[2]):
                lead_data = X[:, :, lead_idx].flatten()
                median_val = np.median(lead_data)
                q75, q25 = np.percentile(lead_data, [75, 25])
                iqr = q75 - q25
                
                params['lead_params'][lead_idx] = {
                    'median': float(median_val),
                    'iqr': float(iqr),
                    'q25': float(q25),
                    'q75': float(q75)
                }
        else:
            params = self.normalization_params
        
        # Apply normalization
        for i in range(X.shape[0]):
            for lead_idx in range(X.shape[2]):
                lead_signal = X[i, :, lead_idx]
                
                if fit or lead_idx not in params['lead_params']:
                    median_val = np.median(lead_signal)
                    q75, q25 = np.percentile(lead_signal, [75, 25])
                    iqr = q75 - q25
                else:
                    median_val = params['lead_params'][lead_idx]['median']
                    iqr = params['lead_params'][lead_idx]['iqr']
                
                if iqr > 1e-8:
                    X_normalized[i, :, lead_idx] = (lead_signal - median_val) / iqr
                else:
                    X_normalized[i, :, lead_idx] = lead_signal - median_val
        
        return X_normalized, params
    
    def denormalize(self, X_normalized: np.ndarray) -> np.ndarray:
        """Reverse normalization to get original scale"""
        if not self.normalization_params:
            raise ValueError("No normalization parameters available")
        
        method = self.normalization_params['method']
        X_denorm = np.zeros_like(X_normalized)
        
        for i in range(X_normalized.shape[0]):
            for lead_idx in range(X_normalized.shape[2]):
                lead_signal = X_normalized[i, :, lead_idx]
                params = self.normalization_params['lead_params'][lead_idx]
                
                if method == 'z-score':
                    X_denorm[i, :, lead_idx] = (lead_signal * params['std']) + params['mean']
                elif method == 'min-max':
                    X_denorm[i, :, lead_idx] = (lead_signal * (params['max'] - params['min'])) + params['min']
                elif method == 'robust':
                    X_denorm[i, :, lead_idx] = (lead_signal * params['iqr']) + params['median']
        
        return X_denorm
    
    def _print_normalization_summary(self, X_original: np.ndarray, 
                                   X_normalized: np.ndarray, 
                                   method: str):
        """Print normalization summary"""
        print(f"\n✅ Normalization complete:")
        print(f"   - Method: {method}")
        print(f"   - Original range: [{X_original.min():.3f}, {X_original.max():.3f}]")
        print(f"   - Normalized range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
        print(f"   - Original mean: {X_original.mean():.3f}")
        print(f"   - Normalized mean: {X_normalized.mean():.3f}")
        print(f"   - Original std: {X_original.std():.3f}")
        print(f"   - Normalized std: {X_normalized.std():.3f}")
    
    def save_parameters(self, filepath: Path):
        """Save normalization parameters"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.normalization_params, f)
    
    def load_parameters(self, filepath: Path):
        """Load normalization parameters"""
        with open(filepath, 'rb') as f:
            self.normalization_params = pickle.load(f)