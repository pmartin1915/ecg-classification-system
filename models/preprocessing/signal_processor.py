"""
Main signal processing pipeline for ECG preprocessing
"""
import numpy as np
import gc
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from config.preprocessing_config import PreprocessingConfig
from models.preprocessing.signal_quality import SignalQualityAssessor
from models.preprocessing.signal_filters import ECGFilterBank, ArtifactDetector
from models.preprocessing.signal_normalizer import SignalNormalizer


class SignalProcessor:
    """Main class for processing ECG signals"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.quality_assessor = SignalQualityAssessor(config)
        self.filter_bank = ECGFilterBank(config)
        self.artifact_detector = ArtifactDetector(config)
        self.normalizer = SignalNormalizer(config)
        
    def process_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single ECG signal through the complete pipeline
        
        Args:
            signal: Raw ECG signal (time_points, leads)
            
        Returns:
            Processed signal and processing info
        """
        processing_info = {}
        
        # Ensure 2D array
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        # Step 1: Apply filters
        filtered_signal = self.filter_bank.apply_filters(signal)
        
        # Step 2: Detect artifacts
        artifacts = self.artifact_detector.detect_artifacts(filtered_signal)
        processing_info['artifacts'] = artifacts
        
        # Step 3: Remove artifacts
        cleaned_signal = self.artifact_detector.remove_artifacts(
            filtered_signal, artifacts, method='clip'
        )
        
        # Step 4: Standardize length
        processed_signal = self._standardize_length(cleaned_signal)
        
        processing_info['original_length'] = signal.shape[0]
        processing_info['final_length'] = processed_signal.shape[0]
        
        return processed_signal, processing_info
    
    def process_batch(self, 
                     X: np.ndarray,
                     use_cache: bool = True,
                     cache_dir: Optional[Path] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process batch of ECG signals
        
        Args:
            X: Batch of raw signals (n_samples, time_points, leads)
            use_cache: Whether to use cached results
            cache_dir: Directory for cache files
            
        Returns:
            Processed signals and processing info for each signal
        """
        print("=== SIGNAL PREPROCESSING ===")
        
        # Check cache
        if use_cache and cache_dir:
            cache_file = cache_dir / f"preprocessed_signals_{len(X)}.pkl"
            if cache_file.exists():
                print(f"Loading from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        print(f"Processing {len(X)} signals...")
        print(f"Filters: {list(self.filter_bank.filters.keys())}")
        
        X_processed = []
        all_processing_info = []
        
        # Process in batches for memory efficiency
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(X), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(X))
            batch = X[i:batch_end]
            
            for signal in batch:
                processed_signal, info = self.process_signal(signal)
                X_processed.append(processed_signal)
                all_processing_info.append(info)
            
            # Memory management
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        X_processed = np.array(X_processed, dtype=self.config.output_dtype)
        
        print(f"\n✅ Preprocessing complete:")
        print(f"   - Output shape: {X_processed.shape}")
        print(f"   - Memory usage: {X_processed.nbytes / (1024**3):.2f} GB")
        
        # Save to cache
        if use_cache and cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump((X_processed, all_processing_info), f)
        
        return X_processed, all_processing_info
    
    def _standardize_length(self, signal: np.ndarray) -> np.ndarray:
        """Standardize signal to target length"""
        current_length = signal.shape[0]
        target_length = self.config.target_length
        
        if current_length == target_length:
            return signal
        
        if current_length > target_length:
            # Truncate from the center
            start = (current_length - target_length) // 2
            return signal[start:start + target_length]
        else:
            # Pad with zeros
            pad_before = (target_length - current_length) // 2
            pad_after = target_length - current_length - pad_before
            
            if signal.ndim == 1:
                return np.pad(signal, (pad_before, pad_after), mode='constant')
            else:
                return np.pad(signal, ((pad_before, pad_after), (0, 0)), mode='constant')
    
    def get_processing_statistics(self, processing_info_list: List[Dict]) -> Dict:
        """Generate statistics from processing info"""
        stats = {
            'total_signals': len(processing_info_list),
            'artifact_counts': {
                'amplitude': 0,
                'gradient': 0,
                'saturation': 0,
                'baseline_drift': 0
            },
            'length_changes': {
                'truncated': 0,
                'padded': 0,
                'unchanged': 0
            }
        }
        
        for info in processing_info_list:
            # Count artifacts
            for artifact_type, artifact_list in info['artifacts'].items():
                if artifact_list:
                    stats['artifact_counts'][artifact_type] += 1
            
            # Count length changes
            if info['original_length'] > info['final_length']:
                stats['length_changes']['truncated'] += 1
            elif info['original_length'] < info['final_length']:
                stats['length_changes']['padded'] += 1
            else:
                stats['length_changes']['unchanged'] += 1
        
        return stats