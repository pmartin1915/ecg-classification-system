"""
Main feature extraction pipeline
"""
import numpy as np
import pandas as pd
import pickle
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from joblib import Parallel, delayed

from config.feature_config import FeatureExtractionConfig
from models.feature_extraction.temporal_features import TemporalFeatureExtractor
from models.feature_extraction.frequency_features import FrequencyFeatureExtractor
from models.feature_extraction.st_segment_features import STSegmentAnalyzer
from models.feature_extraction.wavelet_features import WaveletFeatureExtractor


class ECGFeatureExtractor:
    """Main class for extracting all ECG features"""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        
        # Initialize extractors
        self.temporal_extractor = TemporalFeatureExtractor(config)
        self.frequency_extractor = FrequencyFeatureExtractor(config)
        self.st_analyzer = STSegmentAnalyzer(config)
        self.wavelet_extractor = WaveletFeatureExtractor(config)
    
    def extract_features_single_signal(self, signal: np.ndarray, 
                                     signal_idx: int) -> Dict[str, float]:
        """
        Extract all features from a single ECG signal
        
        Args:
            signal: ECG signal (time_points, leads)
            signal_idx: Index of signal for error reporting
            
        Returns:
            Dictionary of all extracted features
        """
        features = {}
        
        try:
            # Extract temporal features
            temporal_features = self.temporal_extractor.extract_statistical_features(signal)
            features.update(temporal_features)
            
            morphological_features, r_peaks = self.temporal_extractor.extract_morphological_features(signal)
            features.update(morphological_features)
            
            # Extract cross-lead features
            cross_lead_features = self.temporal_extractor.extract_cross_lead_features(signal)
            features.update(cross_lead_features)
            
            # Extract frequency features
            spectral_features = self.frequency_extractor.extract_spectral_features(signal)
            features.update(spectral_features)
            
            # Extract HRV frequency features if we have R-peaks
            if len(r_peaks) > 1:
                rr_intervals = np.diff(r_peaks) / self.config.sampling_rate
                valid_rr = rr_intervals[
                    (rr_intervals >= self.config.hrv_min_rr) & 
                    (rr_intervals <= self.config.hrv_max_rr)
                ]
                if len(valid_rr) > 1:
                    hrv_freq_features = self.frequency_extractor.extract_hrv_frequency_features(valid_rr)
                    features.update(hrv_freq_features)
            
            # Extract ST-segment features
            st_features = self.st_analyzer.extract_st_features(signal, r_peaks)
            features.update(st_features)
            
            # Extract wavelet features
            wavelet_features = self.wavelet_extractor.extract_wavelet_features(signal)
            features.update(wavelet_features)
            
            # Add signal quality indicator
            features['signal_quality_score'] = self._calculate_signal_quality_score(signal, features)
            
        except Exception as e:
            print(f"⚠️  Error extracting features for signal {signal_idx}: {e}")
            # Return minimal features on error
            features = self._get_default_features()
        
        return features
    
    def extract_all_features(self, X: np.ndarray, 
                           use_cache: bool = True,
                           cache_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Extract features from all ECG signals
        
        Args:
            X: Array of ECG signals (n_samples, time_points, leads)
            use_cache: Whether to use cached results
            cache_dir: Directory for cache files
            
        Returns:
            DataFrame of extracted features
        """
        print("=== FEATURE EXTRACTION ===")
        
        # Check cache
        if use_cache and cache_dir:
            cache_file = cache_dir / f"extracted_features_{len(X)}.pkl"
            if cache_file.exists():
                print(f"Loading from cache: {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Cache error: {e}. Extracting fresh features.")
                    cache_file.unlink()
        
        # Extract features in parallel
        print(f"Extracting features from {len(X)} signals...")
        
        if self.config.n_jobs > 1:
            # Parallel extraction
            all_features = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self.extract_features_single_signal)(signal, idx)
                for idx, signal in enumerate(tqdm(X, desc="Processing signals"))
            )
        else:
            # Sequential extraction
            all_features = []
            for idx, signal in enumerate(tqdm(X, desc="Processing signals")):
                features = self.extract_features_single_signal(signal, idx)
                all_features.append(features)
                
                # Memory management
                if idx % 1000 == 0:
                    gc.collect()
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features)
        
        # Handle missing values
        feature_df = self._handle_missing_values(feature_df)
        
        # Remove zero-variance features
        feature_df = self._remove_zero_variance_features(feature_df)
        
        print(f"\n✅ Feature extraction complete:")
        print(f"   - Total features: {len(feature_df.columns):,}")
        print(f"   - Feature matrix shape: {feature_df.shape}")
        print(f"   - Memory usage: {feature_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Save to cache
        if use_cache and cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(feature_df, f)
        
        return feature_df
    
    def _calculate_signal_quality_score(self, signal: np.ndarray, 
                                      features: Dict[str, float]) -> float:
        """Calculate overall signal quality score"""
        quality_indicators = []
        
        # Check R-peak detection quality
        if 'r_peak_count' in features:
            expected_peaks = (signal.shape[0] / self.config.sampling_rate) * 1.2  # ~72 bpm
            peak_ratio = features['r_peak_count'] / expected_peaks
            quality_indicators.append(1.0 - abs(1.0 - peak_ratio))
        
        # Check signal variability
        if 'II_std' in features:  # Lead II standard deviation
            if 0.1 < features['II_std'] < 2.0:  # Reasonable range
                quality_indicators.append(1.0)
            else:
                quality_indicators.append(0.5)
        
        # Check for saturation
        max_val = np.max(np.abs(signal))
        if max_val < 5.0:  # Not saturated
            quality_indicators.append(1.0)
        else:
            quality_indicators.append(0.0)
        
        # Average quality indicators
        if quality_indicators:
            return float(np.mean(quality_indicators))
        else:
            return 0.5
    
    def _handle_missing_values(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature DataFrame"""
        print("\n🔍 Handling missing values:")
        
        missing_counts = feature_df.isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]
        
        if len(features_with_missing) > 0:
            print(f"   - Features with missing values: {len(features_with_missing)}")
            print(f"   - Max missing count: {features_with_missing.max()}")
            
            # Strategy 1: Fill with median for numerical features
            numerical_features = feature_df.select_dtypes(include=[np.number]).columns
            feature_df[numerical_features] = feature_df[numerical_features].fillna(
                feature_df[numerical_features].median()
            )
            
            # Strategy 2: Fill remaining with zeros
            feature_df = feature_df.fillna(0)
            
            print(f"   - Filled missing values")
        else:
            print(f"   - No missing values found")
        
        return feature_df
    
    def _remove_zero_variance_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero or very low variance"""
        variances = feature_df.var()
        zero_var_features = variances[variances < self.config.variance_threshold].index.tolist()
        
        if zero_var_features:
            print(f"   - Removing {len(zero_var_features)} zero-variance features")
            feature_df = feature_df.drop(columns=zero_var_features)
        
        return feature_df
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default features when extraction fails"""
        features = {}
        
        # Add default values for each feature type
        # This ensures consistent feature dimensions
        
        # Temporal features
        for lead in self.config.lead_names:
            for stat in ['mean', 'std', 'var', 'median', 'min', 'max', 'range', 'rms']:
                features[f'{lead}_{stat}'] = 0.0
        
        # Morphological features
        morph_features = ['r_peak_count', 'r_peak_density', 'rr_mean', 'heart_rate_mean']
        for feat in morph_features:
            features[feat] = 0.0
        
        # Frequency features
        for lead in self.config.lead_names:
            features[f'{lead}_total_power'] = 0.0
            for band in self.config.freq_bands:
                features[f'{lead}_{band}_power'] = 0.0
        
        # ST features
        features['mi_probability_score'] = 0.0
        
        # Wavelet features
        features['cwt_total_energy'] = 0.0
        features['dwt_total_energy'] = 0.0
        
        # Quality score
        features['signal_quality_score'] = 0.0
        
        return features
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature names organized by type"""
        return {
            'temporal': ['mean', 'std', 'var', 'median', 'min', 'max', 'range', 'rms', 
                        'skewness', 'kurtosis', 'zero_crossings'],
            'morphological': ['r_peak', 'rr_', 'heart_rate', 'qrs_', 'pnn', 'rmssd', 
                            'sdnn', 'sdsd'],
            'frequency': ['power', 'freq', 'spectral', 'hrv_'],
            'st_segment': ['st_elevation', 'st_depression', 'st_slope', 'j_point', 
                          'stemi_score', 'mi_probability'],
            'wavelet': ['cwt_', 'dwt_', 'wavelet_'],
            'quality': ['signal_quality_score']
        }