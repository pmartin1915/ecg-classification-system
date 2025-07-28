"""
Frequency domain feature extraction for ECG signals
"""
import numpy as np
from scipy import signal
from scipy.signal import welch, periodogram
from scipy.stats import entropy
from typing import Dict, Optional, List

from config.feature_config import FeatureExtractionConfig


class FrequencyFeatureExtractor:
    """Extract frequency domain features from ECG signals"""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
    
    def extract_spectral_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features using FFT and Welch's method
        
        Args:
            ecg_signal: ECG signal (time_points, leads)
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        for lead_idx in range(ecg_signal.shape[1]):
            lead_signal = ecg_signal[:, lead_idx]
            lead_name = self.config.lead_names[lead_idx]
            
            # Power Spectral Density using Welch's method
            nperseg = min(256, len(lead_signal) // 4)
            freqs, psd = welch(
                lead_signal,
                fs=self.config.sampling_rate,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                detrend='constant'
            )
            
            # Total power
            total_power = np.trapz(psd, freqs)
            features[f'{lead_name}_total_power'] = float(total_power)
            
            # Power in different frequency bands
            for band_name, (low_freq, high_freq) in self.config.freq_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                if np.any(band_mask):
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                    features[f'{lead_name}_{band_name}_power'] = float(band_power)
                    
                    # Relative power
                    if total_power > 0:
                        rel_power = band_power / total_power
                        features[f'{lead_name}_{band_name}_rel_power'] = float(rel_power)
                    else:
                        features[f'{lead_name}_{band_name}_rel_power'] = 0.0
                else:
                    features[f'{lead_name}_{band_name}_power'] = 0.0
                    features[f'{lead_name}_{band_name}_rel_power'] = 0.0
            
            # Spectral statistics
            if len(psd) > 0 and np.sum(psd) > 0:
                features[f'{lead_name}_dominant_freq'] = float(freqs[np.argmax(psd)])
                features[f'{lead_name}_spectral_centroid'] = float(np.sum(freqs * psd) / np.sum(psd))
                features[f'{lead_name}_spectral_rolloff'] = float(self._spectral_rolloff(freqs, psd))
                features[f'{lead_name}_spectral_bandwidth'] = float(self._spectral_bandwidth(freqs, psd))
                features[f'{lead_name}_spectral_flatness'] = float(self._spectral_flatness(psd))
                
                # Spectral entropy
                psd_norm = psd / np.sum(psd)
                features[f'{lead_name}_spectral_entropy'] = float(entropy(psd_norm))
            else:
                # Default values if PSD calculation fails
                features[f'{lead_name}_dominant_freq'] = 0.0
                features[f'{lead_name}_spectral_centroid'] = 0.0
                features[f'{lead_name}_spectral_rolloff'] = 0.0
                features[f'{lead_name}_spectral_bandwidth'] = 0.0
                features[f'{lead_name}_spectral_flatness'] = 0.0
                features[f'{lead_name}_spectral_entropy'] = 0.0
            
            # Additional frequency features
            features.update(self._extract_advanced_spectral_features(lead_signal, lead_name))
        
        return features
    
    def extract_hrv_frequency_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract HRV frequency domain features
        
        Args:
            rr_intervals: RR intervals in seconds
            
        Returns:
            Dictionary of HRV frequency features
        """
        features = {}
        
        if len(rr_intervals) < 10:
            # Not enough data for reliable frequency analysis
            return self._get_default_hrv_features()
        
        try:
            # Interpolate RR intervals to regular sampling
            time_original = np.cumsum(rr_intervals)
            time_original = np.insert(time_original, 0, 0)  # Add zero at start
            
            # Create regular time grid at 4 Hz
            fs_interp = 4.0
            time_regular = np.arange(0, time_original[-1], 1.0 / fs_interp)
            
            if len(time_regular) < 10:
                return self._get_default_hrv_features()
            
            # Interpolate RR intervals
            rr_interpolated = np.interp(time_regular, time_original[:-1], rr_intervals)
            
            # Remove trend
            rr_detrended = signal.detrend(rr_interpolated)
            
            # Apply window to reduce spectral leakage
            window = signal.windows.hann(len(rr_detrended))
            rr_windowed = rr_detrended * window
            
            # Power spectral density
            nperseg = min(256, len(rr_windowed) // 2)
            freqs, psd = welch(
                rr_windowed,
                fs=fs_interp,
                nperseg=nperseg,
                noverlap=nperseg // 2
            )
            
            # HRV frequency bands
            vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
            
            # Power in each band
            vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0.0
            lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0.0
            hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0.0
            
            features['hrv_vlf_power'] = float(vlf_power)
            features['hrv_lf_power'] = float(lf_power)
            features['hrv_hf_power'] = float(hf_power)
            
            # Total power
            total_power = vlf_power + lf_power + hf_power
            features['hrv_total_power'] = float(total_power)
            
            # Normalized powers and ratios
            if total_power > 0:
                features['hrv_vlf_norm'] = float(vlf_power / total_power)
                features['hrv_lf_norm'] = float(lf_power / total_power)
                features['hrv_hf_norm'] = float(hf_power / total_power)
                
                # LF/HF ratio
                if hf_power > 0:
                    features['hrv_lf_hf_ratio'] = float(lf_power / hf_power)
                else:
                    features['hrv_lf_hf_ratio'] = float(np.inf)
            else:
                features['hrv_vlf_norm'] = 0.0
                features['hrv_lf_norm'] = 0.0
                features['hrv_hf_norm'] = 0.0
                features['hrv_lf_hf_ratio'] = 0.0
            
            # Peak frequencies
            if np.any(lf_mask) and np.sum(psd[lf_mask]) > 0:
                lf_peak_idx = np.argmax(psd[lf_mask])
                features['hrv_lf_peak_freq'] = float(freqs[lf_mask][lf_peak_idx])
            else:
                features['hrv_lf_peak_freq'] = 0.0
            
            if np.any(hf_mask) and np.sum(psd[hf_mask]) > 0:
                hf_peak_idx = np.argmax(psd[hf_mask])
                features['hrv_hf_peak_freq'] = float(freqs[hf_mask][hf_peak_idx])
            else:
                features['hrv_hf_peak_freq'] = 0.0
            
        except Exception as e:
            print(f"Warning: HRV frequency analysis failed: {e}")
            return self._get_default_hrv_features()
        
        return features
    
    def _spectral_rolloff(self, freqs: np.ndarray, psd: np.ndarray, 
                         threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency"""
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
        
        if total_power == 0:
            return freqs[-1]
        
        rolloff_idx = np.where(cumulative_power >= threshold * total_power)[0]
        return freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    def _spectral_bandwidth(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Calculate spectral bandwidth"""
        if np.sum(psd) == 0:
            return 0.0
        
        centroid = np.sum(freqs * psd) / np.sum(psd)
        return np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd))
    
    def _spectral_flatness(self, psd: np.ndarray) -> float:
        """Calculate spectral flatness (Wiener entropy)"""
        if len(psd) == 0 or np.sum(psd) == 0:
            return 0.0
        
        # Avoid log(0) by adding small epsilon
        psd_safe = psd + 1e-10
        geometric_mean = np.exp(np.mean(np.log(psd_safe)))
        arithmetic_mean = np.mean(psd)
        
        if arithmetic_mean == 0:
            return 0.0
        
        return geometric_mean / arithmetic_mean
    
    def _extract_advanced_spectral_features(self, signal: np.ndarray, 
                                          lead_name: str) -> Dict[str, float]:
        """Extract additional advanced spectral features"""
        features = {}
        
        try:
            # Spectral edge frequencies
            freqs, psd = periodogram(signal, fs=self.config.sampling_rate)
            
            if len(psd) > 0 and np.sum(psd) > 0:
                # Spectral edge at different percentiles
                for percentile in [50, 75, 90, 95]:
                    edge_freq = self._spectral_edge(freqs, psd, percentile / 100)
                    features[f'{lead_name}_spectral_edge_{percentile}'] = float(edge_freq)
                
                # Spectral slope
                # Fit log-log linear regression to estimate 1/f slope
                valid_idx = (freqs > 0) & (psd > 0)
                if np.sum(valid_idx) > 10:
                    log_freqs = np.log10(freqs[valid_idx])
                    log_psd = np.log10(psd[valid_idx])
                    slope, _ = np.polyfit(log_freqs, log_psd, 1)
                    features[f'{lead_name}_spectral_slope'] = float(slope)
                else:
                    features[f'{lead_name}_spectral_slope'] = 0.0
                
                # Spectral crest factor
                max_psd = np.max(psd)
                mean_psd = np.mean(psd)
                if mean_psd > 0:
                    features[f'{lead_name}_spectral_crest'] = float(max_psd / mean_psd)
                else:
                    features[f'{lead_name}_spectral_crest'] = 0.0
                
            else:
                # Default values
                for percentile in [50, 75, 90, 95]:
                    features[f'{lead_name}_spectral_edge_{percentile}'] = 0.0
                features[f'{lead_name}_spectral_slope'] = 0.0
                features[f'{lead_name}_spectral_crest'] = 0.0
                
        except Exception as e:
            print(f"Warning: Advanced spectral features failed: {e}")
            # Return empty features on error
            
        return features
    
    def _spectral_edge(self, freqs: np.ndarray, psd: np.ndarray, 
                      percentile: float) -> float:
        """Calculate spectral edge frequency at given percentile"""
        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
        
        if total_power == 0:
            return 0.0
        
        edge_idx = np.where(cumulative_power >= percentile * total_power)[0]
        return freqs[edge_idx[0]] if len(edge_idx) > 0 else freqs[-1]
    
    def _get_default_hrv_features(self) -> Dict[str, float]:
        """Return default HRV features when calculation fails"""
        return {
            'hrv_vlf_power': 0.0,
            'hrv_lf_power': 0.0,
            'hrv_hf_power': 0.0,
            'hrv_total_power': 0.0,
            'hrv_vlf_norm': 0.0,
            'hrv_lf_norm': 0.0,
            'hrv_hf_norm': 0.0,
            'hrv_lf_hf_ratio': 0.0,
            'hrv_lf_peak_freq': 0.0,
            'hrv_hf_peak_freq': 0.0
        }