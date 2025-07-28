"""
Signal filtering utilities for ECG preprocessing
"""
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from typing import Dict, Tuple, Optional, List

from config.preprocessing_config import PreprocessingConfig


class ECGFilterBank:
    """Manages filters for ECG signal processing"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.filters = self._create_filters()
        
    def _create_filters(self) -> Dict:
        """Create all necessary filters"""
        filters = {}
        nyquist = self.config.sampling_rate / 2
        
        # High-pass filter (remove baseline wander)
        if self.config.highpass_freq > 0:
            normalized_freq = self.config.highpass_freq / nyquist
            filters['highpass'] = butter(
                self.config.filter_order, 
                normalized_freq, 
                btype='high', 
                output='ba'
            )
        
        # Low-pass filter (remove high-frequency noise)
        if self.config.lowpass_freq > 0:
            normalized_freq = self.config.lowpass_freq / nyquist
            filters['lowpass'] = butter(
                self.config.filter_order, 
                normalized_freq, 
                btype='low', 
                output='ba'
            )
        
        # Notch filter (remove powerline interference)
        if self.config.notch_freq > 0:
            normalized_freq = self.config.notch_freq / nyquist
            filters['notch'] = iirnotch(
                normalized_freq, 
                self.config.notch_quality
            )
        
        return filters
    
    def apply_filters(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Apply all filters to ECG signal
        
        Args:
            ecg_signal: Input signal (time_points, leads)
            
        Returns:
            Filtered signal
        """
        filtered_signal = ecg_signal.copy()
        
        # Ensure 2D array
        if filtered_signal.ndim == 1:
            filtered_signal = filtered_signal.reshape(-1, 1)
        
        # Apply filters in sequence
        for filter_name, filter_coeffs in self.filters.items():
            b, a = filter_coeffs
            # Apply filter to each lead
            for lead_idx in range(filtered_signal.shape[1]):
                try:
                    filtered_signal[:, lead_idx] = filtfilt(
                        b, a, filtered_signal[:, lead_idx]
                    )
                except Exception as e:
                    print(f"Warning: Filter {filter_name} failed on lead {lead_idx}: {e}")
        
        return filtered_signal
    
    def get_filter_response(self, filter_name: str = None) -> Dict:
        """Get frequency response of filters"""
        responses = {}
        
        filters_to_check = (
            {filter_name: self.filters[filter_name]} 
            if filter_name and filter_name in self.filters 
            else self.filters
        )
        
        for name, coeffs in filters_to_check.items():
            b, a = coeffs
            w, h = signal.freqz(b, a, worN=8000)
            frequencies = w * self.config.sampling_rate / (2 * np.pi)
            magnitude_db = 20 * np.log10(np.abs(h))
            phase_deg = np.angle(h, deg=True)
            
            responses[name] = {
                'frequencies': frequencies,
                'magnitude_db': magnitude_db,
                'phase_deg': phase_deg
            }
        
        return responses


class ArtifactDetector:
    """Detect and handle artifacts in ECG signals"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
    def detect_artifacts(self, signal: np.ndarray) -> Dict[str, List]:
        """
        Detect various types of artifacts in ECG signal
        
        Args:
            signal: ECG signal (time_points, leads)
            
        Returns:
            Dictionary of detected artifacts by type
        """
        artifacts = {
            'amplitude': [],
            'gradient': [],
            'saturation': [],
            'baseline_drift': []
        }
        
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        for lead_idx in range(signal.shape[1]):
            lead_signal = signal[:, lead_idx]
            
            # Amplitude-based artifact detection
            amplitude_artifacts = self._detect_amplitude_artifacts(lead_signal)
            if amplitude_artifacts.any():
                artifacts['amplitude'].append({
                    'lead': lead_idx,
                    'indices': np.where(amplitude_artifacts)[0]
                })
            
            # Gradient-based artifact detection
            gradient_artifacts = self._detect_gradient_artifacts(lead_signal)
            if gradient_artifacts.any():
                artifacts['gradient'].append({
                    'lead': lead_idx,
                    'indices': np.where(gradient_artifacts)[0]
                })
            
            # Saturation detection
            saturation_artifacts = self._detect_saturation(lead_signal)
            if saturation_artifacts.any():
                artifacts['saturation'].append({
                    'lead': lead_idx,
                    'indices': np.where(saturation_artifacts)[0]
                })
            
            # Baseline drift detection
            drift_artifacts = self._detect_baseline_drift(lead_signal)
            if drift_artifacts.any():
                artifacts['baseline_drift'].append({
                    'lead': lead_idx,
                    'indices': np.where(drift_artifacts)[0]
                })
        
        return artifacts
    
    def _detect_amplitude_artifacts(self, lead_signal: np.ndarray) -> np.ndarray:
        """Detect amplitude-based artifacts"""
        return np.abs(lead_signal) > self.config.amplitude_threshold
    
    def _detect_gradient_artifacts(self, lead_signal: np.ndarray) -> np.ndarray:
        """Detect gradient-based artifacts (sudden changes)"""
        gradient = np.diff(lead_signal, prepend=lead_signal[0])
        return np.abs(gradient) > self.config.gradient_threshold
    
    def _detect_saturation(self, lead_signal: np.ndarray) -> np.ndarray:
        """Detect signal saturation/clipping"""
        signal_max = np.max(lead_signal)
        signal_min = np.min(lead_signal)
        
        saturation_high = lead_signal > (signal_max * self.config.saturation_threshold)
        saturation_low = lead_signal < (signal_min * self.config.saturation_threshold)
        
        return saturation_high | saturation_low
    
    def _detect_baseline_drift(self, lead_signal: np.ndarray, window_size: int = 500) -> np.ndarray:
        """Detect baseline drift using moving average"""
        if len(lead_signal) < window_size:
            return np.zeros_like(lead_signal, dtype=bool)
        
        # Calculate moving average
        kernel = np.ones(window_size) / window_size
        baseline = np.convolve(lead_signal, kernel, mode='same')
        
        # Detect significant deviations from zero baseline
        drift_threshold = 0.5  # mV
        return np.abs(baseline) > drift_threshold
    
    def remove_artifacts(self, 
                        signal: np.ndarray, 
                        artifacts: Dict[str, List],
                        method: str = 'interpolation') -> np.ndarray:
        """
        Remove or correct detected artifacts
        
        Args:
            signal: ECG signal
            artifacts: Detected artifacts
            method: 'interpolation', 'clip', or 'zero'
            
        Returns:
            Cleaned signal
        """
        cleaned_signal = signal.copy()
        
        if signal.ndim == 1:
            cleaned_signal = cleaned_signal.reshape(-1, 1)
        
        for artifact_type, artifact_list in artifacts.items():
            for artifact_info in artifact_list:
                lead_idx = artifact_info['lead']
                indices = artifact_info['indices']
                
                if len(indices) == 0:
                    continue
                
                if method == 'interpolation':
                    # Interpolate over artifact regions
                    cleaned_signal[indices, lead_idx] = np.interp(
                        indices,
                        np.setdiff1d(np.arange(len(signal)), indices),
                        cleaned_signal[np.setdiff1d(np.arange(len(signal)), indices), lead_idx]
                    )
                elif method == 'clip':
                    # Clip to percentile values
                    percentile_low = (100 - self.config.clip_percentile) / 2
                    percentile_high = 100 - percentile_low
                    
                    low_val = np.percentile(cleaned_signal[:, lead_idx], percentile_low)
                    high_val = np.percentile(cleaned_signal[:, lead_idx], percentile_high)
                    
                    cleaned_signal[indices, lead_idx] = np.clip(
                        cleaned_signal[indices, lead_idx], low_val, high_val
                    )
                elif method == 'zero':
                    # Set to zero
                    cleaned_signal[indices, lead_idx] = 0
        
        return cleaned_signal