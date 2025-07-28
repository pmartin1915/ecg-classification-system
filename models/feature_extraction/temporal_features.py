"""
Temporal domain feature extraction for ECG signals
"""
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from typing import Dict, Tuple, List, Optional

from config.feature_config import FeatureExtractionConfig


class TemporalFeatureExtractor:
    """Extract temporal domain features from ECG signals"""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
    
    def extract_statistical_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from all leads
        
        Args:
            signal: ECG signal (time_points, leads)
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        for lead_idx in range(signal.shape[1]):
            lead_signal = signal[:, lead_idx]
            lead_name = self.config.lead_names[lead_idx]
            
            # Basic statistics
            features[f'{lead_name}_mean'] = float(np.mean(lead_signal))
            features[f'{lead_name}_std'] = float(np.std(lead_signal))
            features[f'{lead_name}_var'] = float(np.var(lead_signal))
            features[f'{lead_name}_median'] = float(np.median(lead_signal))
            features[f'{lead_name}_min'] = float(np.min(lead_signal))
            features[f'{lead_name}_max'] = float(np.max(lead_signal))
            features[f'{lead_name}_range'] = float(np.ptp(lead_signal))
            features[f'{lead_name}_rms'] = float(np.sqrt(np.mean(lead_signal**2)))
            
            # Higher-order statistics
            features[f'{lead_name}_skewness'] = float(skew(lead_signal))
            features[f'{lead_name}_kurtosis'] = float(kurtosis(lead_signal))
            
            # Percentiles
            p25, p75 = np.percentile(lead_signal, [25, 75])
            features[f'{lead_name}_p25'] = float(p25)
            features[f'{lead_name}_p75'] = float(p75)
            features[f'{lead_name}_iqr'] = float(p75 - p25)
            
            # Energy and power
            features[f'{lead_name}_energy'] = float(np.sum(lead_signal**2))
            features[f'{lead_name}_power'] = float(np.mean(lead_signal**2))
            
            # Zero crossings
            zero_crossings = np.sum(np.diff(np.sign(lead_signal)) != 0)
            features[f'{lead_name}_zero_crossings'] = float(zero_crossings)
            
            # Additional temporal features
            features[f'{lead_name}_abs_mean'] = float(np.mean(np.abs(lead_signal)))
            features[f'{lead_name}_abs_energy'] = float(np.sum(np.abs(lead_signal)))
            
            # Signal mobility and complexity
            diff1 = np.diff(lead_signal)
            diff2 = np.diff(diff1)
            
            if len(diff1) > 0:
                mobility = np.std(diff1) / (np.std(lead_signal) + 1e-8)
                features[f'{lead_name}_mobility'] = float(mobility)
                
                if len(diff2) > 0:
                    complexity = (np.std(diff2) / (np.std(diff1) + 1e-8)) / (mobility + 1e-8)
                    features[f'{lead_name}_complexity'] = float(complexity)
        
        return features
    
    def extract_morphological_features(self, signal: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Extract morphological features using R-peak detection
        
        Args:
            signal: ECG signal (time_points, leads)
            
        Returns:
            Tuple of (features dict, r_peaks array)
        """
        features = {}
        
        # Use Lead II for R-peak detection (most reliable)
        lead_ii = signal[:, 1]
        
        # Detect R-peaks
        r_peaks, properties = find_peaks(
            lead_ii,
            height=self.config.r_peak_height,
            distance=self.config.r_peak_distance,
            prominence=self.config.r_peak_prominence
        )
        
        # Basic R-peak statistics
        features['r_peak_count'] = float(len(r_peaks))
        features['r_peak_density'] = float(len(r_peaks) / (len(lead_ii) / self.config.sampling_rate))
        
        if len(r_peaks) > 1:
            # RR intervals
            rr_intervals = np.diff(r_peaks) / self.config.sampling_rate
            
            # Filter physiologically plausible RR intervals
            valid_rr = rr_intervals[
                (rr_intervals >= self.config.hrv_min_rr) & 
                (rr_intervals <= self.config.hrv_max_rr)
            ]
            
            if len(valid_rr) > 0:
                # RR interval statistics
                features['rr_mean'] = float(np.mean(valid_rr))
                features['rr_std'] = float(np.std(valid_rr))
                features['rr_min'] = float(np.min(valid_rr))
                features['rr_max'] = float(np.max(valid_rr))
                features['rr_range'] = float(np.ptp(valid_rr))
                
                # Coefficient of variation
                if np.mean(valid_rr) > 0:
                    features['rr_cv'] = float(np.std(valid_rr) / np.mean(valid_rr))
                else:
                    features['rr_cv'] = 0.0
                
                # Heart rate statistics
                hr_values = 60 / valid_rr
                features['heart_rate_mean'] = float(np.mean(hr_values))
                features['heart_rate_std'] = float(np.std(hr_values))
                features['heart_rate_min'] = float(np.min(hr_values))
                features['heart_rate_max'] = float(np.max(hr_values))
                
                # HRV time-domain features
                if len(valid_rr) > 1:
                    rr_diff = np.diff(valid_rr)
                    features['rr_diff_mean'] = float(np.mean(rr_diff))
                    features['rr_diff_std'] = float(np.std(rr_diff))
                    features['rr_diff_rms'] = float(np.sqrt(np.mean(rr_diff**2)))
                    
                    # pNN50: percentage of successive RR intervals differing by >50ms
                    nn50 = np.sum(np.abs(rr_diff) > 0.05)
                    features['pnn50'] = float(nn50 / len(rr_diff) * 100)
                    
                    # pNN20: percentage of successive RR intervals differing by >20ms
                    nn20 = np.sum(np.abs(rr_diff) > 0.02)
                    features['pnn20'] = float(nn20 / len(rr_diff) * 100)
                    
                    # RMSSD: root mean square of successive differences
                    features['rmssd'] = float(np.sqrt(np.mean(rr_diff**2)))
                    
                    # SDNN: standard deviation of NN intervals
                    features['sdnn'] = float(np.std(valid_rr))
                    
                    # SDSD: standard deviation of successive differences
                    features['sdsd'] = float(np.std(rr_diff))
        
        # R-peak amplitude analysis
        if len(r_peaks) > 0:
            r_amplitudes = lead_ii[r_peaks]
            features['r_amplitude_mean'] = float(np.mean(r_amplitudes))
            features['r_amplitude_std'] = float(np.std(r_amplitudes))
            features['r_amplitude_min'] = float(np.min(r_amplitudes))
            features['r_amplitude_max'] = float(np.max(r_amplitudes))
            
            if np.mean(r_amplitudes) > 0:
                features['r_amplitude_cv'] = float(np.std(r_amplitudes) / np.mean(r_amplitudes))
            else:
                features['r_amplitude_cv'] = 0.0
        
        # QRS width estimation
        qrs_features = self._extract_qrs_features(lead_ii, r_peaks)
        features.update(qrs_features)
        
        return features, r_peaks
    
    def _extract_qrs_features(self, lead_signal: np.ndarray, r_peaks: np.ndarray) -> Dict[str, float]:
        """Extract QRS complex features"""
        features = {}
        
        if len(r_peaks) == 0:
            return features
        
        qrs_widths = []
        qrs_areas = []
        
        for peak in r_peaks:
            # Define search window around R-peak
            window_before = int(0.06 * self.config.sampling_rate)  # 60ms before
            window_after = int(0.06 * self.config.sampling_rate)   # 60ms after
            
            start_idx = max(0, peak - window_before)
            end_idx = min(len(lead_signal), peak + window_after)
            
            # Find QRS boundaries using derivative method
            qrs_segment = lead_signal[start_idx:end_idx]
            
            if len(qrs_segment) > 10:
                # Calculate derivative
                derivative = np.gradient(qrs_segment)
                
                # Find QRS onset (maximum negative derivative before R-peak)
                peak_relative = peak - start_idx
                if peak_relative > 5:
                    onset_idx = np.argmin(derivative[:peak_relative])
                else:
                    onset_idx = 0
                
                # Find QRS offset (maximum positive derivative after R-peak)
                if peak_relative < len(derivative) - 5:
                    offset_idx = peak_relative + np.argmax(derivative[peak_relative:])
                else:
                    offset_idx = len(derivative) - 1
                
                # Calculate QRS width
                qrs_width = (offset_idx - onset_idx) / self.config.sampling_rate
                
                if 0.04 < qrs_width < self.config.qrs_duration_max:
                    qrs_widths.append(qrs_width)
                    
                    # Calculate QRS area
                    qrs_area = np.trapz(np.abs(qrs_segment[onset_idx:offset_idx]))
                    qrs_areas.append(qrs_area)
        
        if qrs_widths:
            features['qrs_width_mean'] = float(np.mean(qrs_widths))
            features['qrs_width_std'] = float(np.std(qrs_widths))
            features['qrs_width_min'] = float(np.min(qrs_widths))
            features['qrs_width_max'] = float(np.max(qrs_widths))
        
        if qrs_areas:
            features['qrs_area_mean'] = float(np.mean(qrs_areas))
            features['qrs_area_std'] = float(np.std(qrs_areas))
        
        return features
    
    def extract_cross_lead_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract features from relationships between leads"""
        features = {}
        
        # Lead correlations
        for i in range(len(self.config.limb_leads)):
            for j in range(i + 1, len(self.config.limb_leads)):
                lead1_idx = self.config.limb_leads[i]
                lead2_idx = self.config.limb_leads[j]
                lead1_name = self.config.lead_names[lead1_idx]
                lead2_name = self.config.lead_names[lead2_idx]
                
                correlation = np.corrcoef(signal[:, lead1_idx], signal[:, lead2_idx])[0, 1]
                features[f'{lead1_name}_{lead2_name}_correlation'] = float(correlation)
        
        # Electrical axis estimation (simplified)
        lead_i = signal[:, 0]  # Lead I
        lead_avf = signal[:, 5]  # Lead aVF
        
        # Calculate mean QRS vectors
        mean_i = np.mean(lead_i)
        mean_avf = np.mean(lead_avf)
        
        # Estimate electrical axis
        if mean_i != 0:
            axis_angle = np.arctan2(mean_avf, mean_i) * 180 / np.pi
            features['electrical_axis'] = float(axis_angle)
        
        return features