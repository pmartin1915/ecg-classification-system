"""
Wavelet-based feature extraction for ECG signals
"""
import numpy as np
import pywt
from scipy.stats import entropy
from typing import Dict, List, Optional

from config.feature_config import FeatureExtractionConfig


class WaveletFeatureExtractor:
    """Extract wavelet-based features from ECG signals"""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
    
    def extract_wavelet_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet transform features
        
        Args:
            signal: ECG signal (time_points, leads)
            
        Returns:
            Dictionary of wavelet features
        """
        features = {}
        
        # Use Lead II for detailed wavelet analysis
        lead_ii = signal[:, 1]
        
        # Continuous Wavelet Transform features
        cwt_features = self._extract_cwt_features(lead_ii)
        features.update(cwt_features)
        
        # Discrete Wavelet Transform features
        dwt_features = self._extract_dwt_features(lead_ii)
        features.update(dwt_features)
        
        # Multi-lead wavelet features (simplified)
        multi_lead_features = self._extract_multi_lead_wavelet_features(signal)
        features.update(multi_lead_features)
        
        return features
    
    def _extract_cwt_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract Continuous Wavelet Transform features"""
        features = {}
        
        try:
            # Perform CWT
            coefficients, frequencies = pywt.cwt(
                signal, 
                self.config.wavelet_scales, 
                self.config.wavelet_type,
                sampling_period=1.0/self.config.sampling_rate
            )
            
            # Energy at different scales
            scale_energies = np.sum(coefficients**2, axis=1)
            
            # Features for specific scales
            for i, scale in enumerate(self.config.wavelet_scales[:10]):  # First 10 scales
                if i < len(scale_energies):
                    features[f'cwt_energy_scale_{scale}'] = float(scale_energies[i])
                    features[f'cwt_max_coeff_scale_{scale}'] = float(np.max(np.abs(coefficients[i])))
            
            # Total wavelet energy
            total_energy = np.sum(coefficients**2)
            features['cwt_total_energy'] = float(total_energy)
            
            # Wavelet entropy
            if total_energy > 0:
                energy_distribution = scale_energies / total_energy
                features['cwt_entropy'] = float(entropy(energy_distribution))
            else:
                features['cwt_entropy'] = 0.0
            
            # Dominant scale
            dominant_scale_idx = np.argmax(scale_energies)
            features['cwt_dominant_scale'] = float(self.config.wavelet_scales[dominant_scale_idx])
            features['cwt_dominant_frequency'] = float(frequencies[dominant_scale_idx])
            
            # Scale with maximum coefficient
            max_coeff_scale_idx = np.unravel_index(
                np.argmax(np.abs(coefficients)), 
                coefficients.shape
            )[0]
            features['cwt_max_coeff_scale_global'] = float(self.config.wavelet_scales[max_coeff_scale_idx])
            
            # Energy concentration
            cumsum_energy = np.cumsum(np.sort(scale_energies)[::-1])
            if total_energy > 0:
                energy_90_scales = np.sum(cumsum_energy < 0.9 * total_energy) + 1
                features['cwt_energy_concentration'] = float(energy_90_scales / len(scale_energies))
            else:
                features['cwt_energy_concentration'] = 0.0
            
        except Exception as e:
            print(f"Warning: CWT feature extraction failed: {e}")
            # Return default features
            for scale in self.config.wavelet_scales[:10]:
                features[f'cwt_energy_scale_{scale}'] = 0.0
                features[f'cwt_max_coeff_scale_{scale}'] = 0.0
            features.update({
                'cwt_total_energy': 0.0,
                'cwt_entropy': 0.0,
                'cwt_dominant_scale': 0.0,
                'cwt_dominant_frequency': 0.0,
                'cwt_max_coeff_scale_global': 0.0,
                'cwt_energy_concentration': 0.0
            })
        
        return features
    
    def _extract_dwt_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract Discrete Wavelet Transform features"""
        features = {}
        
        try:
            # Perform DWT
            coeffs = pywt.wavedec(signal, self.config.dwt_wavelet, level=self.config.dwt_level)
            
            # Features for each decomposition level
            for i, coeff in enumerate(coeffs):
                # Energy
                energy = np.sum(coeff**2)
                features[f'dwt_energy_level_{i}'] = float(energy)
                
                # Statistical features
                features[f'dwt_mean_level_{i}'] = float(np.mean(coeff))
                features[f'dwt_std_level_{i}'] = float(np.std(coeff))
                features[f'dwt_max_level_{i}'] = float(np.max(np.abs(coeff)))
                features[f'dwt_rms_level_{i}'] = float(np.sqrt(np.mean(coeff**2)))
                
                # Entropy
                if len(coeff) > 0 and np.sum(np.abs(coeff)) > 0:
                    coeff_normalized = np.abs(coeff) / np.sum(np.abs(coeff))
                    features[f'dwt_entropy_level_{i}'] = float(entropy(coeff_normalized))
                else:
                    features[f'dwt_entropy_level_{i}'] = 0.0
            
            # Total energy across all levels
            total_energy = sum(np.sum(c**2) for c in coeffs)
            features['dwt_total_energy'] = float(total_energy)
            
            # Energy distribution across levels
            if total_energy > 0:
                for i, coeff in enumerate(coeffs):
                    level_energy = np.sum(coeff**2)
                    features[f'dwt_energy_ratio_level_{i}'] = float(level_energy / total_energy)
            else:
                for i in range(len(coeffs)):
                    features[f'dwt_energy_ratio_level_{i}'] = 0.0
            
            # Reconstruct signals at different levels for morphological analysis
            # Approximation at level 3 (captures QRS complexes)
            if len(coeffs) > 3:
                approx_3 = pywt.waverec(coeffs[:4] + [np.zeros_like(c) for c in coeffs[4:]], 
                                      self.config.dwt_wavelet)
                if len(approx_3) > len(signal):
                    approx_3 = approx_3[:len(signal)]
                features['dwt_qrs_energy'] = float(np.sum(approx_3**2))
            
            # Detail at levels 4-5 (captures P and T waves)
            if len(coeffs) > 5:
                detail_pt = pywt.waverec([np.zeros_like(coeffs[0])] + 
                                       [np.zeros_like(coeffs[1])] +
                                       [np.zeros_like(coeffs[2])] +
                                       [np.zeros_like(coeffs[3])] +
                                       coeffs[4:6] +
                                       [np.zeros_like(c) for c in coeffs[6:]], 
                                       self.config.dwt_wavelet)
                if len(detail_pt) > len(signal):
                    detail_pt = detail_pt[:len(signal)]
                features['dwt_pt_wave_energy'] = float(np.sum(detail_pt**2))
            
        except Exception as e:
            print(f"Warning: DWT feature extraction failed: {e}")
            # Return default features
            for i in range(self.config.dwt_level + 1):
                features[f'dwt_energy_level_{i}'] = 0.0
                features[f'dwt_mean_level_{i}'] = 0.0
                features[f'dwt_std_level_{i}'] = 0.0
                features[f'dwt_max_level_{i}'] = 0.0
                features[f'dwt_rms_level_{i}'] = 0.0
                features[f'dwt_entropy_level_{i}'] = 0.0
                features[f'dwt_energy_ratio_level_{i}'] = 0.0
            features['dwt_total_energy'] = 0.0
            features['dwt_qrs_energy'] = 0.0
            features['dwt_pt_wave_energy'] = 0.0
        
        return features
    
    def _extract_multi_lead_wavelet_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract simplified wavelet features from multiple leads"""
        features = {}
        
        # Extract energy features from key leads
        key_leads = {
            'V1': 6,   # Septal activity
            'V5': 10,  # Lateral wall
            'aVF': 5   # Inferior wall
        }
        
        for lead_name, lead_idx in key_leads.items():
            if lead_idx < signal.shape[1]:
                try:
                    # Simple DWT for computational efficiency
                    coeffs = pywt.wavedec(signal[:, lead_idx], 'db4', level=3)
                    
                    # Energy in QRS band (level 2-3)
                    qrs_energy = np.sum(coeffs[2]**2) + np.sum(coeffs[3]**2)
                    features[f'{lead_name}_wavelet_qrs_energy'] = float(qrs_energy)
                    
                    # Energy in baseline/ST band (level 4+)
                    if len(coeffs) > 4:
                        st_energy = sum(np.sum(c**2) for c in coeffs[4:])
                        features[f'{lead_name}_wavelet_st_energy'] = float(st_energy)
                    else:
                        features[f'{lead_name}_wavelet_st_energy'] = 0.0
                    
                except Exception:
                    features[f'{lead_name}_wavelet_qrs_energy'] = 0.0
                    features[f'{lead_name}_wavelet_st_energy'] = 0.0
        
        return features