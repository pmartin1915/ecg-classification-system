"""
ST-segment feature extraction for MI detection
"""
import numpy as np
from typing import Dict, List, Optional

from config.feature_config import FeatureExtractionConfig


class STSegmentAnalyzer:
    """Extract ST-segment features for MI detection"""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        
        # Define lead groups for MI localization
        self.lead_groups = {
            'anterior': ['V1', 'V2', 'V3', 'V4'],
            'inferior': ['II', 'III', 'aVF'],
            'lateral': ['I', 'aVL', 'V5', 'V6'],
            'septal': ['V1', 'V2'],
            'extensive_anterior': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        }
    
    def extract_st_features(self, signal: np.ndarray, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        Extract ST-segment features from ECG signal
        
        Args:
            signal: ECG signal (time_points, leads)
            r_peaks: R-peak locations
            
        Returns:
            Dictionary of ST-segment features
        """
        features = {}
        
        if len(r_peaks) == 0:
            return self._get_default_st_features()
        
        # Analyze ST-segment in each lead
        lead_st_features = {}
        
        for lead_idx in range(signal.shape[1]):
            lead_signal = signal[:, lead_idx]
            lead_name = self.config.lead_names[lead_idx]
            
            # Extract ST measurements for this lead
            st_measurements = self._analyze_lead_st_segments(lead_signal, r_peaks)
            
            # Store lead-specific features
            lead_st_features[lead_name] = st_measurements
            
            # Add to global features
            features.update(self._create_lead_features(lead_name, st_measurements))
        
        # Add group-based features for MI localization
        features.update(self._create_group_features(lead_st_features))
        
        # Add diagnostic criteria features
        features.update(self._evaluate_mi_criteria(lead_st_features))
        
        return features
    
    def _analyze_lead_st_segments(self, lead_signal: np.ndarray, 
                                 r_peaks: np.ndarray) -> Dict[str, List[float]]:
        """Analyze ST segments for a single lead"""
        measurements = {
            'elevations': [],
            'depressions': [],
            'slopes': [],
            'j_points': [],
            'areas': []
        }
        
        for r_peak in r_peaks:
            # Skip if too close to signal boundaries
            if r_peak < 20 or r_peak >= len(lead_signal) - 20:
                continue
            
            try:
                # Extract ST segment measurements
                st_data = self._extract_single_st_segment(lead_signal, r_peak)
                
                measurements['elevations'].append(st_data['elevation'])
                measurements['depressions'].append(st_data['depression'])
                measurements['slopes'].append(st_data['slope'])
                measurements['j_points'].append(st_data['j_point'])
                measurements['areas'].append(st_data['area'])
                
            except Exception as e:
                # Skip problematic beats
                continue
        
        return measurements
    
    def _extract_single_st_segment(self, lead_signal: np.ndarray, 
                                  r_peak: int) -> Dict[str, float]:
        """Extract measurements from a single ST segment"""
        # Define measurement points
        baseline_start = max(0, r_peak - int(self.config.st_reference_offset * self.config.sampling_rate))
        baseline_end = max(0, r_peak - int(0.02 * self.config.sampling_rate))
        
        j_point = min(len(lead_signal) - 1, r_peak + int(0.04 * self.config.sampling_rate))
        st_start = min(len(lead_signal) - 1, r_peak + int(self.config.st_start_offset * self.config.sampling_rate))
        st_end = min(len(lead_signal) - 1, r_peak + int(self.config.st_end_offset * self.config.sampling_rate))
        
        # Calculate baseline
        baseline_segment = lead_signal[baseline_start:baseline_end]
        if len(baseline_segment) > 0:
            baseline = np.median(baseline_segment)
        else:
            baseline = lead_signal[r_peak]
        
        # J-point measurement
        j_point_value = lead_signal[j_point] - baseline
        
        # ST segment measurements
        st_segment = lead_signal[st_start:st_end]
        
        if len(st_segment) > 0:
            # ST level (measured at 60-80ms after J point)
            st_level = np.mean(st_segment) - baseline
            
            # ST elevation and depression
            elevation = max(0, st_level)
            depression = min(0, st_level)
            
            # ST slope
            if len(st_segment) > 1:
                x = np.arange(len(st_segment))
                slope = np.polyfit(x, st_segment, 1)[0] * self.config.sampling_rate  # mV/s
            else:
                slope = 0.0
            
            # ST area (integral)
            area = np.trapz(st_segment - baseline) / self.config.sampling_rate
            
        else:
            elevation = 0.0
            depression = 0.0
            slope = 0.0
            area = 0.0
        
        return {
            'elevation': elevation,
            'depression': depression,
            'slope': slope,
            'j_point': j_point_value,
            'area': area
        }
    
    def _create_lead_features(self, lead_name: str, 
                            measurements: Dict[str, List[float]]) -> Dict[str, float]:
        """Create features for a single lead"""
        features = {}
        
        for measurement_type, values in measurements.items():
            if values:
                # Basic statistics
                features[f'{lead_name}_st_{measurement_type}_mean'] = float(np.mean(values))
                features[f'{lead_name}_st_{measurement_type}_std'] = float(np.std(values))
                features[f'{lead_name}_st_{measurement_type}_max'] = float(np.max(values))
                features[f'{lead_name}_st_{measurement_type}_min'] = float(np.min(values))
                
                # Additional statistics for key measurements
                if measurement_type in ['elevations', 'depressions']:
                    features[f'{lead_name}_st_{measurement_type}_median'] = float(np.median(values))
                    
                    # Percentage of beats with significant changes
                    if measurement_type == 'elevations':
                        significant = np.sum(np.array(values) > 0.1)  # >0.1mV elevation
                    else:
                        significant = np.sum(np.array(values) < -0.05)  # >0.05mV depression
                    
                    features[f'{lead_name}_st_{measurement_type}_significant_ratio'] = float(significant / len(values))
            
            else:
                # No valid measurements
                features[f'{lead_name}_st_{measurement_type}_mean'] = 0.0
                features[f'{lead_name}_st_{measurement_type}_std'] = 0.0
                features[f'{lead_name}_st_{measurement_type}_max'] = 0.0
                features[f'{lead_name}_st_{measurement_type}_min'] = 0.0
                
                if measurement_type in ['elevations', 'depressions']:
                    features[f'{lead_name}_st_{measurement_type}_median'] = 0.0
                    features[f'{lead_name}_st_{measurement_type}_significant_ratio'] = 0.0
        
        return features
    
    def _create_group_features(self, lead_st_features: Dict[str, Dict]) -> Dict[str, float]:
        """Create features for lead groups (MI localization)"""
        features = {}
        
        for group_name, lead_names in self.lead_groups.items():
            group_elevations = []
            group_depressions = []
            
            for lead_name in lead_names:
                if lead_name in lead_st_features:
                    measurements = lead_st_features[lead_name]
                    if 'elevations' in measurements and measurements['elevations']:
                        group_elevations.extend(measurements['elevations'])
                    if 'depressions' in measurements and measurements['depressions']:
                        group_depressions.extend(measurements['depressions'])
            
            # Group statistics
            if group_elevations:
                features[f'{group_name}_st_elevation_mean'] = float(np.mean(group_elevations))
                features[f'{group_name}_st_elevation_max'] = float(np.max(group_elevations))
                features[f'{group_name}_st_elevation_sum'] = float(np.sum(group_elevations))
            else:
                features[f'{group_name}_st_elevation_mean'] = 0.0
                features[f'{group_name}_st_elevation_max'] = 0.0
                features[f'{group_name}_st_elevation_sum'] = 0.0
            
            if group_depressions:
                features[f'{group_name}_st_depression_mean'] = float(np.mean(group_depressions))
                features[f'{group_name}_st_depression_min'] = float(np.min(group_depressions))
            else:
                features[f'{group_name}_st_depression_mean'] = 0.0
                features[f'{group_name}_st_depression_min'] = 0.0
        
        return features
    
    def _evaluate_mi_criteria(self, lead_st_features: Dict[str, Dict]) -> Dict[str, float]:
        """Evaluate clinical criteria for MI detection"""
        features = {}
        
        # STEMI criteria
        # Anterior MI: ST elevation in V1-V4
        anterior_criteria = self._check_stemi_criteria(
            lead_st_features, 
            ['V1', 'V2', 'V3', 'V4'], 
            elevation_threshold=0.2  # 2mm
        )
        features['anterior_stemi_score'] = float(anterior_criteria)
        
        # Inferior MI: ST elevation in II, III, aVF
        inferior_criteria = self._check_stemi_criteria(
            lead_st_features,
            ['II', 'III', 'aVF'],
            elevation_threshold=0.1  # 1mm
        )
        features['inferior_stemi_score'] = float(inferior_criteria)
        
        # Lateral MI: ST elevation in I, aVL, V5, V6
        lateral_criteria = self._check_stemi_criteria(
            lead_st_features,
            ['I', 'aVL', 'V5', 'V6'],
            elevation_threshold=0.1  # 1mm
        )
        features['lateral_stemi_score'] = float(lateral_criteria)
        
        # Reciprocal changes
        # Inferior MI often has reciprocal depression in I, aVL
        if inferior_criteria > 0:
            reciprocal_score = self._check_reciprocal_changes(
                lead_st_features,
                ['I', 'aVL'],
                depression_threshold=-0.05
            )
            features['inferior_reciprocal_score'] = float(reciprocal_score)
        else:
            features['inferior_reciprocal_score'] = 0.0
        
        # Overall MI probability score
        features['mi_probability_score'] = float(
            max(anterior_criteria, inferior_criteria, lateral_criteria)
        )
        
        return features
    
    def _check_stemi_criteria(self, lead_st_features: Dict[str, Dict],
                            lead_names: List[str],
                            elevation_threshold: float) -> float:
        """Check STEMI criteria for given leads"""
        elevated_leads = 0
        total_elevation = 0.0
        
        for lead_name in lead_names:
            if lead_name in lead_st_features:
                measurements = lead_st_features[lead_name]
                if 'elevations' in measurements and measurements['elevations']:
                    max_elevation = np.max(measurements['elevations'])
                    if max_elevation > elevation_threshold:
                        elevated_leads += 1
                        total_elevation += max_elevation
        
        # Score based on number of elevated leads and total elevation
        if elevated_leads >= 2:  # At least 2 contiguous leads
            return min(1.0, (elevated_leads / len(lead_names)) * (total_elevation / (elevation_threshold * len(lead_names))))
        else:
            return 0.0
    
    def _check_reciprocal_changes(self, lead_st_features: Dict[str, Dict],
                                 lead_names: List[str],
                                 depression_threshold: float) -> float:
        """Check for reciprocal ST depression"""
        depressed_leads = 0
        
        for lead_name in lead_names:
            if lead_name in lead_st_features:
                measurements = lead_st_features[lead_name]
                if 'depressions' in measurements and measurements['depressions']:
                    min_depression = np.min(measurements['depressions'])
                    if min_depression < depression_threshold:
                        depressed_leads += 1
        
        return depressed_leads / len(lead_names)
    
    def _get_default_st_features(self) -> Dict[str, float]:
        """Return default ST features when no R-peaks available"""
        features = {}
        
        # Lead-specific features
        for lead_name in self.config.lead_names:
            for measurement in ['elevations', 'depressions', 'slopes', 'j_points', 'areas']:
                features[f'{lead_name}_st_{measurement}_mean'] = 0.0
                features[f'{lead_name}_st_{measurement}_std'] = 0.0
                features[f'{lead_name}_st_{measurement}_max'] = 0.0
                features[f'{lead_name}_st_{measurement}_min'] = 0.0
        
        # Group features
        for group_name in self.lead_groups:
            features[f'{group_name}_st_elevation_mean'] = 0.0
            features[f'{group_name}_st_elevation_max'] = 0.0
            features[f'{group_name}_st_elevation_sum'] = 0.0
            features[f'{group_name}_st_depression_mean'] = 0.0
            features[f'{group_name}_st_depression_min'] = 0.0
        
        # Criteria features
        features['anterior_stemi_score'] = 0.0
        features['inferior_stemi_score'] = 0.0
        features['lateral_stemi_score'] = 0.0
        features['inferior_reciprocal_score'] = 0.0
        features['mi_probability_score'] = 0.0
        
        return features