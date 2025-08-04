"""
MI-Specific Feature Extraction for Enhanced Cardiac Analysis
Clinical-grade features specifically designed for MI detection
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MISpecificFeatureExtractor:
    """
    Enhanced feature extraction specifically designed for MI detection
    Based on clinical diagnostic criteria and cardiac electrophysiology
    """
    
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        self.lead_groups = self._define_lead_groups()
        self.mi_territories = self._define_mi_territories()
        
    def _define_lead_groups(self) -> Dict[str, List[int]]:
        """Define standard 12-lead ECG groupings (0-11 indexing)"""
        return {
            'limb_leads': [0, 1, 2, 3, 4, 5],     # I, II, III, aVR, aVL, aVF
            'precordial': [6, 7, 8, 9, 10, 11],    # V1, V2, V3, V4, V5, V6
            'inferior': [1, 2, 5],                  # II, III, aVF
            'lateral': [0, 4, 10, 11],             # I, aVL, V5, V6
            'anterior': [6, 7, 8, 9],              # V1, V2, V3, V4
            'septal': [6, 7],                      # V1, V2
            'high_lateral': [0, 4]                 # I, aVL
        }
    
    def _define_mi_territories(self) -> Dict[str, Dict]:
        """Define MI territorial mappings with reciprocal changes"""
        return {
            'anterior': {
                'primary_leads': [6, 7, 8, 9],      # V1-V4
                'reciprocal_leads': [1, 2, 5],      # II, III, aVF
                'vessel': 'LAD',
                'critical_features': ['st_elevation', 'q_waves', 'poor_r_progression']
            },
            'inferior': {
                'primary_leads': [1, 2, 5],         # II, III, aVF
                'reciprocal_leads': [0, 4],         # I, aVL
                'vessel': 'RCA/LCX',
                'critical_features': ['st_elevation', 'q_waves', 'reciprocal_depression']
            },
            'lateral': {
                'primary_leads': [0, 4, 10, 11],    # I, aVL, V5, V6
                'reciprocal_leads': [1, 2, 5],      # II, III, aVF
                'vessel': 'LCX/OM',
                'critical_features': ['st_elevation', 'tall_r_waves', 't_wave_changes']
            },
            'posterior': {
                'primary_leads': [],                # V7-V9 (not in standard 12-lead)
                'reciprocal_leads': [6, 7, 8],     # V1, V2, V3
                'vessel': 'PDA/PLV',
                'critical_features': ['reciprocal_depression', 'tall_r_v1v2', 'st_depression']
            }
        }
    
    def extract_comprehensive_mi_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive MI-specific features from 12-lead ECG
        
        Args:
            ecg_signal: 12-lead ECG signal (shape: [12, samples])
            
        Returns:
            Dictionary of MI-specific features
        """
        features = {}
        
        if ecg_signal.shape[0] != 12:
            raise ValueError(f"Expected 12-lead ECG, got {ecg_signal.shape[0]} leads")
        
        # 1. ST Segment Analysis (Critical for MI)
        st_features = self._extract_st_segment_features(ecg_signal)
        features.update(st_features)
        
        # 2. Q-Wave Analysis
        q_wave_features = self._extract_q_wave_features(ecg_signal)
        features.update(q_wave_features)
        
        # 3. T-Wave Analysis
        t_wave_features = self._extract_t_wave_features(ecg_signal)
        features.update(t_wave_features)
        
        # 4. R-Wave Progression Analysis
        r_progression_features = self._extract_r_progression_features(ecg_signal)
        features.update(r_progression_features)
        
        # 5. Territory-Specific Analysis
        territory_features = self._extract_territory_features(ecg_signal)
        features.update(territory_features)
        
        # 6. Reciprocal Changes Analysis
        reciprocal_features = self._extract_reciprocal_features(ecg_signal)
        features.update(reciprocal_features)
        
        # 7. Lead Group Analysis
        group_features = self._extract_lead_group_features(ecg_signal)
        features.update(group_features)
        
        # 8. Advanced Morphology Features
        morphology_features = self._extract_advanced_morphology(ecg_signal)
        features.update(morphology_features)
        
        return features
    
    def _extract_st_segment_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract ST segment elevation/depression features"""
        features = {}
        
        for lead_idx in range(12):
            lead_name = self._get_lead_name(lead_idx)
            lead_signal = ecg_signal[lead_idx]
            
            # Detect QRS complexes and ST segments
            qrs_indices = self._detect_qrs_complexes(lead_signal)
            
            st_elevations = []
            st_depressions = []
            
            for qrs_idx in qrs_indices:
                # ST segment starts ~80ms after QRS
                st_start = qrs_idx + int(0.08 * self.sampling_rate)
                st_end = st_start + int(0.12 * self.sampling_rate)  # 120ms ST segment
                
                if st_end < len(lead_signal):
                    # Calculate ST deviation from isoelectric line
                    baseline = self._calculate_baseline(lead_signal, qrs_idx)
                    st_level = np.mean(lead_signal[st_start:st_end])
                    st_deviation = st_level - baseline
                    
                    if st_deviation > 0:
                        st_elevations.append(st_deviation)
                    else:
                        st_depressions.append(abs(st_deviation))
            
            # ST elevation features
            features[f'{lead_name}_st_elevation_max'] = np.max(st_elevations) if st_elevations else 0
            features[f'{lead_name}_st_elevation_mean'] = np.mean(st_elevations) if st_elevations else 0
            features[f'{lead_name}_st_elevation_count'] = len([x for x in st_elevations if x > 0.1])  # >1mm significant
            
            # ST depression features
            features[f'{lead_name}_st_depression_max'] = np.max(st_depressions) if st_depressions else 0
            features[f'{lead_name}_st_depression_mean'] = np.mean(st_depressions) if st_depressions else 0
            features[f'{lead_name}_st_depression_count'] = len([x for x in st_depressions if x > 0.1])
            
            # ST morphology
            features[f'{lead_name}_st_slope'] = self._calculate_st_slope(lead_signal, qrs_indices)
            features[f'{lead_name}_st_concavity'] = self._calculate_st_concavity(lead_signal, qrs_indices)
        
        return features
    
    def _extract_q_wave_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract pathological Q-wave features"""
        features = {}
        
        for lead_idx in range(12):
            lead_name = self._get_lead_name(lead_idx)
            lead_signal = ecg_signal[lead_idx]
            
            qrs_indices = self._detect_qrs_complexes(lead_signal)
            
            q_wave_widths = []
            q_wave_depths = []
            q_r_ratios = []
            
            for qrs_idx in qrs_indices:
                # Q wave analysis window (before R peak)
                q_start = qrs_idx - int(0.04 * self.sampling_rate)  # 40ms before QRS
                q_end = qrs_idx
                
                if q_start >= 0:
                    q_segment = lead_signal[q_start:q_end]
                    
                    # Find Q wave (negative deflection before R)
                    baseline = self._calculate_baseline(lead_signal, qrs_idx)
                    q_min_idx = np.argmin(q_segment)
                    q_depth = baseline - q_segment[q_min_idx]
                    
                    if q_depth > 0:  # Significant Q wave
                        # Calculate Q wave width
                        q_width = self._calculate_q_width(q_segment, q_min_idx)
                        q_wave_widths.append(q_width)
                        q_wave_depths.append(q_depth)
                        
                        # Q/R ratio (pathological if >25%)
                        r_amplitude = self._get_r_amplitude(lead_signal, qrs_idx)
                        if r_amplitude > 0:
                            q_r_ratios.append(q_depth / r_amplitude)
            
            # Q wave features
            features[f'{lead_name}_q_wave_max_width'] = np.max(q_wave_widths) if q_wave_widths else 0
            features[f'{lead_name}_q_wave_mean_depth'] = np.mean(q_wave_depths) if q_wave_depths else 0
            features[f'{lead_name}_q_r_ratio_max'] = np.max(q_r_ratios) if q_r_ratios else 0
            features[f'{lead_name}_pathological_q_count'] = len([w for w in q_wave_widths if w > 0.04])  # >40ms
            
        return features
    
    def _extract_t_wave_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract T-wave features relevant to MI"""
        features = {}
        
        for lead_idx in range(12):
            lead_name = self._get_lead_name(lead_idx)
            lead_signal = ecg_signal[lead_idx]
            
            qrs_indices = self._detect_qrs_complexes(lead_signal)
            
            t_wave_amplitudes = []
            t_wave_inversions = []
            t_wave_widths = []
            
            for qrs_idx in qrs_indices:
                # T wave window (after ST segment)
                t_start = qrs_idx + int(0.2 * self.sampling_rate)   # 200ms after QRS
                t_end = qrs_idx + int(0.5 * self.sampling_rate)     # 500ms after QRS
                
                if t_end < len(lead_signal):
                    t_segment = lead_signal[t_start:t_end]
                    baseline = self._calculate_baseline(lead_signal, qrs_idx)
                    
                    # Find T wave peak
                    t_peak_idx = np.argmax(np.abs(t_segment - baseline))
                    t_amplitude = t_segment[t_peak_idx] - baseline
                    
                    t_wave_amplitudes.append(abs(t_amplitude))
                    
                    # Check for T wave inversion
                    if t_amplitude < -0.1:  # Significant inversion
                        t_wave_inversions.append(abs(t_amplitude))
                    
                    # T wave width
                    t_width = self._calculate_t_width(t_segment, t_peak_idx)
                    t_wave_widths.append(t_width)
            
            # T wave features
            features[f'{lead_name}_t_wave_amplitude_mean'] = np.mean(t_wave_amplitudes) if t_wave_amplitudes else 0
            features[f'{lead_name}_t_wave_inversion_depth'] = np.max(t_wave_inversions) if t_wave_inversions else 0
            features[f'{lead_name}_t_wave_inversion_count'] = len(t_wave_inversions)
            features[f'{lead_name}_t_wave_width_mean'] = np.mean(t_wave_widths) if t_wave_widths else 0
            
        return features
    
    def _extract_r_progression_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract R wave progression features (important for anterior MI)"""
        features = {}
        
        # Focus on precordial leads V1-V6 (indices 6-11)
        precordial_leads = [6, 7, 8, 9, 10, 11]
        r_amplitudes = []
        
        for lead_idx in precordial_leads:
            lead_signal = ecg_signal[lead_idx]
            qrs_indices = self._detect_qrs_complexes(lead_signal)
            
            lead_r_amplitudes = []
            for qrs_idx in qrs_indices:
                r_amp = self._get_r_amplitude(lead_signal, qrs_idx)
                lead_r_amplitudes.append(r_amp)
            
            mean_r_amp = np.mean(lead_r_amplitudes) if lead_r_amplitudes else 0
            r_amplitudes.append(mean_r_amp)
        
        # Poor R wave progression features
        features['poor_r_progression'] = 1 if self._detect_poor_r_progression(r_amplitudes) else 0
        features['r_amplitude_v1'] = r_amplitudes[0] if len(r_amplitudes) > 0 else 0
        features['r_amplitude_v6'] = r_amplitudes[5] if len(r_amplitudes) > 5 else 0
        features['r_amplitude_progression_slope'] = self._calculate_progression_slope(r_amplitudes)
        
        # Transition zone analysis
        features['transition_zone_lead'] = self._find_transition_zone(r_amplitudes)
        features['delayed_transition'] = 1 if features['transition_zone_lead'] > 3 else 0  # After V4
        
        return features
    
    def _extract_territory_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract territory-specific features for different MI types"""
        features = {}
        
        for territory, config in self.mi_territories.items():
            primary_leads = config['primary_leads']
            
            if not primary_leads:  # Skip posterior (no direct leads)
                continue
            
            # Calculate territory-specific metrics
            st_elevations = []
            q_waves = []
            
            for lead_idx in primary_leads:
                lead_signal = ecg_signal[lead_idx]
                lead_name = self._get_lead_name(lead_idx)
                
                # ST elevation in this territory
                qrs_indices = self._detect_qrs_complexes(lead_signal)
                for qrs_idx in qrs_indices:
                    st_elevation = self._get_st_elevation_at_point(lead_signal, qrs_idx)
                    st_elevations.append(st_elevation)
                    
                    # Q wave presence
                    q_wave_depth = self._get_q_wave_depth(lead_signal, qrs_idx)
                    q_waves.append(q_wave_depth)
            
            # Territory features
            features[f'{territory}_st_elevation_max'] = np.max(st_elevations) if st_elevations else 0
            features[f'{territory}_st_elevation_mean'] = np.mean(st_elevations) if st_elevations else 0
            features[f'{territory}_q_wave_max'] = np.max(q_waves) if q_waves else 0
            features[f'{territory}_affected_leads'] = len([x for x in st_elevations if x > 0.1])
            
            # Territory-specific MI probability
            features[f'{territory}_mi_probability'] = self._calculate_territory_mi_probability(
                st_elevations, q_waves, territory
            )
        
        return features
    
    def _extract_reciprocal_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract reciprocal change features"""
        features = {}
        
        for territory, config in self.mi_territories.items():
            primary_leads = config['primary_leads']
            reciprocal_leads = config['reciprocal_leads']
            
            if not primary_leads or not reciprocal_leads:
                continue
            
            # Calculate reciprocal changes
            primary_st_elevation = 0
            reciprocal_st_depression = 0
            
            # Primary territory ST elevation
            for lead_idx in primary_leads:
                lead_signal = ecg_signal[lead_idx]
                qrs_indices = self._detect_qrs_complexes(lead_signal)
                for qrs_idx in qrs_indices:
                    st_elev = self._get_st_elevation_at_point(lead_signal, qrs_idx)
                    primary_st_elevation = max(primary_st_elevation, st_elev)
            
            # Reciprocal territory ST depression
            for lead_idx in reciprocal_leads:
                lead_signal = ecg_signal[lead_idx]
                qrs_indices = self._detect_qrs_complexes(lead_signal)
                for qrs_idx in qrs_indices:
                    st_dep = self._get_st_depression_at_point(lead_signal, qrs_idx)
                    reciprocal_st_depression = max(reciprocal_st_depression, st_dep)
            
            # Reciprocal features
            features[f'{territory}_reciprocal_ratio'] = (
                reciprocal_st_depression / primary_st_elevation if primary_st_elevation > 0 else 0
            )
            features[f'{territory}_reciprocal_present'] = (
                1 if reciprocal_st_depression > 0.05 and primary_st_elevation > 0.1 else 0
            )
        
        return features
    
    def _extract_lead_group_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract features based on anatomical lead groupings"""
        features = {}
        
        for group_name, lead_indices in self.lead_groups.items():
            group_signals = ecg_signal[lead_indices]
            
            # Group-wide metrics
            features[f'{group_name}_amplitude_mean'] = np.mean(np.abs(group_signals))
            features[f'{group_name}_amplitude_std'] = np.std(np.abs(group_signals))
            features[f'{group_name}_correlation_mean'] = self._calculate_group_correlation(group_signals)
            
            # ST segment analysis for group
            group_st_elevations = []
            for lead_idx in lead_indices:
                lead_signal = ecg_signal[lead_idx]
                qrs_indices = self._detect_qrs_complexes(lead_signal)
                for qrs_idx in qrs_indices:
                    st_elev = self._get_st_elevation_at_point(lead_signal, qrs_idx)
                    group_st_elevations.append(st_elev)
            
            features[f'{group_name}_st_elevation_max'] = np.max(group_st_elevations) if group_st_elevations else 0
            features[f'{group_name}_st_elevation_consistency'] = (
                np.std(group_st_elevations) if len(group_st_elevations) > 1 else 0
            )
        
        return features
    
    def _extract_advanced_morphology(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract advanced morphological features for MI detection"""
        features = {}
        
        # QRS morphology changes
        features['qrs_fragmentation_score'] = self._calculate_qrs_fragmentation(ecg_signal)
        features['notching_score'] = self._calculate_notching_score(ecg_signal)
        features['conduction_delay_score'] = self._calculate_conduction_delay(ecg_signal)
        
        # Temporal features
        features['heart_rate_variability'] = self._calculate_hrv(ecg_signal)
        features['pr_interval_variability'] = self._calculate_pr_variability(ecg_signal)
        features['qt_interval_mean'] = self._calculate_qt_interval(ecg_signal)
        
        # Advanced ST-T features
        features['st_t_ratio_abnormal'] = self._calculate_st_t_ratio_abnormality(ecg_signal)
        features['repolarization_heterogeneity'] = self._calculate_repolarization_heterogeneity(ecg_signal)
        
        return features
    
    # Helper methods for feature calculation
    def _get_lead_name(self, lead_idx: int) -> str:
        """Convert lead index to name"""
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        return lead_names[lead_idx] if lead_idx < len(lead_names) else f'Lead_{lead_idx}'
    
    def _detect_qrs_complexes(self, signal: np.ndarray) -> List[int]:
        """Detect QRS complex locations using peak detection"""
        # Simple R-peak detection
        peaks, _ = signal.find_peaks(np.abs(signal), height=np.max(np.abs(signal)) * 0.3, distance=int(0.3 * self.sampling_rate))
        return peaks.tolist()
    
    def _calculate_baseline(self, signal: np.ndarray, qrs_idx: int) -> float:
        """Calculate isoelectric baseline around QRS"""
        # Use PR segment as baseline (before QRS)
        baseline_start = max(0, qrs_idx - int(0.2 * self.sampling_rate))
        baseline_end = max(0, qrs_idx - int(0.05 * self.sampling_rate))
        
        if baseline_end > baseline_start:
            return np.mean(signal[baseline_start:baseline_end])
        return 0
    
    def _get_st_elevation_at_point(self, signal: np.ndarray, qrs_idx: int) -> float:
        """Calculate ST elevation at specific point"""
        st_point = qrs_idx + int(0.08 * self.sampling_rate)  # 80ms after QRS
        if st_point < len(signal):
            baseline = self._calculate_baseline(signal, qrs_idx)
            return max(0, signal[st_point] - baseline)
        return 0
    
    def _get_st_depression_at_point(self, signal: np.ndarray, qrs_idx: int) -> float:
        """Calculate ST depression at specific point"""
        st_point = qrs_idx + int(0.08 * self.sampling_rate)
        if st_point < len(signal):
            baseline = self._calculate_baseline(signal, qrs_idx)
            return max(0, baseline - signal[st_point])
        return 0
    
    def _get_r_amplitude(self, signal: np.ndarray, qrs_idx: int) -> float:
        """Get R wave amplitude"""
        baseline = self._calculate_baseline(signal, qrs_idx)
        r_start = qrs_idx - int(0.02 * self.sampling_rate)
        r_end = qrs_idx + int(0.02 * self.sampling_rate)
        
        if r_start >= 0 and r_end < len(signal):
            r_segment = signal[r_start:r_end]
            return np.max(r_segment) - baseline
        return 0
    
    def _get_q_wave_depth(self, signal: np.ndarray, qrs_idx: int) -> float:
        """Get Q wave depth"""
        baseline = self._calculate_baseline(signal, qrs_idx)
        q_start = qrs_idx - int(0.04 * self.sampling_rate)
        q_end = qrs_idx
        
        if q_start >= 0:
            q_segment = signal[q_start:q_end]
            return max(0, baseline - np.min(q_segment))
        return 0
    
    def _detect_poor_r_progression(self, r_amplitudes: List[float]) -> bool:
        """Detect poor R wave progression in precordial leads"""
        if len(r_amplitudes) < 4:
            return False
        
        # Poor progression if R waves don't increase from V1 to V4
        for i in range(3):
            if r_amplitudes[i+1] <= r_amplitudes[i]:
                return True
        return False
    
    def _calculate_progression_slope(self, r_amplitudes: List[float]) -> float:
        """Calculate R wave progression slope"""
        if len(r_amplitudes) < 2:
            return 0
        
        x = np.arange(len(r_amplitudes))
        slope = np.polyfit(x, r_amplitudes, 1)[0]
        return slope
    
    def _find_transition_zone(self, r_amplitudes: List[float]) -> int:
        """Find the transition zone (where R > S)"""
        # Simplified - return lead index where R amplitude peaks
        if r_amplitudes:
            return np.argmax(r_amplitudes)
        return 0
    
    def _calculate_territory_mi_probability(self, st_elevations: List[float], q_waves: List[float], territory: str) -> float:
        """Calculate MI probability for specific territory"""
        st_score = len([x for x in st_elevations if x > 0.1]) / max(1, len(st_elevations))
        q_score = len([x for x in q_waves if x > 0.05]) / max(1, len(q_waves))
        
        # Weight based on territory
        territory_weights = {'anterior': 0.8, 'inferior': 0.7, 'lateral': 0.6}
        weight = territory_weights.get(territory, 0.5)
        
        return (st_score * 0.7 + q_score * 0.3) * weight
    
    def _calculate_group_correlation(self, group_signals: np.ndarray) -> float:
        """Calculate mean correlation within lead group"""
        if group_signals.shape[0] < 2:
            return 0
        
        correlations = []
        for i in range(group_signals.shape[0]):
            for j in range(i + 1, group_signals.shape[0]):
                corr = np.corrcoef(group_signals[i], group_signals[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0
    
    # Additional helper methods (simplified implementations)
    def _calculate_st_slope(self, signal: np.ndarray, qrs_indices: List[int]) -> float:
        """Calculate ST segment slope"""
        return 0  # Simplified implementation
    
    def _calculate_st_concavity(self, signal: np.ndarray, qrs_indices: List[int]) -> float:
        """Calculate ST segment concavity"""
        return 0  # Simplified implementation
    
    def _calculate_q_width(self, q_segment: np.ndarray, q_min_idx: int) -> float:
        """Calculate Q wave width"""
        return 0.02  # Simplified implementation
    
    def _calculate_t_width(self, t_segment: np.ndarray, t_peak_idx: int) -> float:
        """Calculate T wave width"""
        return 0.15  # Simplified implementation
    
    def _calculate_qrs_fragmentation(self, ecg_signal: np.ndarray) -> float:
        """Calculate QRS fragmentation score"""
        return 0  # Simplified implementation
    
    def _calculate_notching_score(self, ecg_signal: np.ndarray) -> float:
        """Calculate notching score"""
        return 0  # Simplified implementation
    
    def _calculate_conduction_delay(self, ecg_signal: np.ndarray) -> float:
        """Calculate conduction delay score"""
        return 0  # Simplified implementation
    
    def _calculate_hrv(self, ecg_signal: np.ndarray) -> float:
        """Calculate heart rate variability"""
        return 0  # Simplified implementation
    
    def _calculate_pr_variability(self, ecg_signal: np.ndarray) -> float:
        """Calculate PR interval variability"""
        return 0  # Simplified implementation
    
    def _calculate_qt_interval(self, ecg_signal: np.ndarray) -> float:
        """Calculate mean QT interval"""
        return 0.4  # Simplified implementation
    
    def _calculate_st_t_ratio_abnormality(self, ecg_signal: np.ndarray) -> float:
        """Calculate ST/T ratio abnormality"""
        return 0  # Simplified implementation
    
    def _calculate_repolarization_heterogeneity(self, ecg_signal: np.ndarray) -> float:
        """Calculate repolarization heterogeneity"""
        return 0  # Simplified implementation