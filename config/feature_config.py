"""
Feature extraction configuration for ECG signals
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np


@dataclass
class FeatureExtractionConfig:
    """Configuration for ECG feature extraction"""
    
    # Signal parameters
    sampling_rate: int = 100
    signal_length: int = 1000  # From Phase 2
    
    # Lead configuration
    lead_names: List[str] = None
    limb_leads: List[int] = None
    chest_leads: List[int] = None
    
    # R-peak detection parameters
    r_peak_height: float = 0.3  # Minimum height for R-peak
    r_peak_distance: int = 60   # Min distance between peaks (0.6s at 100Hz)
    r_peak_prominence: float = 0.1
    
    # HRV parameters
    hrv_min_rr: float = 0.3  # Minimum RR interval (300ms)
    hrv_max_rr: float = 2.0  # Maximum RR interval (2000ms)
    
    # Frequency bands for spectral analysis
    freq_bands: Dict[str, Tuple[float, float]] = None
    
    # ST-segment parameters
    st_start_offset: float = 0.08   # 80ms after R-peak
    st_end_offset: float = 0.12     # 120ms after R-peak
    st_reference_offset: float = 0.04  # 40ms before R-peak
    
    # Morphological parameters
    qrs_duration_max: float = 0.12  # Maximum QRS duration (120ms)
    pr_interval_range: Tuple[float, float] = (0.12, 0.20)
    qt_interval_max: float = 0.44   # Maximum QT interval (440ms)
    
    # Wavelet parameters
    wavelet_scales: np.ndarray = None
    wavelet_type: str = 'morl'  # Using 'morl' for pywt compatibility
    dwt_wavelet: str = 'db4'
    dwt_level: int = 6
    
    # Feature selection parameters
    feature_selection_k: int = 50
    correlation_threshold: float = 0.95
    variance_threshold: float = 1e-10
    
    # Processing parameters
    use_cache: bool = True
    n_jobs: int = 4
    batch_size: int = 100
    
    def __post_init__(self):
        """Initialize default values and validate"""
        if self.lead_names is None:
            self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        if self.limb_leads is None:
            self.limb_leads = [0, 1, 2, 3, 4, 5]  # I, II, III, aVR, aVL, aVF
        
        if self.chest_leads is None:
            self.chest_leads = [6, 7, 8, 9, 10, 11]  # V1-V6
        
        if self.freq_bands is None:
            self.freq_bands = {
                'ultra_low': (0.0, 0.003),
                'very_low': (0.003, 0.04),
                'low': (0.04, 0.15),
                'high': (0.15, 0.4)
            }
        
        if self.wavelet_scales is None:
            self.wavelet_scales = np.arange(1, 32)
        
        # Validate parameters
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters"""
        # Validate R-peak parameters
        if self.r_peak_height <= 0:
            raise ValueError("R-peak height must be positive")
        
        if self.r_peak_distance <= 0:
            raise ValueError("R-peak distance must be positive")
        
        # Validate HRV parameters
        if self.hrv_min_rr >= self.hrv_max_rr:
            raise ValueError("HRV min RR must be less than max RR")
        
        # Validate ST-segment parameters
        if self.st_start_offset >= self.st_end_offset:
            raise ValueError("ST start offset must be less than end offset")
        
        # Validate feature selection parameters
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("Correlation threshold must be between 0 and 1")
        
        if self.feature_selection_k <= 0:
            raise ValueError("Feature selection k must be positive")


# Preset configurations
FEATURE_EXTRACTION_PRESETS = {
    'standard': FeatureExtractionConfig(),
    
    'comprehensive': FeatureExtractionConfig(
        feature_selection_k=100,
        wavelet_scales=np.arange(1, 64),
        dwt_level=8
    ),
    
    'fast': FeatureExtractionConfig(
        feature_selection_k=30,
        wavelet_scales=np.arange(1, 16),
        dwt_level=4,
        batch_size=200
    ),
    
    'clinical': FeatureExtractionConfig(
        r_peak_height=0.5,  # More conservative
        r_peak_prominence=0.2,
        feature_selection_k=75,
        correlation_threshold=0.90
    )
}