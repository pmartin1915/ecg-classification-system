"""
Preprocessing configuration for ECG signals
"""
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class PreprocessingConfig:
    """Configuration parameters for ECG preprocessing"""
    
    # Sampling parameters
    sampling_rate: int = 100
    target_length: int = 1000  # Standardized signal length
    
    # Filter parameters
    highpass_freq: float = 0.5      # Remove baseline wander
    lowpass_freq: float = 40        # Remove high-frequency noise
    notch_freq: float = 49          # Powerline interference (50Hz EU, 60Hz US)
    notch_quality: float = 30       # Q-factor for notch filter
    filter_order: int = 4           # Filter order
    
    # Artifact detection parameters
    amplitude_threshold: float = 5.0     # mV - for amplitude-based detection
    gradient_threshold: float = 2.0      # mV/sample - for gradient-based detection
    saturation_threshold: float = 0.95   # Percentage of max value
    
    # Normalization parameters
    normalization_method: Literal['z-score', 'min-max', 'robust'] = 'z-score'
    clip_percentile: float = 99.5        # Percentile for outlier clipping
    
    # Quality control parameters
    min_signal_length: int = 500         # Minimum required signal length
    max_missing_leads: int = 2           # Maximum bad leads allowed
    min_lead_std: float = 1e-6          # Minimum std dev for valid lead
    max_lead_std: float = 10.0          # Maximum std dev for valid lead
    min_signal_range: float = 0.1       # Minimum range for valid lead
    
    # Processing parameters
    output_dtype: str = 'float32'        # Data type for memory efficiency
    use_cache: bool = True               # Whether to use caching
    batch_size: int = 1000               # Batch size for processing
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate filter frequencies
        nyquist = self.sampling_rate / 2
        if self.highpass_freq >= nyquist:
            raise ValueError(f"Highpass frequency ({self.highpass_freq}) must be less than Nyquist frequency ({nyquist})")
        if self.lowpass_freq >= nyquist:
            raise ValueError(f"Lowpass frequency ({self.lowpass_freq}) must be less than Nyquist frequency ({nyquist})")
        if self.notch_freq >= nyquist:
            raise ValueError(f"Notch frequency ({self.notch_freq}) must be less than Nyquist frequency ({nyquist})")
        
        # Validate percentiles
        if not 0 < self.clip_percentile < 100:
            raise ValueError("Clip percentile must be between 0 and 100")
        
        # Validate thresholds
        if self.amplitude_threshold <= 0:
            raise ValueError("Amplitude threshold must be positive")
        if self.gradient_threshold <= 0:
            raise ValueError("Gradient threshold must be positive")
        if not 0 < self.saturation_threshold <= 1:
            raise ValueError("Saturation threshold must be between 0 and 1")


# Preset configurations for different use cases
PREPROCESSING_PRESETS = {
    'standard': PreprocessingConfig(),
    
    'high_quality': PreprocessingConfig(
        highpass_freq=0.05,
        lowpass_freq=40,  # Changed from 100 to 40
        clip_percentile=99.9,
        max_missing_leads=1
    ),
    
    'fast': PreprocessingConfig(
        target_length=500,
        filter_order=2,
        batch_size=2000
    ),
    
    'robust': PreprocessingConfig(
        normalization_method='robust',
        clip_percentile=98,
        max_missing_leads=3,
        min_lead_std=1e-5,
        max_lead_std=15.0
    )
}