"""
Comprehensive Cardiac Analysis Configuration
Expands beyond basic 5 conditions to full clinical spectrum
"""

# COMPREHENSIVE TARGET CONDITIONS for Clinical Cardiac Analysis
COMPREHENSIVE_CONDITIONS = {
    # === NORMAL ===
    'NORM': ['NORM'],  # Normal ECG
    
    # === MYOCARDIAL INFARCTION (Multiple Types) ===
    'AMI': ['AMI', 'ASMI', 'ALMI'],          # Anterior MI variants
    'IMI': ['IMI', 'ILMI', 'IPLMI', 'IPMI'], # Inferior MI variants  
    'LMI': ['LMI'],                          # Lateral MI
    'PMI': ['PMI'],                          # Posterior MI
    
    # === ARRHYTHMIAS & RHYTHM DISORDERS ===
    'PVC': ['PVC'],                          # Premature Ventricular Contractions
    'PAC': ['PAC'],                          # Premature Atrial Contractions
    'AFIB': ['AFIB'],                        # Atrial Fibrillation
    'AFLT': ['AFLT'],                        # Atrial Flutter
    'SVTAC': ['SVTAC'],                      # Supraventricular Tachycardia
    'VTAC': ['VTAC'],                        # Ventricular Tachycardia
    
    # === CONDUCTION DISORDERS ===
    'AVB1': ['1AVB'],                        # 1st Degree AV Block
    'AVB2': ['2AVB'],                        # 2nd Degree AV Block  
    'AVB3': ['3AVB'],                        # 3rd Degree AV Block
    'RBBB': ['CRBBB', 'IRBBB'],            # Right Bundle Branch Block
    'LBBB': ['CLBBB', 'ILBBB'],            # Left Bundle Branch Block
    'LAFB': ['LAFB'],                        # Left Anterior Fascicular Block
    'LPFB': ['LPFB'],                        # Left Posterior Fascicular Block
    'IVCD': ['IVCD'],                        # Intraventricular Conduction Delay
    'WPW': ['WPW'],                          # Wolff-Parkinson-White Syndrome
    
    # === HYPERTROPHY ===
    'LVH': ['LVH', 'VCLVH'],                # Left Ventricular Hypertrophy
    'RVH': ['RVH'],                          # Right Ventricular Hypertrophy
    'LAE': ['LAO/LAE'],                      # Left Atrial Enlargement
    'RAE': ['RAO/RAE'],                      # Right Atrial Enlargement
    
    # === ST-T CHANGES & ISCHEMIA ===
    'ISCH': ['ISC_', 'ISCAL', 'ISCIN', 'ISCIL', 'ISCAS', 'ISCLA', 'ISCAN'], # Ischemic changes
    'STTC': ['NST_', 'STD_'],               # Non-specific ST-T changes
    'LNGQT': ['LNGQT'],                      # Long QT interval
    
    # === OTHER CLINICALLY SIGNIFICANT ===
    'PACE': ['PACE'],                        # Paced rhythm
    'DIG': ['DIG'],                          # Digitalis effect
    'LOWT': ['LOWT'],                        # Low T-wave voltage
}

# ARRHYTHMIA DIAGNOSTIC CODE MAPPING (for ECG Arrhythmia dataset)
ARRHYTHMIA_CODE_MAPPING = {
    # SNOMED-CT codes to clinical conditions
    '164889003': 'AFIB',     # Atrial fibrillation
    '59118001': 'RBBB',      # Right bundle branch block
    '164934002': 'VT',       # Ventricular tachycardia
    '427393009': 'PVC',      # Premature ventricular contraction
    '17338001': 'AVB1',      # First degree AV block
    '251268003': 'LBBB',     # Left bundle branch block
    '39732003': 'LVH',       # Left ventricular hypertrophy
    '164909002': 'PACE',     # Paced rhythm
    '427084000': 'AFLT',     # Atrial flutter
    '426434006': 'AVB2',     # Second degree AV block
    '27885002': 'AVB3',      # Third degree AV block
    # Add more mappings as needed
}

# CLINICAL PRIORITY LEVELS
CLINICAL_PRIORITY = {
    'CRITICAL': ['AMI', 'IMI', 'VTAC', 'AVB3', 'LMI', 'PMI'],  # Immediate attention
    'HIGH': ['PVC', 'AFIB', 'AVB2', 'LBBB', 'WPW'],           # Close monitoring  
    'MEDIUM': ['AVB1', 'RBBB', 'LVH', 'ISCH'],                # Regular follow-up
    'LOW': ['NORM', 'STTC', 'DIG'],                           # Routine care
}

# ENHANCED FEATURES FOR ARRHYTHMIA DETECTION
ARRHYTHMIA_FEATURES = {
    'rhythm_analysis': True,      # R-R interval variability
    'p_wave_detection': True,     # P-wave morphology analysis
    'qrs_morphology': True,       # QRS complex analysis
    'heart_rate_variability': True, # HRV parameters
    'frequency_domain': True,     # FFT analysis for arrhythmias
    'wavelet_analysis': True,     # Time-frequency analysis
}

def get_comprehensive_config():
    """Return comprehensive cardiac analysis configuration"""
    return {
        'target_conditions': list(COMPREHENSIVE_CONDITIONS.keys()),
        'condition_mapping': COMPREHENSIVE_CONDITIONS,
        'arrhythmia_codes': ARRHYTHMIA_CODE_MAPPING,
        'priority_levels': CLINICAL_PRIORITY,
        'enhanced_features': ARRHYTHMIA_FEATURES,
        'total_conditions': len(COMPREHENSIVE_CONDITIONS)
    }

def get_condition_description(condition_code):
    """Get clinical description of condition"""
    descriptions = {
        'NORM': 'Normal sinus rhythm',
        'AMI': 'Anterior myocardial infarction',
        'IMI': 'Inferior myocardial infarction', 
        'PVC': 'Premature ventricular contractions',
        'AFIB': 'Atrial fibrillation',
        'AVB1': 'First-degree AV block',
        'LBBB': 'Left bundle branch block',
        'LVH': 'Left ventricular hypertrophy',
        # Add more as needed
    }
    return descriptions.get(condition_code, condition_code)

if __name__ == "__main__":
    config = get_comprehensive_config()
    print("🫀 COMPREHENSIVE CARDIAC ANALYSIS CONFIGURATION")
    print("=" * 60)
    print(f"Total conditions: {config['total_conditions']}")
    print(f"Target conditions: {config['target_conditions']}")
    print(f"Enhanced features: {list(config['enhanced_features'].keys())}")