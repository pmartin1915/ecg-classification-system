"""
Comprehensive Condition Mapping for Advanced Cardiac Detection
Maps PTB-XL and ECG Arrhythmia conditions to comprehensive 30-condition system
"""

from typing import Dict, List, Set
import pandas as pd

class ComprehensiveConditionMapper:
    """Maps various ECG conditions to comprehensive 30-condition classification"""
    
    def __init__(self):
        self.ptbxl_mapping = self._create_ptbxl_mapping()
        self.arrhythmia_snomed_mapping = self._create_arrhythmia_mapping()
        
    def _create_ptbxl_mapping(self) -> Dict[str, str]:
        """Create mapping from PTB-XL conditions to comprehensive conditions"""
        return {
            # === NORMAL ===
            'NORM': 'NORM',
            
            # === MYOCARDIAL INFARCTION ===
            'AMI': 'AMI',           # Anterior MI
            'ASMI': 'AMI',          # Anteroseptal MI -> Anterior MI
            'ALMI': 'AMI',          # Anterolateral MI -> Anterior MI
            'IMI': 'IMI',           # Inferior MI
            'ILMI': 'IMI',          # Inferolateral MI -> Inferior MI
            'IPLMI': 'IMI',         # Inferoposterolateral MI -> Inferior MI
            'IPMI': 'IMI',          # Inferoposterior MI -> Inferior MI
            'LMI': 'LMI',           # Lateral MI
            'PMI': 'PMI',           # Posterior MI
            
            # === ARRHYTHMIAS ===
            'AFIB': 'AFIB',         # Atrial Fibrillation
            'AFLT': 'AFLT',         # Atrial Flutter
            'SVTAC': 'SVTAC',       # Supraventricular Tachycardia
            'VTAC': 'VTAC',         # Ventricular Tachycardia
            'PVC': 'PVC',           # Premature Ventricular Contractions
            'PAC': 'PAC',           # Premature Atrial Contractions
            
            # === CONDUCTION DISORDERS ===
            '1AVB': 'AVB1',         # 1st Degree AV Block
            '2AVB': 'AVB2',         # 2nd Degree AV Block
            '3AVB': 'AVB3',         # 3rd Degree AV Block
            'CRBBB': 'RBBB',        # Complete Right Bundle Branch Block
            'IRBBB': 'RBBB',        # Incomplete Right Bundle Branch Block
            'CLBBB': 'LBBB',        # Complete Left Bundle Branch Block
            'ILBBB': 'LBBB',        # Incomplete Left Bundle Branch Block
            'LAFB': 'LAFB',         # Left Anterior Fascicular Block
            'LPFB': 'LPFB',         # Left Posterior Fascicular Block
            'IVCD': 'IVCD',         # Intraventricular Conduction Disturbance
            'WPW': 'WPW',           # Wolff-Parkinson-White Syndrome
            
            # === HYPERTROPHY ===
            'LVH': 'LVH',           # Left Ventricular Hypertrophy
            'VCLVH': 'LVH',         # Voltage Criteria for LVH -> LVH
            'RVH': 'RVH',           # Right Ventricular Hypertrophy
            'LAO/LAE': 'LAE',       # Left Atrial Enlargement
            'RAO/RAE': 'RAE',       # Right Atrial Enlargement
            
            # === ST-T CHANGES & ISCHEMIA ===
            'ISC_': 'ISCH',         # Non-specific Ischemic
            'ISCAL': 'ISCH',        # Ischemic in Anterolateral leads
            'ISCIN': 'ISCH',        # Ischemic in Inferior leads
            'ISCIL': 'ISCH',        # Ischemic in Inferolateral leads
            'ISCAS': 'ISCH',        # Ischemic in Anteroseptal leads
            'ISCLA': 'ISCH',        # Ischemic in Lateral leads
            'ISCAN': 'ISCH',        # Ischemic in Anterior leads
            'NST_': 'STTC',         # Non-specific ST changes
            'STD_': 'STTC',         # Non-specific ST depression
            'LNGQT': 'LNGQT',       # Long QT interval
            
            # === OTHER CONDITIONS ===
            'PACE': 'PACE',         # Paced rhythm
            'DIG': 'DIG',           # Digitalis effect
            'LOWT': 'LOWT',         # Low T-wave voltage
            
            # === LEGACY MAPPINGS (map to broader categories) ===
            'MI': 'AMI',            # Generic MI -> Anterior MI
            'STTC': 'STTC',         # ST-T Changes
            'CD': 'IVCD',           # Conduction Disorders -> IVCD
            'HYP': 'LVH',           # Hypertrophy -> LVH
        }
    
    def _create_arrhythmia_mapping(self) -> Dict[str, str]:
        """Create mapping from SNOMED-CT codes to comprehensive conditions"""
        return {
            # Common SNOMED-CT codes from ECG Arrhythmia dataset
            '164889003': 'AFIB',    # Atrial fibrillation
            '59118001': 'RBBB',     # Right bundle branch block
            '164934002': 'VTAC',    # Ventricular tachycardia
            '427393009': 'PVC',     # Premature ventricular contraction
            '17338001': 'AVB1',     # First degree AV block
            '251268003': 'LBBB',    # Left bundle branch block
            '39732003': 'LVH',      # Left ventricular hypertrophy
            '164909002': 'PACE',    # Paced rhythm
            '427084000': 'AFLT',    # Atrial flutter
            '426434006': 'AVB2',    # Second degree AV block
            '27885002': 'AVB3',     # Third degree AV block
            '195080001': 'ISCH',    # Ischemic heart disease
            '164931005': 'STTC',    # ST-T changes
            '164890007': 'SVTAC',   # Supraventricular tachycardia
            '67198005': 'PAC',      # Premature atrial contraction
            '164947007': 'LAE',     # Left atrial enlargement
            '164948002': 'RAE',     # Right atrial enlargement
            '164951009': 'RVH',     # Right ventricular hypertrophy
            '426177001': 'WPW',     # Wolff-Parkinson-White syndrome
            '428750005': 'LNGQT',   # Long QT syndrome
            # Add more mappings as discovered
        }
    
    def map_ptbxl_conditions(self, conditions: List[str]) -> List[str]:
        """Map PTB-XL conditions to comprehensive conditions"""
        mapped = []
        for condition in conditions:
            if condition in self.ptbxl_mapping:
                mapped_condition = self.ptbxl_mapping[condition]
                if mapped_condition not in mapped:  # Avoid duplicates
                    mapped.append(mapped_condition)
            else:
                # Unknown condition - could log for future mapping
                print(f"Warning: Unknown PTB-XL condition '{condition}' - mapping to STTC")
                if 'STTC' not in mapped:
                    mapped.append('STTC')
        
        return mapped if mapped else ['NORM']  # Default to NORM if no mapping found
    
    def map_arrhythmia_snomed_codes(self, snomed_codes: List[str]) -> List[str]:
        """Map SNOMED-CT codes from ECG Arrhythmia dataset to comprehensive conditions"""
        mapped = []
        for code in snomed_codes:
            code = str(code).strip()  # Ensure string and remove whitespace
            if code in self.arrhythmia_snomed_mapping:
                mapped_condition = self.arrhythmia_snomed_mapping[code]
                if mapped_condition not in mapped:  # Avoid duplicates
                    mapped.append(mapped_condition)
            else:
                # Unknown code - could log for future mapping
                print(f"Warning: Unknown SNOMED code '{code}' - mapping to STTC")
                if 'STTC' not in mapped:
                    mapped.append('STTC')
        
        return mapped if mapped else ['NORM']  # Default to NORM if no mapping found
    
    def get_primary_condition(self, conditions: List[str]) -> str:
        """Get the primary (most critical) condition from a list"""
        # Priority order (most critical first)
        priority_order = [
            'AMI', 'IMI', 'LMI', 'PMI',     # Myocardial Infarction (highest priority)
            'VTAC', 'AVB3',                  # Life-threatening arrhythmias
            'AFIB', 'AFLT', 'AVB2',         # Serious arrhythmias
            'PVC', 'PAC', 'SVTAC',          # Other arrhythmias
            'LBBB', 'WPW',                  # Serious conduction disorders
            'RBBB', 'AVB1', 'LAFB', 'LPFB', 'IVCD',  # Other conduction disorders
            'LVH', 'RVH', 'LAE', 'RAE',     # Structural abnormalities
            'ISCH', 'LNGQT',                # Ischemia and repolarization
            'STTC', 'PACE', 'DIG', 'LOWT',  # Other findings
            'NORM'                          # Normal (lowest priority)
        ]
        
        for priority_condition in priority_order:
            if priority_condition in conditions:
                return priority_condition
        
        return conditions[0] if conditions else 'NORM'
    
    def get_condition_info(self, condition: str) -> Dict[str, str]:
        """Get clinical information about a condition"""
        condition_info = {
            # === NORMAL ===
            'NORM': {'name': 'Normal Sinus Rhythm', 'category': 'Normal', 'priority': 'LOW'},
            
            # === MYOCARDIAL INFARCTION ===
            'AMI': {'name': 'Anterior Myocardial Infarction', 'category': 'MI', 'priority': 'CRITICAL'},
            'IMI': {'name': 'Inferior Myocardial Infarction', 'category': 'MI', 'priority': 'CRITICAL'},
            'LMI': {'name': 'Lateral Myocardial Infarction', 'category': 'MI', 'priority': 'CRITICAL'},
            'PMI': {'name': 'Posterior Myocardial Infarction', 'category': 'MI', 'priority': 'CRITICAL'},
            
            # === ARRHYTHMIAS ===
            'AFIB': {'name': 'Atrial Fibrillation', 'category': 'Arrhythmia', 'priority': 'HIGH'},
            'AFLT': {'name': 'Atrial Flutter', 'category': 'Arrhythmia', 'priority': 'HIGH'},
            'VTAC': {'name': 'Ventricular Tachycardia', 'category': 'Arrhythmia', 'priority': 'CRITICAL'},
            'SVTAC': {'name': 'Supraventricular Tachycardia', 'category': 'Arrhythmia', 'priority': 'MEDIUM'},
            'PVC': {'name': 'Premature Ventricular Contractions', 'category': 'Arrhythmia', 'priority': 'HIGH'},
            'PAC': {'name': 'Premature Atrial Contractions', 'category': 'Arrhythmia', 'priority': 'MEDIUM'},
            
            # === CONDUCTION DISORDERS ===
            'AVB1': {'name': 'First-Degree AV Block', 'category': 'Conduction', 'priority': 'MEDIUM'},
            'AVB2': {'name': 'Second-Degree AV Block', 'category': 'Conduction', 'priority': 'HIGH'},
            'AVB3': {'name': 'Third-Degree AV Block', 'category': 'Conduction', 'priority': 'CRITICAL'},
            'RBBB': {'name': 'Right Bundle Branch Block', 'category': 'Conduction', 'priority': 'MEDIUM'},
            'LBBB': {'name': 'Left Bundle Branch Block', 'category': 'Conduction', 'priority': 'HIGH'},
            'LAFB': {'name': 'Left Anterior Fascicular Block', 'category': 'Conduction', 'priority': 'MEDIUM'},
            'LPFB': {'name': 'Left Posterior Fascicular Block', 'category': 'Conduction', 'priority': 'MEDIUM'},
            'IVCD': {'name': 'Intraventricular Conduction Delay', 'category': 'Conduction', 'priority': 'MEDIUM'},
            'WPW': {'name': 'Wolff-Parkinson-White Syndrome', 'category': 'Conduction', 'priority': 'HIGH'},
            
            # === HYPERTROPHY ===
            'LVH': {'name': 'Left Ventricular Hypertrophy', 'category': 'Structural', 'priority': 'MEDIUM'},
            'RVH': {'name': 'Right Ventricular Hypertrophy', 'category': 'Structural', 'priority': 'MEDIUM'},
            'LAE': {'name': 'Left Atrial Enlargement', 'category': 'Structural', 'priority': 'LOW'},
            'RAE': {'name': 'Right Atrial Enlargement', 'category': 'Structural', 'priority': 'LOW'},
            
            # === ST-T CHANGES ===
            'ISCH': {'name': 'Ischemic Changes', 'category': 'Ischemia', 'priority': 'MEDIUM'},
            'STTC': {'name': 'Non-specific ST-T Changes', 'category': 'ST-T', 'priority': 'LOW'},
            'LNGQT': {'name': 'Long QT Interval', 'category': 'Repolarization', 'priority': 'MEDIUM'},
            
            # === OTHER ===
            'PACE': {'name': 'Paced Rhythm', 'category': 'Other', 'priority': 'LOW'},
            'DIG': {'name': 'Digitalis Effect', 'category': 'Drug Effect', 'priority': 'LOW'},
            'LOWT': {'name': 'Low T-wave Voltage', 'category': 'Other', 'priority': 'LOW'},
        }
        
        return condition_info.get(condition, {'name': condition, 'category': 'Unknown', 'priority': 'MEDIUM'})

# Global instance for easy import
comprehensive_mapper = ComprehensiveConditionMapper()