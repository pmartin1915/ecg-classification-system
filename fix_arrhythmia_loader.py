"""
Fixed ECG Arrhythmia Dataset Loader
Handles the specific format with embedded Dx codes in .hea files
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.io
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


class FixedECGArrhythmiaLoader:
    """Fixed loader for ECG Arrhythmia dataset that handles the actual data format"""
    
    def __init__(self):
        self.base_path = Path("data/raw/ecg-arrhythmia-dataset/WFDBRecords")
        
        # SNOMED-CT codes to condition mapping (based on actual dataset)
        self.dx_mapping = {
            # Normal sinus rhythm patterns
            '426783006': 'NORM',  # Normal sinus rhythm
            '164889003': 'NORM',  # Normal ECG
            '270492004': 'NORM',  # Normal
            
            # MI patterns - these are the key ones we want
            '164865005': 'MI',    # Myocardial infarction
            '22298006': 'MI',     # Myocardial infarction  
            '164861001': 'MI',    # Old myocardial infarction
            '164931005': 'MI',    # Anterior wall myocardial infarction
            '54329005': 'MI',     # Acute myocardial infarction
            '164930006': 'MI',    # Inferior wall myocardial infarction
            '164932003': 'MI',    # Lateral wall myocardial infarction
            
            # ST/T wave changes
            '164934002': 'STTC',  # ST elevation
            '59118001': 'STTC',   # ST depression
            '164937009': 'STTC',  # T wave abnormality
            '251146004': 'STTC',  # ST-T changes
            
            # Conduction disorders
            '426177001': 'CD',    # Bundle branch block
            '164909002': 'CD',    # Left bundle branch block
            '164912004': 'CD',    # Right bundle branch block
            '164889003': 'CD',    # AV block
            
            # Hypertrophy
            '164873001': 'HYP',   # Left ventricular hypertrophy
            '446358003': 'HYP',   # Right ventricular hypertrophy
            '164921003': 'HYP',   # Atrial hypertrophy
        }
    
    def parse_header_file(self, hea_path: Path) -> Dict[str, Any]:
        """Parse .hea file to extract metadata and Dx codes"""
        try:
            with open(hea_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            metadata = {
                'record_id': hea_path.stem,
                'age': None,
                'sex': None, 
                'dx_codes': [],
                'sampling_rate': 500,  # Default
                'n_leads': 12,
                'length': 5000
            }
            
            # Parse first line for basic info
            if lines:
                first_line = lines[0].strip().split()
                if len(first_line) >= 4:
                    metadata['n_leads'] = int(first_line[1])
                    metadata['sampling_rate'] = int(first_line[2])
                    metadata['length'] = int(first_line[3])
            
            # Parse metadata lines starting with #
            for line in lines:
                line = line.strip()
                if line.startswith('#Age:'):
                    try:
                        metadata['age'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('#Sex:'):
                    metadata['sex'] = line.split(':')[1].strip()
                elif line.startswith('#Dx:'):
                    dx_part = line.split(':')[1].strip()
                    # Split by comma and clean up codes
                    codes = [code.strip() for code in dx_part.split(',') if code.strip()]
                    metadata['dx_codes'] = codes
            
            return metadata
            
        except Exception as e:
            print(f"Error parsing header {hea_path}: {e}")
            return None
    
    def load_signal_file(self, mat_path: Path) -> Optional[np.ndarray]:
        """Load .mat signal file"""
        try:
            # Load MATLAB file
            mat_data = scipy.io.loadmat(str(mat_path))
            
            # Try different common variable names
            signal = None
            for key in ['val', 'data', 'ecg', 'signal']:
                if key in mat_data:
                    signal = mat_data[key]
                    break
            
            if signal is None:
                # Try the first non-system key
                non_system_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if non_system_keys:
                    signal = mat_data[non_system_keys[0]]
            
            if signal is not None:
                # Ensure proper shape: (time_steps, leads)
                if signal.ndim == 2:
                    if signal.shape[0] > signal.shape[1]:
                        return signal  # Already correct: (time, leads)
                    else:
                        return signal.T  # Transpose: (leads, time) -> (time, leads)
                elif signal.ndim == 1:
                    return signal.reshape(-1, 1)  # Single lead
            
            return None
            
        except Exception as e:
            print(f"Error loading signal {mat_path}: {e}")
            return None
    
    def map_dx_to_target(self, dx_codes: List[str]) -> Optional[str]:
        """Map SNOMED-CT diagnosis codes to target conditions"""
        # Prioritize MI detection
        for code in dx_codes:
            if code in self.dx_mapping and self.dx_mapping[code] == 'MI':
                return 'MI'
        
        # Then other conditions
        for code in dx_codes:
            if code in self.dx_mapping:
                return self.dx_mapping[code]
        
        return None  # Unknown condition
    
    def scan_for_records(self, max_records: int = 100, target_mi_records: int = 50) -> List[Dict]:
        """Scan dataset and find records with valid data"""
        print(f"Scanning for up to {max_records} records ({target_mi_records} MI target)...")
        
        found_records = []
        mi_count = 0
        total_checked = 0
        
        # Walk through all folders
        for folder in sorted(self.base_path.iterdir()):
            if not folder.is_dir():
                continue
                
            for subfolder in sorted(folder.iterdir()):
                if not subfolder.is_dir():
                    continue
                
                # Look for .hea files
                for hea_file in subfolder.glob("*.hea"):
                    if len(found_records) >= max_records:
                        break
                    
                    total_checked += 1
                    if total_checked % 100 == 0:
                        print(f"Checked {total_checked} files, found {len(found_records)} valid ({mi_count} MI)")
                    
                    # Parse header
                    metadata = self.parse_header_file(hea_file)
                    if not metadata or not metadata['dx_codes']:
                        continue
                    
                    # Map to target condition
                    target_condition = self.map_dx_to_target(metadata['dx_codes'])
                    if not target_condition:
                        continue
                    
                    # Check if corresponding .mat file exists
                    mat_file = hea_file.with_suffix('.mat')
                    if not mat_file.exists():
                        continue
                    
                    # Prioritize MI records
                    if target_condition == 'MI':
                        if mi_count < target_mi_records:
                            found_records.append({
                                'hea_path': hea_file,
                                'mat_path': mat_file,
                                'metadata': metadata,
                                'target_condition': target_condition
                            })
                            mi_count += 1
                    else:
                        # Add other conditions if we haven't hit the limit
                        if len(found_records) - mi_count < (max_records - target_mi_records):
                            found_records.append({
                                'hea_path': hea_file,
                                'mat_path': mat_file,
                                'metadata': metadata,
                                'target_condition': target_condition
                            })
                
                if len(found_records) >= max_records:
                    break
            
            if len(found_records) >= max_records:
                break
        
        print(f"Scan complete: Found {len(found_records)} valid records ({mi_count} MI)")
        return found_records
    
    def load_records(self, record_list: List[Dict], target_length: int = 1000) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load the actual signal data for selected records"""
        signals = []
        conditions = []
        record_ids = []
        
        print(f"Loading {len(record_list)} records...")
        
        for i, record_info in enumerate(record_list):
            if i % 10 == 0:
                print(f"Loading record {i+1}/{len(record_list)}")
            
            # Load signal
            signal = self.load_signal_file(record_info['mat_path'])
            if signal is None:
                continue
            
            # Ensure we have 12 leads
            if signal.shape[1] != 12:
                continue
            
            # Resample/truncate to target length
            if signal.shape[0] >= target_length:
                signal = signal[:target_length, :]
            else:
                # Pad if too short
                padding = np.zeros((target_length - signal.shape[0], signal.shape[1]))
                signal = np.vstack([signal, padding])
            
            signals.append(signal)
            conditions.append(record_info['target_condition'])
            record_ids.append(record_info['metadata']['record_id'])
        
        if signals:
            return np.array(signals), conditions, record_ids
        else:
            return np.array([]), [], []


def test_fixed_loader():
    """Test the fixed loader"""
    print("TESTING FIXED ECG ARRHYTHMIA LOADER")
    print("=" * 50)
    
    loader = FixedECGArrhythmiaLoader()
    
    # Test with small numbers first
    records = loader.scan_for_records(max_records=20, target_mi_records=10)
    
    if records:
        print(f"\nFound {len(records)} valid records")
        
        # Show condition distribution
        from collections import Counter
        conditions = [r['target_condition'] for r in records]
        print(f"Condition distribution: {Counter(conditions)}")
        
        # Load the signals
        X, labels, ids = loader.load_records(records[:10])  # Test with first 10
        
        if len(X) > 0:
            print(f"\nLoaded signals successfully!")
            print(f"Shape: {X.shape}")
            print(f"Labels: {Counter(labels)}")
            return X, labels, ids, records
        else:
            print("\nNo signals could be loaded")
            return None, None, None, records
    else:
        print("No valid records found")
        return None, None, None, []


if __name__ == "__main__":
    X, labels, ids, records = test_fixed_loader()
    
    if X is not None:
        print("\n" + "=" * 50)
        print("SUCCESS: Fixed loader working!")
        print("Ready to integrate with PTB-XL for MI enhancement")
    else:
        print("\n" + "=" * 50)
        print("Issue found - need to debug further")