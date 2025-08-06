"""
Enhanced MI Database Loader for Large-Scale ECG Database
Integrates 45,152 physician-validated ECG records with MI focus
"""
import os
import pickle
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings

from config.settings import CACHE_DIR

warnings.filterwarnings('ignore')


class MIDatabaseLoader:
    """
    Professional loader for Large-Scale 12-Lead ECG Database for Arrhythmia Study
    Optimized for MI detection enhancement with clinical-grade data handling
    """
    
    def __init__(self, database_path: str):
        """Initialize MI database loader
        
        Args:
            database_path: Path to the extracted database folder
        """
        self.database_path = Path(database_path)
        self.wfdb_records_path = self.database_path / "WFDBRecords"
        self.conditions_file = self.database_path / "ConditionNames_SNOMED-CT.csv"
        
        # Load condition mappings
        self.condition_mappings = self._load_condition_mappings()
        self.mi_focused_mappings = self._create_mi_focused_mappings()
        
        # Verify database structure
        self._verify_database_structure()
        
    def _verify_database_structure(self) -> None:
        """Verify database structure and report statistics"""
        if not self.wfdb_records_path.exists():
            raise FileNotFoundError(f"WFDBRecords not found at: {self.wfdb_records_path}")
            
        if not self.conditions_file.exists():
            raise FileNotFoundError(f"Conditions file not found at: {self.conditions_file}")
            
        # Count total records
        total_records = sum(1 for _ in self.wfdb_records_path.rglob("*.hea"))
        print(f"[OK] MI Database verified: {total_records:,} ECG records available")
        
    def _load_condition_mappings(self) -> Dict[str, str]:
        """Load SNOMED-CT condition mappings"""
        try:
            df = pd.read_csv(self.conditions_file, encoding='utf-8-sig')
            mappings = dict(zip(df['Snomed_CT'].astype(str), df['Acronym Name']))
            print(f"[OK] Loaded {len(mappings)} condition mappings")
            return mappings
        except Exception as e:
            print(f"[WARNING] Could not load condition mappings: {e}")
            return {}
    
    def _create_mi_focused_mappings(self) -> Dict[str, str]:
        """Create enhanced MI-focused label mappings for 5-class system"""
        return {
            # MI Types - All map to MI class
            'MI': 'MI',
            'MIBW': 'MI',    # MI back wall
            'MIFW': 'MI',    # MI front wall  
            'MILW': 'MI',    # MI lower wall
            'MISW': 'MI',    # MI side wall
            'AMI': 'MI',     # Acute MI
            'IMI': 'MI',     # Inferior MI
            'LMI': 'MI',     # Lateral MI
            'PMI': 'MI',     # Posterior MI
            
            # ST/T Changes
            'STTC': 'STTC',
            'TWC': 'STTC',   # T wave changes
            'STD': 'STTC',   # ST depression  
            'STE': 'STTC',   # ST elevation
            
            # Conduction Disorders
            'CD': 'CD',
            'RBBB': 'CD',    # Right bundle branch block
            'LBBB': 'CD',    # Left bundle branch block
            'IVCB': 'CD',    # Intraventricular conduction block
            '1AVB': 'CD',    # First degree AV block
            '2AVB': 'CD',    # Second degree AV block
            '3AVB': 'CD',    # Third degree AV block
            'AVB': 'CD',     # AV block
            'IVB': 'CD',     # Intraventricular block
            
            # Hypertrophy
            'HYP': 'HYP',
            'LVH': 'HYP',    # Left ventricular hypertrophy
            'RVH': 'HYP',    # Right ventricular hypertrophy
            'RAE': 'HYP',    # Right atrial enlargement
            'LAE': 'HYP',    # Left atrial enlargement
            
            # Normal and other conditions map to NORM
            'NORM': 'NORM',
            'NSR': 'NORM',   # Normal sinus rhythm
        }
    
    def extract_diagnosis_codes(self, header_file: Path) -> List[str]:
        """Extract diagnosis codes from ECG header file"""
        try:
            with open(header_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            diagnosis_codes = []
            for line in lines:
                if line.strip().startswith('#Dx:'):
                    dx_codes = line.replace('#Dx:', '').strip()
                    if dx_codes and dx_codes != 'Unknown':
                        # Split multiple codes
                        codes = [code.strip() for code in dx_codes.split(',')]
                        diagnosis_codes.extend(codes)
                    break
                    
            return diagnosis_codes
        except Exception as e:
            return []
    
    def load_ecg_signal(self, record_path: Path) -> Optional[np.ndarray]:
        """Load ECG signal from .mat file"""
        try:
            mat_data = scipy.io.loadmat(record_path)
            
            # Find ECG data - usually stored as 'val' or 'data'
            ecg_data = None
            for key in ['val', 'data', 'ecg', 'signal']:
                if key in mat_data:
                    ecg_data = mat_data[key]
                    break
            
            if ecg_data is None:
                # Try to find the largest numerical array
                for key, value in mat_data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray) and value.dtype in [np.float64, np.float32, np.int16, np.int32]:
                        if ecg_data is None or value.size > ecg_data.size:
                            ecg_data = value
            
            if ecg_data is not None:
                # Convert to standard format: (n_leads, n_samples)
                ecg_data = np.array(ecg_data, dtype=np.float32)
                
                # Handle different orientations
                if ecg_data.shape[0] > ecg_data.shape[1]:
                    ecg_data = ecg_data.T
                    
                return ecg_data
                
        except Exception as e:
            return None
            
        return None
    
    def map_diagnoses_to_labels(self, diagnosis_codes: List[str]) -> List[str]:
        """Map diagnosis codes to target condition labels"""
        labels = []
        
        for code in diagnosis_codes:
            # First check if code maps to condition name
            if code in self.condition_mappings:
                condition_name = self.condition_mappings[code]
                
                # Then map condition to our 5-class system
                if condition_name in self.mi_focused_mappings:
                    labels.append(self.mi_focused_mappings[condition_name])
        
        # If no specific labels found, check for MI diagnosis code directly
        if not labels:
            if '164865005' in diagnosis_codes:  # MI SNOMED code
                labels.append('MI')
        
        # Default to NORM if no condition identified
        if not labels:
            labels.append('NORM')
            
        return list(set(labels))  # Remove duplicates
    
    def load_mi_enhanced_dataset(self, 
                                max_records: int = 10000, 
                                target_mi_records: int = 2000,
                                target_sampling_rate: int = 100,
                                use_cache: bool = True) -> Tuple[np.ndarray, List[str], List[str], Dict]:
        """
        Load MI-enhanced dataset with focus on MI detection
        
        Args:
            max_records: Maximum total records to load
            target_mi_records: Target number of MI records to include
            target_sampling_rate: Resample signals to this rate
            use_cache: Use cached results if available
            
        Returns:
            signals: ECG signals array (n_samples, n_leads, signal_length)
            labels: Target condition labels 
            record_ids: Record identifiers
            metadata: Loading statistics and information
        """
        
        cache_file = CACHE_DIR / f"mi_enhanced_dataset_{max_records}_{target_mi_records}_{target_sampling_rate}.pkl"
        
        if use_cache and cache_file.exists():
            print("[CACHE] Loading from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"[LOADING] MI-enhanced dataset (max: {max_records:,}, target MI: {target_mi_records:,})")
        
        signals = []
        labels = []
        record_ids = []
        mi_count = 0
        other_count = 0
        
        # Get all record files
        all_record_files = []
        for mat_file in self.wfdb_records_path.rglob("*.mat"):
            header_file = mat_file.with_suffix('.hea')
            if header_file.exists():
                all_record_files.append((mat_file, header_file))
        
        print(f"[INFO] Found {len(all_record_files):,} complete record pairs")
        
        # Separate MI and non-MI records for balanced sampling
        mi_records = []
        other_records = []
        
        print(f"[SCANNING] Pre-scanning for MI records...")
        scan_limit = min(len(all_record_files), max_records * 5)  # Scan more files to find MI
        
        for mat_file, header_file in tqdm(all_record_files[:scan_limit], desc="Scanning for MI"):
            diagnosis_codes = self.extract_diagnosis_codes(header_file)
            record_labels = self.map_diagnoses_to_labels(diagnosis_codes)
            
            if 'MI' in record_labels:
                mi_records.append((mat_file, header_file, record_labels))
            else:
                other_records.append((mat_file, header_file, record_labels))
                
            # Stop early if we have enough MI records
            if len(mi_records) >= target_mi_records:
                break
        
        print(f"[FOUND] {len(mi_records)} MI records, {len(other_records)} other records")
        
        # Combine records with MI priority
        selected_records = mi_records[:target_mi_records] + other_records[:(max_records - len(mi_records))]
        np.random.shuffle(selected_records)  # Shuffle the selected records
        
        # Process selected records
        for mat_file, header_file, record_labels in tqdm(selected_records, desc="Loading records"):
            if len(signals) >= max_records:
                break
                
            # Use pre-determined labels
            has_mi = 'MI' in record_labels
            
            if has_mi:
                if mi_count >= target_mi_records:
                    continue
            else:
                if other_count >= (max_records - target_mi_records):
                    continue
            
            # Load ECG signal
            signal = self.load_ecg_signal(mat_file)
            if signal is None:
                continue
                
            # Ensure 12-lead format
            if signal.shape[0] < 12:
                continue
            signal = signal[:12]  # Take first 12 leads
            
            # Resample if needed (from 500Hz to target rate)
            original_length = signal.shape[1]
            if target_sampling_rate != 500:
                target_length = int(original_length * target_sampling_rate / 500)
                signal_resampled = np.zeros((12, target_length))
                for lead in range(12):
                    indices = np.linspace(0, original_length - 1, target_length)
                    signal_resampled[lead] = np.interp(indices, np.arange(original_length), signal[lead])
                signal = signal_resampled
            
            # Standardize length (10 seconds at target sampling rate)
            target_length = target_sampling_rate * 10
            if signal.shape[1] > target_length:
                signal = signal[:, :target_length]
            elif signal.shape[1] < target_length:
                padding = target_length - signal.shape[1]
                signal = np.pad(signal, ((0, 0), (0, padding)), mode='constant', constant_values=0)
            
            # Add to dataset
            signals.append(signal.T)  # Transpose to (time_samples, leads)
            labels.append(record_labels[0])  # Take primary label
            record_ids.append(mat_file.stem)
            
            if has_mi:
                mi_count += 1
            else:
                other_count += 1
        
        # Convert to arrays
        signals = np.array(signals, dtype=np.float32)
        
        # Prepare metadata
        metadata = {
            'total_records': len(signals),
            'mi_records': mi_count,
            'other_records': other_count,
            'mi_percentage': (mi_count / len(signals)) * 100 if len(signals) > 0 else 0,
            'sampling_rate': target_sampling_rate,
            'signal_length': signals.shape[1] if len(signals) > 0 else 0,
            'n_leads': 12,
            'database_source': 'Large-Scale ECG Database for Arrhythmia Study',
            'label_distribution': {label: labels.count(label) for label in set(labels)}
        }
        
        print(f"[SUCCESS] Loaded {len(signals):,} records ({mi_count} MI, {other_count} other)")
        print(f"[STATS] MI percentage: {metadata['mi_percentage']:.1f}%")
        
        # Cache results
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump((signals, labels, record_ids, metadata), f)
            print(f"[CACHE] Results cached to: {cache_file}")
        
        return signals, labels, record_ids, metadata


def test_mi_database_loader():
    """Test the MI database loader functionality"""
    
    # Initialize loader
    database_path = "C:\\ecg-classification-system-pc\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
    
    try:
        loader = MIDatabaseLoader(database_path)
        
        # Test with larger sample to find MI records (0.27% of database)
        signals, labels, record_ids, metadata = loader.load_mi_enhanced_dataset(
            max_records=1000,
            target_mi_records=50,
            target_sampling_rate=100,
            use_cache=False
        )
        
        print(f"[SUCCESS] Test successful!")
        print(f"[STATS] Loaded {len(signals)} signals with shape {signals.shape}")
        print(f"[LABELS] Label distribution: {metadata['label_distribution']}")
        print(f"[MI] MI records: {metadata['mi_records']} ({metadata['mi_percentage']:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False


if __name__ == "__main__":
    test_mi_database_loader()