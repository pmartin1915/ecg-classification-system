"""
Data loading utilities for ECG Classification System
Handles PTB-XL and ECG-Arrhythmia datasets
"""
import ast
import os
import pickle
import gc
import scipy.io
from pathlib import Path
from collections import Counter
from typing import Tuple, List, Optional, Dict, Any
import warnings

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed
from tqdm import tqdm

from config.settings import DATASET_CONFIG, TARGET_CONDITIONS, CACHE_DIR

warnings.filterwarnings('ignore')


class ECGDataLoader:
    """Main data loader for ECG datasets"""
    
    def __init__(self, dataset_name: str = "ptbxl"):
        self.dataset_name = dataset_name
        self.dataset_config = DATASET_CONFIG[dataset_name]
        self.data_path = self.dataset_config["path"]
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def download_metadata(self) -> Path:
        """Download metadata files if they don't exist"""
        print(f"=== DOWNLOADING {self.dataset_name.upper()} METADATA ===")
        
        if self.dataset_name == "ptbxl":
            return self._download_ptbxl_metadata()
        else:
            raise NotImplementedError(f"Download not implemented for {self.dataset_name}")
    
    def _download_ptbxl_metadata(self) -> Path:
        """Download PTB-XL metadata files"""
        metadata_files = self.dataset_config["metadata_files"]
        urls = self.dataset_config["urls"]
        
        # Check if files already exist
        db_file = self.data_path / metadata_files["database"]
        scp_file = self.data_path / metadata_files["scp_statements"]
        
        if db_file.exists() and scp_file.exists():
            print("OK: PTB-XL metadata files already exist")
            return self.data_path
        
        # Download files using urllib instead of wget for portability
        import urllib.request
        
        for file_key, filename in metadata_files.items():
            filepath = self.data_path / filename
            if not filepath.exists():
                print(f"Downloading {filename}...")
                url = urls[file_key]
                urllib.request.urlretrieve(url, filepath)
                print(f"OK: Downloaded {filename}")
        
        return self.data_path
    
    def load_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process metadata"""
        print("=== PROCESSING METADATA ===")
        
        if self.dataset_name == "ptbxl":
            return self._load_ptbxl_metadata()
        else:
            raise NotImplementedError(f"Metadata loading not implemented for {self.dataset_name}")
    
    def _load_ptbxl_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load PTB-XL metadata"""
        # Ensure metadata is downloaded
        self._download_ptbxl_metadata()
        
        # Load main database
        db_file = self.data_path / self.dataset_config["metadata_files"]["database"]
        scp_file = self.data_path / self.dataset_config["metadata_files"]["scp_statements"]
        
        print("Loading ptbxl_database.csv...")
        Y = pd.read_csv(db_file, index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
        
        # Load diagnostic mapping
        print("Loading scp_statements.csv...")
        agg_df = pd.read_csv(scp_file, index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        # Apply diagnostic aggregation
        Y['diagnostic_superclass'] = Y.scp_codes.apply(
            lambda y_dic: list(set(
                agg_df.loc[key].diagnostic_class for key in y_dic.keys()
                if key in agg_df.index
            ))
        )
        
        print(f"OK: Loaded metadata for {len(Y)} records")
        return Y, agg_df
    
    def filter_by_conditions(self, Y: pd.DataFrame, 
                           conditions: List[str] = None) -> pd.DataFrame:
        """Filter records by target conditions"""
        if conditions is None:
            conditions = TARGET_CONDITIONS
            
        print(f"=== FILTERING BY CONDITIONS: {conditions} ===")
        
        # Count conditions
        all_conditions = []
        for diagnostic_list in Y['diagnostic_superclass']:
            all_conditions.extend(diagnostic_list)
        
        condition_counts = Counter(all_conditions)
        
        print("\nCondition distribution:")
        for condition in conditions:
            count = condition_counts.get(condition, 0)
            print(f"  {condition}: {count:,} records")
        
        # Filter records
        def has_target_condition(diagnostic_list):
            return any(condition in diagnostic_list for condition in conditions)
        
        Y_filtered = Y[Y['diagnostic_superclass'].apply(has_target_condition)]
        
        print(f"\nOK: Filtered dataset: {len(Y_filtered):,} records")
        print(f"   Reduction: {((len(Y) - len(Y_filtered)) / len(Y) * 100):.1f}%")
        
        return Y_filtered
    
    def load_signals(self, df_subset: pd.DataFrame, 
                    sampling_rate: int = 100,
                    max_records: Optional[int] = None,
                    batch_size: int = 1000,
                    n_jobs: int = 4,
                    use_cache: bool = True) -> Tuple[np.ndarray, List, List]:
        """Load ECG signals with optimized parallel processing"""
        print("=== LOADING ECG SIGNAL DATA ===")
        
        # Limit records if specified
        if max_records is not None:
            df_subset = df_subset.head(max_records)
            print(f"Limited to {max_records} records")
        
        # Check cache
        cache_file = CACHE_DIR / f"signals_{self.dataset_name}_{sampling_rate}hz_{len(df_subset)}.pkl"
        
        if use_cache and cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    signals_array, labels, record_ids = pickle.load(f)
                
                # Verify cache integrity
                if len(signals_array) == len(df_subset):
                    print(f"OK: Cache loaded successfully with {len(signals_array)} records.")
                    return signals_array, labels, record_ids
                else:
                    print(f"WARNING: Cache mismatch. Expected {len(df_subset)} records, found {len(signals_array)}.")
                    print("Discarding invalid cache file.")
                    cache_file.unlink()
                    # Continue to load fresh data
                    
            except Exception as e:
                print(f"ERROR: Cache error: {e}. Deleting cache and loading fresh data.")
                if cache_file.exists():
                    cache_file.unlink()
        
        # Download missing files first
        self._download_missing_records(df_subset, sampling_rate)
        
        # Load signals in parallel
        signals, labels, record_ids = self._parallel_load_signals(
            df_subset, sampling_rate, batch_size, n_jobs
        )
        
        if len(signals) > 0:
            signals_array = np.array(signals, dtype=np.float32)
            print(f"OK: Loaded {len(signals):,} records")
            print(f"   Shape: {signals_array.shape}")
            print(f"   Memory: {signals_array.nbytes / (1024**3):.2f} GB")
            
            # Save to cache
            if use_cache:
                print(f"Saving to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump((signals_array, labels, record_ids), f)
            
            return signals_array, labels, record_ids
        else:
            print("❌ No signals loaded successfully!")
            return np.array([]), [], []
    
    def _download_missing_records(self, df_subset: pd.DataFrame, sampling_rate: int):
        """Download missing signal files"""
        print(f"Checking for missing signal files ({sampling_rate}Hz)...")
        
        filename_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
        base_url = self.dataset_config["urls"]["records_base"]
        
        # Create necessary directories
        unique_dirs = set()
        for file_path in df_subset[filename_col]:
            folder = self.data_path / Path(file_path).parent
            unique_dirs.add(folder)
        
        for folder in unique_dirs:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Check for missing files
        download_items = []
        for _, row in df_subset.iterrows():
            file_path = row[filename_col]
            base_path = self.data_path / file_path
            
            for ext in ['.hea', '.dat']:
                file_with_ext = Path(str(base_path) + ext)
                if not file_with_ext.exists():
                    url = f"{base_url}{file_path}{ext}"
                    download_items.append((url, file_with_ext))
        
        if not download_items:
            print("OK: All required files already exist")
            return
        
        # Download missing files
        import urllib.request
        print(f"Downloading {len(download_items)} missing files...")
        
        for url, filepath in tqdm(download_items, desc="Downloading"):
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
    
    def _load_single_record(self, args: Tuple) -> Tuple[int, Optional[np.ndarray], Optional[List]]:
        """Load a single ECG record"""
        record_id, row, filename_col = args
        
        try:
            file_path = self.data_path / row[filename_col]
            signal, _ = wfdb.rdsamp(str(file_path))
            return record_id, signal, row['diagnostic_superclass']
        except Exception as e:
            return record_id, None, None
    
    def _parallel_load_signals(self, df_subset: pd.DataFrame, 
                             sampling_rate: int,
                             batch_size: int,
                             n_jobs: int) -> Tuple[List, List, List]:
        """Load signals in parallel batches"""
        filename_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
        
        # Prepare arguments
        args_list = [(record_id, row, filename_col) 
                    for record_id, row in df_subset.iterrows()]
        
        signals = []
        labels = []
        record_ids = []
        failed_loads = 0
        
        # Process in batches
        for i in range(0, len(args_list), batch_size):
            batch_args = args_list[i:i+batch_size]
            
            # Parallel processing
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(self._load_single_record)(args) 
                for args in tqdm(batch_args, desc=f"Batch {i//batch_size + 1}", leave=False)
            )
            
            # Collect results
            for record_id, signal, label in results:
                if signal is not None:
                    signals.append(signal)
                    labels.append(label)
                    record_ids.append(record_id)
                else:
                    failed_loads += 1
            
            # Memory management
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        if failed_loads > 0:
            print(f"⚠️  Failed to load {failed_loads} records")
        
        return signals, labels, record_ids


class ArrhythmiaDataLoader(ECGDataLoader):
    """Specialized loader for MIT-BIH Arrhythmia dataset"""
    
    def __init__(self):
        super().__init__("ecg_arrhythmia")
    
    def find_all_records(self) -> List[Path]:
        """Recursively find all WFDB records in the dataset"""
        print("=== SCANNING FOR ARRHYTHMIA RECORDS ===")
        
        record_files = []
        base_path = self.data_path / "physionet.org" / "files" / "ecg-arrhythmia" / "1.0.0" / "WFDBRecords"
        
        if not base_path.exists():
            print(f"❌ Dataset path not found: {base_path}")
            return []
        
        # Find all .hea files
        for hea_file in base_path.rglob("*.hea"):
            record_files.append(hea_file.with_suffix(''))
        
        print(f"OK: Found {len(record_files)} records")
        return record_files
    
    def load_records_batch(self, record_paths: List[Path], 
                          batch_size: int = 100) -> Dict[str, np.ndarray]:
        """Load arrhythmia records in batches"""
        records = {}
        
        for i in range(0, len(record_paths), batch_size):
            batch = record_paths[i:i+batch_size]
            print(f"Loading batch {i//batch_size + 1}/{(len(record_paths)-1)//batch_size + 1}")
            
            for record_path in tqdm(batch, desc="Loading records"):
                try:
                    signal, fields = wfdb.rdsamp(str(record_path))
                    record_name = record_path.stem
                    records[record_name] = {
                        'signal': signal,
                        'fields': fields
                    }
                except Exception as e:
                    print(f"Error loading {record_path}: {e}")
            
            gc.collect()
        
        return records


class ECGArrhythmiaDataLoader:
    """
    Enhanced loader for PhysioNet ECG Arrhythmia dataset (45,152 records)
    Supports .mat + .hea files with comprehensive label mapping
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or DATASET_CONFIG["ecg_arrhythmia"]["path"]
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Label mapping from ECG Arrhythmia to our TARGET_CONDITIONS
        self.label_mapping = {
            # MI-related conditions (critical for improving MI detection)
            'MI': 'MI',  # Myocardial Infarction
            'STEMI': 'MI',  # ST-elevation MI
            'NSTEMI': 'MI',  # Non-ST-elevation MI
            'IMI': 'MI',  # Inferior MI
            'AMI': 'MI',  # Anterior MI
            'LMI': 'MI',  # Lateral MI
            
            # Normal conditions
            'NORM': 'NORM',  # Normal sinus rhythm
            'NSR': 'NORM',  # Normal sinus rhythm
            'SB': 'NORM',    # Sinus bradycardia (can be normal)
            'ST': 'NORM',    # Sinus tachycardia (can be normal)
            
            # ST/T Changes
            'STTC': 'STTC',  # ST/T Changes
            'STD': 'STTC',   # ST Depression
            'STE': 'STTC',   # ST Elevation
            'INVT': 'STTC',  # Inverted T waves
            'TAB': 'STTC',   # T wave abnormality
            
            # Conduction Disorders
            'CD': 'CD',      # Conduction Disorders
            'RBBB': 'CD',    # Right bundle branch block
            'LBBB': 'CD',    # Left bundle branch block
            'BBB': 'CD',     # Bundle branch block
            'IVCD': 'CD',    # Intraventricular conduction delay
            'AVB': 'CD',     # AV block
            'WPW': 'CD',     # Wolff-Parkinson-White
            
            # Hypertrophy
            'HYP': 'HYP',    # Hypertrophy
            'LVH': 'HYP',    # Left ventricular hypertrophy
            'RVH': 'HYP',    # Right ventricular hypertrophy
            'LAE': 'HYP',    # Left atrial enlargement
            'RAE': 'HYP',    # Right atrial enlargement
            
            # Arrhythmias - map to closest category or exclude
            'AFIB': 'CD',    # Atrial fibrillation -> Conduction disorder
            'AFL': 'CD',     # Atrial flutter -> Conduction disorder
            'AT': 'CD',      # Atrial tachycardia -> Conduction disorder
            'PVC': 'CD',     # Premature ventricular contraction -> Conduction disorder
            'PAC': 'CD',     # Premature atrial contraction -> Conduction disorder
        }
    
    def scan_dataset_structure(self) -> Dict[str, Any]:
        """Scan the dataset to understand structure and available records"""
        print("=== SCANNING ECG ARRHYTHMIA DATASET ===")
        
        if not self.data_path.exists():
            print(f"❌ Dataset path not found: {self.data_path}")
            return {"total_records": 0, "folders": [], "sample_files": []}
        
        # Look for WFDBRecords folder structure
        wfdb_path = self.data_path / "WFDBRecords"
        if not wfdb_path.exists():
            # Try alternative paths
            for alt_path in ["physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords", 
                           "ecg-arrhythmia/1.0.0/WFDBRecords"]:
                alt_full_path = self.data_path / alt_path
                if alt_full_path.exists():
                    wfdb_path = alt_full_path
                    break
        
        if not wfdb_path.exists():
            print(f"❌ WFDBRecords folder not found in: {self.data_path}")
            return {"total_records": 0, "folders": [], "sample_files": []}
        
        # Scan folder structure
        folders = [f for f in wfdb_path.iterdir() if f.is_dir()]
        folders.sort()
        
        # Count total records
        total_records = 0
        sample_files = []
        
        for folder in folders[:3]:  # Check first 3 folders for structure
            subfolders = [sf for sf in folder.iterdir() if sf.is_dir()]
            for subfolder in subfolders[:2]:  # Check first 2 subfolders
                mat_files = list(subfolder.glob("*.mat"))
                hea_files = list(subfolder.glob("*.hea"))
                total_records += len(mat_files)
                
                if sample_files == [] and mat_files:
                    sample_files = [mat_files[0], hea_files[0] if hea_files else None]
        
        # Estimate total based on structure (46 folders × 10 subfolders × ~100 records)
        estimated_total = len(folders) * 10 * 100
        
        print(f"OK: Found {len(folders)} top-level folders")
        print(f"OK: Estimated {estimated_total} total records")
        
        return {
            "total_records": estimated_total,
            "folders": folders,
            "sample_files": sample_files,
            "wfdb_path": wfdb_path
        }
    
    def load_single_record_wfdb(self, record_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single WFDB record (.mat + .hea)"""
        try:
            # Use wfdb.rdsamp which handles both .dat and .mat files
            record_name = str(record_path.with_suffix(''))
            signal, fields = wfdb.rdsamp(record_name)
            
            # Extract metadata from header
            annotation = wfdb.rdann(record_name, 'atr') if Path(f"{record_name}.atr").exists() else None
            
            return {
                'signal': signal,
                'fields': fields,
                'annotation': annotation,
                'record_id': record_path.stem,
                'sampling_rate': fields.get('fs', 500),  # Default 500Hz for this dataset
                'n_leads': signal.shape[1] if signal.ndim > 1 else 1,
                'duration': len(signal) / fields.get('fs', 500)
            }
            
        except Exception as e:
            print(f"Error loading {record_path}: {e}")
            return None
    
    def extract_labels_from_header(self, header_info: Dict) -> List[str]:
        """Extract diagnostic labels from WFDB header information"""
        labels = []
        
        # Extract from comments field which often contains diagnostic info
        if 'comments' in header_info:
            comments = header_info['comments']
            if isinstance(comments, list):
                for comment in comments:
                    # Look for common diagnostic patterns
                    comment_upper = comment.upper()
                    for label_key in self.label_mapping.keys():
                        if label_key in comment_upper:
                            labels.append(label_key)
        
        # Extract from signal names if available
        if 'sig_name' in header_info:
            sig_names = header_info['sig_name']
            if isinstance(sig_names, list):
                for sig_name in sig_names:
                    sig_name_upper = sig_name.upper()
                    for label_key in self.label_mapping.keys():
                        if label_key in sig_name_upper:
                            labels.append(label_key)
        
        return list(set(labels))  # Remove duplicates
    
    def map_labels_to_target_conditions(self, raw_labels: List[str]) -> List[str]:
        """Map raw diagnostic labels to our TARGET_CONDITIONS"""
        mapped_labels = []
        
        for label in raw_labels:
            if label in self.label_mapping:
                target_label = self.label_mapping[label]
                if target_label in TARGET_CONDITIONS:
                    mapped_labels.append(target_label)
        
        # Default to NORM if no specific condition found
        if not mapped_labels:
            mapped_labels = ['NORM']
        
        return list(set(mapped_labels))  # Remove duplicates
    
    def load_records_batch(self, max_records: Optional[int] = None, 
                          sampling_rate: int = 500,
                          use_cache: bool = True,
                          target_mi_records: int = 1000) -> Dict[str, Any]:
        """
        Load ECG Arrhythmia records with focus on MI detection
        
        Args:
            max_records: Maximum total records to load
            sampling_rate: Target sampling rate (500Hz native)
            use_cache: Use cached results if available
            target_mi_records: Minimum MI records to collect
        """
        cache_key = f"ecg_arrhythmia_{sampling_rate}hz_{max_records}_{target_mi_records}mi"
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        if use_cache and cache_file.exists():
            print(f"OK: Loading ECG Arrhythmia data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("=== LOADING ECG ARRHYTHMIA DATASET ===")
        
        # Scan dataset structure
        structure = self.scan_dataset_structure()
        if structure["total_records"] == 0:
            return {"X": np.array([]), "labels": [], "record_ids": [], "stats": {}}
        
        wfdb_path = structure["wfdb_path"]
        folders = structure["folders"]
        
        # Storage for results
        signals = []
        labels = []
        record_ids = []
        mi_count = 0
        total_loaded = 0
        
        print(f"Target: {target_mi_records} MI records, {max_records or 'unlimited'} total records")
        
        # Iterate through folder structure
        for folder in tqdm(folders, desc="Processing folders"):
            if max_records and total_loaded >= max_records:
                break
                
            subfolders = [sf for sf in folder.iterdir() if sf.is_dir()]
            
            for subfolder in subfolders:
                if max_records and total_loaded >= max_records:
                    break
                    
                # Find all .mat files in this subfolder
                mat_files = list(subfolder.glob("*.mat"))
                
                for mat_file in mat_files:
                    if max_records and total_loaded >= max_records:
                        break
                        
                    # Load the record
                    record_data = self.load_single_record_wfdb(mat_file)
                    
                    if record_data is None:
                        continue
                    
                    # Extract and map labels
                    raw_labels = self.extract_labels_from_header(record_data['fields'])
                    mapped_labels = self.map_labels_to_target_conditions(raw_labels)
                    
                    # Prioritize MI records
                    has_mi = 'MI' in mapped_labels
                    if has_mi:
                        mi_count += 1
                    
                    # Include record if it has MI or we haven't reached max_records
                    if has_mi or mi_count >= target_mi_records or not max_records:
                        # Resample if needed
                        signal = record_data['signal']
                        current_fs = record_data['sampling_rate']
                        
                        if current_fs != sampling_rate:
                            from scipy import signal as scipy_signal
                            # Resample to target sampling rate
                            new_length = int(len(signal) * sampling_rate / current_fs)
                            signal = scipy_signal.resample(signal, new_length, axis=0)
                        
                        # Ensure consistent shape (12 leads)
                        if signal.ndim == 1:
                            signal = signal.reshape(-1, 1)
                        
                        if signal.shape[1] < 12:
                            # Pad with zeros if fewer than 12 leads
                            padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
                            signal = np.hstack([signal, padding])
                        elif signal.shape[1] > 12:
                            # Take first 12 leads if more than 12
                            signal = signal[:, :12]
                        
                        signals.append(signal)
                        labels.append(mapped_labels[0])  # Take primary label
                        record_ids.append(record_data['record_id'])
                        total_loaded += 1
                    
                    # Early termination if we have enough MI records
                    if mi_count >= target_mi_records and total_loaded >= (max_records or target_mi_records * 2):
                        break
        
        # Convert to arrays
        if signals:
            X = np.array(signals)
            print(f"OK: Loaded {len(signals)} records ({mi_count} MI records)")
        else:
            X = np.array([])
            print("❌ No records loaded")
        
        # Statistics
        stats = {
            "total_loaded": total_loaded,
            "mi_records": mi_count,
            "label_distribution": Counter(labels),
            "avg_signal_length": np.mean([len(s) for s in signals]) if signals else 0,
            "sampling_rate": sampling_rate
        }
        
        result = {
            "X": X,
            "labels": labels,
            "record_ids": record_ids,
            "stats": stats
        }
        
        # Cache results
        if use_cache and signals:
            print(f"Caching results to: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        
        return result