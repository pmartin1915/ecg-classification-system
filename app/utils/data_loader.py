"""
Data loading utilities for ECG Classification System
Handles PTB-XL and ECG-Arrhythmia datasets
"""
import ast
import os
import pickle
import gc
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
            print("✅ PTB-XL metadata files already exist")
            return self.data_path
        
        # Download files using urllib instead of wget for portability
        import urllib.request
        
        for file_key, filename in metadata_files.items():
            filepath = self.data_path / filename
            if not filepath.exists():
                print(f"Downloading {filename}...")
                url = urls[file_key]
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ Downloaded {filename}")
        
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
        
        print(f"✅ Loaded metadata for {len(Y)} records")
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
        
        print(f"\n✅ Filtered dataset: {len(Y_filtered):,} records")
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
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache error: {e}. Loading fresh data.")
                cache_file.unlink()
        
        # Download missing files first
        self._download_missing_records(df_subset, sampling_rate)
        
        # Load signals in parallel
        signals, labels, record_ids = self._parallel_load_signals(
            df_subset, sampling_rate, batch_size, n_jobs
        )
        
        if len(signals) > 0:
            signals_array = np.array(signals, dtype=np.float32)
            print(f"✅ Loaded {len(signals):,} records")
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
            print("✅ All required files already exist")
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
        
        print(f"✅ Found {len(record_files)} records")
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