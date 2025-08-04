"""
Optimized High-Performance Dataset Loader
Handles complete PTB-XL and ECG Arrhythmia datasets with intelligent caching and parallel processing
"""

import os
import gc
import time
import pickle
import asyncio
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import wfdb
import scipy.io
from tqdm import tqdm
from joblib import Parallel, delayed, Memory

from config.settings import DATASET_CONFIG, TARGET_CONDITIONS, CACHE_DIR, DATA_DIR

warnings.filterwarnings('ignore')

@dataclass
class DatasetStats:
    """Statistics for dataset processing"""
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    processing_time: float = 0.0
    conditions_found: Dict[str, int] = None
    memory_usage_mb: float = 0.0
    
    def __post_init__(self):
        if self.conditions_found is None:
            self.conditions_found = {}

class OptimizedDatasetLoader:
    """High-performance dataset loader with intelligent caching and parallel processing"""
    
    def __init__(self, use_multiprocessing: bool = True, max_workers: Optional[int] = None):
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Limit to prevent memory issues
        
        # Setup paths
        self.arrhythmia_path = Path("C:/ecg-classification-system-pc/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0")
        self.ptbxl_path = DATA_DIR / "raw" / "ptbxl"
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup joblib memory for caching
        self.memory = Memory(location=str(self.cache_dir / "joblib_cache"), verbose=0)
        
        # Load condition mappings
        self.condition_mappings = self._load_condition_mappings()
        
        print(f"🚀 Optimized Dataset Loader initialized")
        print(f"   • Max workers: {self.max_workers}")
        print(f"   • Multiprocessing: {self.use_multiprocessing}")
        print(f"   • Cache directory: {self.cache_dir}")
    
    def _load_condition_mappings(self) -> Dict[str, str]:
        """Load condition mappings from SNOMED-CT file"""
        try:
            condition_file = self.arrhythmia_path / "ConditionNames_SNOMED-CT.csv"
            if condition_file.exists():
                df = pd.read_csv(condition_file)
                # Create mapping from acronym to target conditions
                mappings = {}
                
                # Enhanced MI mapping
                mi_conditions = ['MI', 'MIBW', 'MIFW', 'AMI', 'IMI', 'LMI', 'PMI']
                for condition in mi_conditions:
                    mappings[condition] = 'MI'
                
                # Arrhythmia mappings
                arrhythmia_map = {
                    'AFIB': 'AFIB', 'AFLT': 'AFLT', 'AFL': 'AFLT',
                    'VT': 'VTAC', 'VTAC': 'VTAC', 'VF': 'VTAC',
                    'SVT': 'SVTAC', 'SVTAC': 'SVTAC',
                    'APB': 'PAC', 'PAC': 'PAC',
                    'VPB': 'PVC', 'PVC': 'PVC'
                }
                mappings.update(arrhythmia_map)
                
                # Conduction disorders
                conduction_map = {
                    '1AVB': 'AVB1', '2AVB': 'AVB2', '3AVB': 'AVB3',
                    'LBBB': 'LBBB', 'RBBB': 'RBBB',
                    'LAFB': 'LAFB', 'LPFB': 'LPFB',
                    'IVCD': 'IVCD', 'IVB': 'IVCD'
                }
                mappings.update(conduction_map)
                
                # Hypertrophy and structural
                structural_map = {
                    'LVH': 'LVH', 'RVH': 'RVH',
                    'LAE': 'LAE', 'RAE': 'RAE'
                }
                mappings.update(structural_map)
                
                return mappings
            else:
                print(f"⚠️ Condition mapping file not found: {condition_file}")
                return {}
        except Exception as e:
            print(f"⚠️ Error loading condition mappings: {e}")
            return {}
    
    def load_complete_arrhythmia_dataset(self, 
                                       max_records: Optional[int] = None,
                                       sampling_rate: int = 100,
                                       target_conditions: Optional[List[str]] = None,
                                       progress_callback: Optional[callable] = None) -> Tuple[np.ndarray, List[str], List[str], DatasetStats]:
        """
        Load complete ECG Arrhythmia dataset with optimized processing
        
        Args:
            max_records: Maximum number of records to load (None for all)
            sampling_rate: Target sampling rate for signals
            target_conditions: List of target conditions to focus on
            progress_callback: Callback function for progress updates
            
        Returns:
            Tuple of (signals, labels, record_ids, stats)
        """
        cache_key = f"arrhythmia_complete_{max_records}_{sampling_rate}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache first
        if cache_file.exists():
            print(f"📁 Loading from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}")
        
        print(f"🔄 Loading complete ECG Arrhythmia dataset...")
        print(f"   • Target records: {max_records or 'ALL'}")
        print(f"   • Sampling rate: {sampling_rate} Hz")
        
        start_time = time.time()
        stats = DatasetStats()
        
        # Get all record directories
        record_dirs = self._get_arrhythmia_record_directories()
        
        if max_records:
            record_dirs = record_dirs[:max_records]
        
        stats.total_records = len(record_dirs)
        print(f"   • Found {stats.total_records} record directories")
        
        # Process records in parallel
        signals_list = []
        labels_list = []
        ids_list = []
        
        if self.use_multiprocessing and len(record_dirs) > 100:
            # Use multiprocessing for large datasets
            results = self._process_arrhythmia_records_parallel(record_dirs, sampling_rate, progress_callback)
        else:
            # Use single process for smaller datasets or when multiprocessing is disabled
            results = self._process_arrhythmia_records_sequential(record_dirs, sampling_rate, progress_callback)
        
        # Collect results
        for signal, label, record_id in results:
            if signal is not None:
                signals_list.append(signal)
                labels_list.append(label)
                ids_list.append(record_id)
                stats.processed_records += 1
                
                # Update condition counts
                if label in stats.conditions_found:
                    stats.conditions_found[label] += 1
                else:
                    stats.conditions_found[label] = 1
            else:
                stats.failed_records += 1
        
        # Convert to arrays
        if signals_list:
            signals = np.array(signals_list)
            labels = labels_list
            ids = ids_list
        else:
            signals = np.array([])
            labels = []
            ids = []
        
        stats.processing_time = time.time() - start_time
        stats.memory_usage_mb = signals.nbytes / (1024 * 1024) if signals.size > 0 else 0
        
        print(f"✅ Arrhythmia dataset loaded successfully!")
        print(f"   • Processed: {stats.processed_records}/{stats.total_records}")
        print(f"   • Failed: {stats.failed_records}")
        print(f"   • Processing time: {stats.processing_time:.1f}s")
        print(f"   • Memory usage: {stats.memory_usage_mb:.1f} MB")
        print(f"   • Signal shape: {signals.shape if signals.size > 0 else 'Empty'}")
        
        # Cache results
        result = (signals, labels, ids, stats)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"💾 Results cached to: {cache_file}")
        except Exception as e:
            print(f"⚠️ Caching failed: {e}")
        
        return result
    
    def _get_arrhythmia_record_directories(self) -> List[Path]:
        """Get all record directories from the arrhythmia dataset"""
        records_file = self.arrhythmia_path / "RECORDS"
        
        if not records_file.exists():
            print(f"❌ RECORDS file not found: {records_file}")
            return []
        
        directories = []
        with open(records_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    dir_path = self.arrhythmia_path / line
                    if dir_path.exists():
                        directories.append(dir_path)
        
        return directories
    
    def _process_arrhythmia_records_parallel(self, 
                                           record_dirs: List[Path], 
                                           sampling_rate: int,
                                           progress_callback: Optional[callable] = None) -> List[Tuple]:
        """Process arrhythmia records using parallel processing"""
        print(f"🔄 Processing {len(record_dirs)} records in parallel...")
        
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_dir = {
                executor.submit(self._process_single_arrhythmia_directory, dir_path, sampling_rate): dir_path 
                for dir_path in record_dirs
            }
            
            # Collect results with progress bar
            with tqdm(total=len(record_dirs), desc="Processing records") as pbar:
                for future in as_completed(future_to_dir):
                    result = future.result()
                    if result:
                        results.extend(result)
                    pbar.update(1)
                    
                    if progress_callback:
                        progress_callback(pbar.n / len(record_dirs))
        
        return results
    
    def _process_arrhythmia_records_sequential(self, 
                                             record_dirs: List[Path], 
                                             sampling_rate: int,
                                             progress_callback: Optional[callable] = None) -> List[Tuple]:
        """Process arrhythmia records sequentially"""
        print(f"🔄 Processing {len(record_dirs)} records sequentially...")
        
        results = []
        with tqdm(total=len(record_dirs), desc="Processing records") as pbar:
            for dir_path in record_dirs:
                try:
                    dir_results = self._process_single_arrhythmia_directory(dir_path, sampling_rate)
                    if dir_results:
                        results.extend(dir_results)
                except Exception as e:
                    print(f"⚠️ Error processing {dir_path}: {e}")
                
                pbar.update(1)
                if progress_callback:
                    progress_callback(pbar.n / len(record_dirs))
        
        return results
    
    @staticmethod
    def _process_single_arrhythmia_directory(dir_path: Path, sampling_rate: int) -> List[Tuple]:
        """Process a single arrhythmia directory containing multiple records"""
        results = []
        
        try:
            # Find all .hea files in the directory
            hea_files = list(dir_path.glob("*.hea"))
            
            for hea_file in hea_files:
                try:
                    record_name = hea_file.stem
                    mat_file = dir_path / f"{record_name}.mat"
                    
                    if not mat_file.exists():
                        continue
                    
                    # Load header to get condition information
                    header = wfdb.rdheader(str(hea_file).replace('.hea', ''))
                    
                    # Extract condition from comments
                    condition = 'UNKNOWN'
                    if hasattr(header, 'comments') and header.comments:
                        for comment in header.comments:
                            if 'Dx:' in comment:
                                # Extract diagnosis
                                dx_part = comment.split('Dx:')[1].strip()
                                condition = dx_part.split(',')[0].strip()  # Take first condition
                                break
                    
                    # Load signal data
                    try:
                        mat_data = scipy.io.loadmat(str(mat_file))
                        
                        # Find the ECG data (usually stored as 'val' or 'ECG')
                        signal_data = None
                        for key in ['val', 'ECG', 'data', 'signal']:
                            if key in mat_data:
                                signal_data = mat_data[key]
                                break
                        
                        if signal_data is None:
                            # Try the largest array
                            arrays = [(k, v) for k, v in mat_data.items() 
                                    if isinstance(v, np.ndarray) and v.ndim >= 2]
                            if arrays:
                                signal_data = max(arrays, key=lambda x: x[1].size)[1]
                        
                        if signal_data is not None:
                            # Ensure correct shape (leads x samples)
                            if signal_data.shape[0] > signal_data.shape[1]:
                                signal_data = signal_data.T
                            
                            # Resample if needed (simplified - just truncate or pad)
                            target_length = sampling_rate * 10  # 10 seconds
                            if signal_data.shape[1] > target_length:
                                signal_data = signal_data[:, :target_length]
                            elif signal_data.shape[1] < target_length:
                                # Pad with zeros
                                pad_width = ((0, 0), (0, target_length - signal_data.shape[1]))
                                signal_data = np.pad(signal_data, pad_width, mode='constant')
                            
                            # Normalize
                            signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
                            
                            results.append((signal_data, condition, record_name))
                    
                    except Exception as e:
                        # Skip this record if loading fails
                        continue
                
                except Exception as e:
                    # Skip this record if processing fails
                    continue
        
        except Exception as e:
            # Skip this directory if it fails
            pass
        
        return results
    
    def optimize_ptbxl_loading(self, 
                              max_records: Optional[int] = None,
                              sampling_rate: int = 100,
                              use_cache: bool = True) -> Tuple[np.ndarray, List[str], List[str], DatasetStats]:
        """
        Optimized PTB-XL dataset loading with intelligent caching
        
        Args:
            max_records: Maximum number of records to load
            sampling_rate: Target sampling rate
            use_cache: Whether to use caching
            
        Returns:
            Tuple of (signals, labels, record_ids, stats)
        """
        cache_key = f"ptbxl_optimized_{max_records}_{sampling_rate}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Try cache first
        if use_cache and cache_file.exists():
            print(f"📁 Loading PTB-XL from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}")
        
        print(f"🔄 Optimizing PTB-XL dataset loading...")
        start_time = time.time()
        stats = DatasetStats()
        
        # Load metadata
        metadata_file = self.ptbxl_path / "ptbxl_database.csv"
        if not metadata_file.exists():
            print(f"❌ PTB-XL metadata not found: {metadata_file}")
            return np.array([]), [], [], stats
        
        df = pd.read_csv(metadata_file, index_col='ecg_id')
        
        if max_records:
            df = df.head(max_records)
        
        stats.total_records = len(df)
        print(f"   • Loading {stats.total_records} PTB-XL records")
        
        # Process records
        signals_list = []
        labels_list = []
        ids_list = []
        
        # Use vectorized operations where possible
        with tqdm(df.iterrows(), total=len(df), desc="Loading PTB-XL") as pbar:
            for idx, row in pbar:
                try:
                    # Load signal
                    signal_file = self.ptbxl_path / row['filename_lr']  # Use low-res for speed
                    
                    if signal_file.exists():
                        signal, _ = wfdb.rdsamp(str(signal_file).replace('.dat', ''))
                        
                        # Process signal
                        signal = signal.T  # Transpose to (leads, samples)
                        
                        # Resample/resize if needed
                        target_length = sampling_rate * 10  # 10 seconds
                        if signal.shape[1] != target_length:
                            # Simple resampling by interpolation
                            from scipy import interpolate
                            x_old = np.linspace(0, 1, signal.shape[1])
                            x_new = np.linspace(0, 1, target_length)
                            signal_resampled = np.array([
                                interpolate.interp1d(x_old, lead, kind='linear')(x_new) 
                                for lead in signal
                            ])
                            signal = signal_resampled
                        
                        # Normalize
                        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                        
                        # Extract label
                        scp_codes = eval(row['scp_codes']) if pd.notna(row['scp_codes']) else {}
                        label = self._extract_ptbxl_label(scp_codes)
                        
                        signals_list.append(signal)
                        labels_list.append(label)
                        ids_list.append(str(idx))
                        stats.processed_records += 1
                        
                        # Update condition counts
                        if label in stats.conditions_found:
                            stats.conditions_found[label] += 1
                        else:
                            stats.conditions_found[label] = 1
                    
                except Exception as e:
                    stats.failed_records += 1
                    continue
        
        # Convert to arrays
        if signals_list:
            signals = np.array(signals_list)
            labels = labels_list
            ids = ids_list
        else:
            signals = np.array([])
            labels = []
            ids = []
        
        stats.processing_time = time.time() - start_time
        stats.memory_usage_mb = signals.nbytes / (1024 * 1024) if signals.size > 0 else 0
        
        print(f"✅ PTB-XL dataset optimized!")
        print(f"   • Processed: {stats.processed_records}/{stats.total_records}")
        print(f"   • Failed: {stats.failed_records}")
        print(f"   • Processing time: {stats.processing_time:.1f}s")
        print(f"   • Memory usage: {stats.memory_usage_mb:.1f} MB")
        
        # Cache results
        result = (signals, labels, ids, stats)
        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                print(f"💾 Results cached to: {cache_file}")
            except Exception as e:
                print(f"⚠️ Caching failed: {e}")
        
        return result
    
    def _extract_ptbxl_label(self, scp_codes: Dict) -> str:
        """Extract the primary label from PTB-XL SCP codes"""
        # Priority order for conditions
        priority_conditions = ['MI', 'AFIB', 'VTAC', 'SVTAC', 'LBBB', 'RBBB', 'AVB3', 'LVH']
        
        for condition in priority_conditions:
            if condition in scp_codes:
                return condition
        
        # Check for normal
        if 'NORM' in scp_codes:
            return 'NORM'
        
        # Default to first available condition
        if scp_codes:
            return list(scp_codes.keys())[0]
        
        return 'UNKNOWN'
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about available datasets"""
        stats = {
            'arrhythmia_dataset': {
                'path': str(self.arrhythmia_path),
                'available': self.arrhythmia_path.exists(),
                'total_directories': 0,
                'estimated_records': 0
            },
            'ptbxl_dataset': {
                'path': str(self.ptbxl_path),
                'available': self.ptbxl_path.exists(),
                'metadata_available': (self.ptbxl_path / "ptbxl_database.csv").exists(),
                'total_records': 0
            },
            'cache_status': {
                'cache_dir': str(self.cache_dir),
                'cached_files': len(list(self.cache_dir.glob("*.pkl"))),
                'cache_size_mb': sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl")) / (1024 * 1024)
            }
        }
        
        # Count arrhythmia records
        if stats['arrhythmia_dataset']['available']:
            record_dirs = self._get_arrhythmia_record_directories()
            stats['arrhythmia_dataset']['total_directories'] = len(record_dirs)
            
            # Estimate total records (rough estimate based on sample)
            if record_dirs:
                sample_dir = record_dirs[0]
                sample_count = len(list(sample_dir.glob("*.hea")))
                stats['arrhythmia_dataset']['estimated_records'] = len(record_dirs) * sample_count
        
        # Count PTB-XL records
        if stats['ptbxl_dataset']['metadata_available']:
            try:
                df = pd.read_csv(self.ptbxl_path / "ptbxl_database.csv")
                stats['ptbxl_dataset']['total_records'] = len(df)
            except Exception:
                pass
        
        return stats

# Global instance for easy import
optimized_loader = OptimizedDatasetLoader()