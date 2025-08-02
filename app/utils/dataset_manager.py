"""
High-level dataset management for ECG Classification System
Provides a unified interface for data loading and preprocessing
"""
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from app.utils.data_loader import ECGDataLoader, ArrhythmiaDataLoader, ECGArrhythmiaDataLoader
from config.settings import TARGET_CONDITIONS, PROCESSING_CONFIG


class DatasetManager:
    """Manages all dataset operations for the ECG classification system"""
    
    def __init__(self):
        self.ptbxl_loader = ECGDataLoader("ptbxl")
        self.arrhythmia_loader = ArrhythmiaDataLoader()
        self.ecg_arrhythmia_loader = ECGArrhythmiaDataLoader()
        self.data_cache = {}
        
    def load_ptbxl_complete(self, 
                          max_records: Optional[int] = None,
                          sampling_rate: int = 100,
                          use_cache: bool = True) -> Dict[str, Any]:
        """
        Complete PTB-XL loading pipeline
        
        Returns:
            Dictionary containing:
            - X: Signal data array
            - labels: Diagnostic labels
            - ids: Record IDs
            - metadata: Filtered metadata DataFrame
            - target_conditions: List of target conditions
            - stats: Loading statistics
        """
        print("LOADING PTB-XL DATASET")
        print("=" * 60)
        
        start_time = time.time()
        
        # Check cache first
        cache_key = f"ptbxl_{sampling_rate}hz_{max_records}"
        if use_cache and cache_key in self.data_cache:
            print("OK: Loading PTB-XL data from memory cache")
            return self.data_cache[cache_key]
        
        # Step 1: Load and process metadata
        Y, agg_df = self.ptbxl_loader.load_metadata()
        
        # Step 3: Filter by target conditions
        Y_filtered = self.ptbxl_loader.filter_by_conditions(Y, TARGET_CONDITIONS)
        
        # Step 4: Load signals
        X, labels, ids = self.ptbxl_loader.load_signals(
            Y_filtered,
            sampling_rate=sampling_rate,
            max_records=max_records,
            batch_size=PROCESSING_CONFIG["batch_size"],
            n_jobs=PROCESSING_CONFIG["n_jobs"],
            use_cache=use_cache
        )
        
        end_time = time.time()
        
        # Prepare results
        results = {
            'X': X,
            'labels': labels,
            'ids': ids,
            'metadata': Y_filtered,
            'target_conditions': TARGET_CONDITIONS,
            'stats': {
                'total_records': len(X),
                'shape': X.shape if len(X) > 0 else None,
                'sampling_rate': sampling_rate,
                'load_time': end_time - start_time,
                'memory_gb': X.nbytes / (1024**3) if len(X) > 0 else 0
            }
        }
        
        # Cache results
        cache_key = f"ptbxl_{sampling_rate}hz_{max_records}"
        self.data_cache[cache_key] = results
        
        self._print_summary(results)
        
        return results
    
    def load_arrhythmia_dataset(self, max_records: Optional[int] = None) -> Dict[str, Any]:
        """
        Load MIT-BIH Arrhythmia dataset
        
        Returns:
            Dictionary containing loaded records and metadata
        """
        print("LOADING MIT-BIH ARRHYTHMIA DATASET")
        print("=" * 60)
        
        start_time = time.time()
        
        # Find all records
        record_paths = self.arrhythmia_loader.find_all_records()
        
        if max_records:
            record_paths = record_paths[:max_records]
            print(f"Limited to {max_records} records")
        
        # Load records in batches
        records = self.arrhythmia_loader.load_records_batch(
            record_paths, 
            batch_size=100
        )
        
        end_time = time.time()
        
        results = {
            'records': records,
            'stats': {
                'total_records': len(records),
                'load_time': end_time - start_time
            }
        }
        
        print(f"\nOK: Loaded {len(records)} records in {end_time - start_time:.2f} seconds")
        
        return results
    
    def load_ecg_arrhythmia_complete(self, 
                                   max_records: Optional[int] = None,
                                   sampling_rate: int = 500,
                                   target_mi_records: int = 1000,
                                   use_cache: bool = True) -> Dict[str, Any]:
        """
        Complete ECG Arrhythmia dataset loading pipeline
        
        Args:
            max_records: Maximum total records to load
            sampling_rate: Target sampling rate (500Hz native)
            target_mi_records: Minimum MI records to collect
            use_cache: Use cached results if available
            
        Returns:
            Dictionary containing:
            - X: Signal data array
            - labels: Diagnostic labels
            - ids: Record IDs
            - stats: Loading statistics
        """
        print("LOADING ECG ARRHYTHMIA DATASET")
        print("=" * 60)
        
        start_time = time.time()
        
        # Check cache first
        cache_key = f"ecg_arrhythmia_{sampling_rate}hz_{max_records}_{target_mi_records}mi"
        if use_cache and cache_key in self.data_cache:
            print("OK: Loading ECG Arrhythmia data from memory cache")
            return self.data_cache[cache_key]
        
        # Load data using the specialized loader
        data = self.ecg_arrhythmia_loader.load_records_batch(
            max_records=max_records,
            sampling_rate=sampling_rate,
            use_cache=use_cache,
            target_mi_records=target_mi_records
        )
        
        end_time = time.time()
        
        # Prepare results in consistent format
        results = {
            'X': data['X'],
            'labels': data['labels'],
            'ids': data['record_ids'],
            'target_conditions': TARGET_CONDITIONS,
            'stats': {
                **data['stats'],
                'load_time': end_time - start_time,
                'memory_gb': data['X'].nbytes / (1024**3) if len(data['X']) > 0 else 0,
                'dataset_source': 'ECG_Arrhythmia'
            }
        }
        
        # Cache results
        self.data_cache[cache_key] = results
        
        self._print_summary(results)
        
        return results
    
    def prepare_combined_dataset(self, 
                               ptbxl_max_records: Optional[int] = 5000,
                               arrhythmia_max_records: Optional[int] = 2000,
                               target_mi_records: int = 1000,
                               sampling_rate: int = 100,
                               use_cache: bool = True) -> Dict[str, Any]:
        """
        Prepare a combined dataset from PTB-XL and ECG Arrhythmia for improved MI detection
        
        Args:
            ptbxl_max_records: Maximum PTB-XL records to load
            arrhythmia_max_records: Maximum ECG Arrhythmia records to load
            target_mi_records: Minimum MI records to collect from ECG Arrhythmia
            sampling_rate: Target sampling rate for consistency
            use_cache: Whether to use cached data
            
        Returns:
            Combined dataset dictionary with enhanced MI representation
        """
        print("PREPARING COMBINED DATASET FOR IMPROVED MI DETECTION")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load PTB-XL dataset
        print("\n1. Loading PTB-XL dataset...")
        ptbxl_data = self.load_ptbxl_complete(
            max_records=ptbxl_max_records,
            sampling_rate=sampling_rate,
            use_cache=use_cache
        )
        
        # Load ECG Arrhythmia dataset with focus on MI records
        print("\n2. Loading ECG Arrhythmia dataset (MI focus)...")
        arrhythmia_data = self.load_ecg_arrhythmia_complete(
            max_records=arrhythmia_max_records,
            sampling_rate=sampling_rate,
            target_mi_records=target_mi_records,
            use_cache=use_cache
        )
        
        print("\n3. Combining datasets...")
        
        # Handle case where one dataset failed to load
        if len(ptbxl_data['X']) == 0 and len(arrhythmia_data['X']) == 0:
            print("❌ Both datasets failed to load")
            return {'X': np.array([]), 'labels': [], 'ids': [], 'stats': {}}
        elif len(ptbxl_data['X']) == 0:
            print("⚠️  PTB-XL failed to load, using only ECG Arrhythmia")
            return arrhythmia_data
        elif len(arrhythmia_data['X']) == 0:
            print("⚠️  ECG Arrhythmia failed to load, using only PTB-XL")
            return ptbxl_data
        
        # Combine signals (ensure same sampling rate and lead count)
        ptbxl_signals = ptbxl_data['X']
        arrhythmia_signals = arrhythmia_data['X']
        
        # Resample ECG Arrhythmia to match PTB-XL sampling rate if needed
        if sampling_rate == 100 and arrhythmia_data['stats']['sampling_rate'] != 100:
            print("   Resampling ECG Arrhythmia signals to 100Hz...")
            from scipy import signal as scipy_signal
            new_length = int(arrhythmia_signals.shape[1] * 100 / arrhythmia_data['stats']['sampling_rate'])
            
            resampled_signals = []
            for i in range(len(arrhythmia_signals)):
                resampled = scipy_signal.resample(arrhythmia_signals[i], new_length, axis=0)
                resampled_signals.append(resampled)
            arrhythmia_signals = np.array(resampled_signals)
        
        # Ensure consistent signal length (truncate or pad to match PTB-XL)
        target_length = ptbxl_signals.shape[1]
        if arrhythmia_signals.shape[1] != target_length:
            print(f"   Adjusting signal length to {target_length} samples...")
            if arrhythmia_signals.shape[1] > target_length:
                # Truncate
                arrhythmia_signals = arrhythmia_signals[:, :target_length, :]
            else:
                # Pad with zeros
                padding_length = target_length - arrhythmia_signals.shape[1]
                padding = np.zeros((arrhythmia_signals.shape[0], padding_length, arrhythmia_signals.shape[2]))
                arrhythmia_signals = np.concatenate([arrhythmia_signals, padding], axis=1)
        
        # Combine arrays
        combined_X = np.concatenate([ptbxl_signals, arrhythmia_signals], axis=0)
        combined_labels = ptbxl_data['labels'] + arrhythmia_data['labels']
        combined_ids = ptbxl_data['ids'] + [f"ARR_{id}" for id in arrhythmia_data['ids']]
        
        end_time = time.time()
        
        # Calculate combined statistics
        from collections import Counter
        label_dist = Counter(combined_labels)
        
        combined_stats = {
            'total_records': len(combined_X),
            'ptbxl_records': len(ptbxl_data['X']),
            'arrhythmia_records': len(arrhythmia_data['X']),
            'mi_records': label_dist.get('MI', 0),
            'shape': combined_X.shape,
            'sampling_rate': sampling_rate,
            'load_time': end_time - start_time,
            'memory_gb': combined_X.nbytes / (1024**3),
            'label_distribution': dict(label_dist),
            'mi_improvement': f"{label_dist.get('MI', 0)} MI records (vs {ptbxl_data['stats'].get('label_distribution', {}).get('MI', 0)} in PTB-XL only)",
            'dataset_sources': 'PTB-XL + ECG_Arrhythmia'
        }
        
        results = {
            'X': combined_X,
            'labels': combined_labels,
            'ids': combined_ids,
            'target_conditions': TARGET_CONDITIONS,
            'stats': combined_stats
        }
        
        # Print summary
        print(f"\n✅ COMBINED DATASET SUMMARY")
        print(f"   Total records: {len(combined_X):,}")
        print(f"   PTB-XL: {len(ptbxl_data['X']):,} records")
        print(f"   ECG Arrhythmia: {len(arrhythmia_data['X']):,} records")
        print(f"   MI records: {label_dist.get('MI', 0):,} (critical improvement!)")
        print(f"   Signal shape: {combined_X.shape}")
        print(f"   Memory usage: {combined_stats['memory_gb']:.2f} GB")
        print(f"   Total load time: {end_time - start_time:.2f} seconds")
        
        return results
    
    def get_train_test_split(self, 
                           data: Dict[str, Any],
                           test_size: float = 0.2,
                           stratify: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Split data into train and test sets
        
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        from sklearn.model_selection import train_test_split
        
        X = data['X']
        labels = data['labels']
        ids = data['ids']
        
        # Convert labels to single label for stratification
        # (taking first condition if multiple)
        y_stratify = [label[0] if label else 'UNKNOWN' for label in labels]
        
        # Split data
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=y_stratify if stratify else None,
            random_state=42
        )
        
        # Create train and test dictionaries
        train_data = {
            'X': X[train_idx],
            'labels': [labels[i] for i in train_idx],
            'ids': [ids[i] for i in train_idx],
            'metadata': data['metadata'].iloc[train_idx] if 'metadata' in data else None
        }
        
        test_data = {
            'X': X[test_idx],
            'labels': [labels[i] for i in test_idx],
            'ids': [ids[i] for i in test_idx],
            'metadata': data['metadata'].iloc[test_idx] if 'metadata' in data else None
        }
        
        print(f"OK: Split data: {len(train_idx)} train, {len(test_idx)} test")
        
        return train_data, test_data
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of loaded data"""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        
        stats = results['stats']
        
        if stats['total_records'] > 0:
            print(f"Total records: {stats['total_records']:,}")
            print(f"Data shape: {stats['shape']}")
            print(f"Sampling rate: {stats['sampling_rate']} Hz")
            print(f"Memory usage: {stats['memory_gb']:.2f} GB")
            print(f"Load time: {stats['load_time']:.2f} seconds")
            print(f"Target conditions: {results['target_conditions']}")
        else:
            print("❌ No data loaded")
        
        print("=" * 60)


# Convenience functions for backwards compatibility
def run_phase1_foundation(max_records: Optional[int] = None,
                         sampling_rate: int = 100,
                         use_cache: bool = True) -> Tuple:
    """
    Backwards compatible function for Phase 1 execution
    
    Returns the same tuple format as the original Colab code
    """
    manager = DatasetManager()
    results = manager.load_ptbxl_complete(
        max_records=max_records,
        sampling_rate=sampling_rate,
        use_cache=use_cache
    )
    
    # Return in original format
    return (
        results['X'],
        results['labels'],
        results['ids'],
        results['metadata'],
        results['target_conditions'],
        None  # drive_path not needed in local version
    )


def run_combined_dataset_loading(ptbxl_max_records: Optional[int] = 1000,
                                arrhythmia_max_records: Optional[int] = 500,
                                target_mi_records: int = 200,
                                sampling_rate: int = 100,
                                use_cache: bool = True) -> Tuple:
    """
    Load combined PTB-XL + ECG Arrhythmia dataset for improved MI detection
    
    Returns:
        Tuple in original format for compatibility with existing code
    """
    manager = DatasetManager()
    results = manager.prepare_combined_dataset(
        ptbxl_max_records=ptbxl_max_records,
        arrhythmia_max_records=arrhythmia_max_records,
        target_mi_records=target_mi_records,
        sampling_rate=sampling_rate,
        use_cache=use_cache
    )
    
    # Return in original format
    return (
        results['X'],
        results['labels'],
        results['ids'],
        results.get('metadata', None),  # May not have metadata for combined dataset
        results['target_conditions'],
        results['stats']  # Return stats instead of drive_path
    )


if __name__ == "__main__":
    # Test the dataset manager
    manager = DatasetManager()
    
    # Load a small subset for testing
    results = manager.load_ptbxl_complete(max_records=100)
    
    if results['stats']['total_records'] > 0:
        print("\nSUCCESS: Dataset manager test successful!")
        print(f"   Loaded {results['stats']['total_records']} records")
    else:
        print("\n❌ Dataset manager test failed!")