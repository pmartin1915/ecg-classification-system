"""
High-level dataset management for ECG Classification System
Provides a unified interface for data loading and preprocessing
"""
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from app.utils.data_loader import ECGDataLoader, ArrhythmiaDataLoader
from config.settings import TARGET_CONDITIONS, PROCESSING_CONFIG


class DatasetManager:
    """Manages all dataset operations for the ECG classification system"""
    
    def __init__(self):
        self.ptbxl_loader = ECGDataLoader("ptbxl")
        self.arrhythmia_loader = ArrhythmiaDataLoader()
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
        print("🚀 LOADING PTB-XL DATASET")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Download metadata
        self.ptbxl_loader.download_metadata()
        
        # Step 2: Load and process metadata
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
        print("🚀 LOADING MIT-BIH ARRHYTHMIA DATASET")
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
        
        print(f"\n✅ Loaded {len(records)} records in {end_time - start_time:.2f} seconds")
        
        return results
    
    def prepare_combined_dataset(self, 
                               ptbxl_ratio: float = 0.8,
                               use_cache: bool = True) -> Dict[str, Any]:
        """
        Prepare a combined dataset from PTB-XL and MIT-BIH
        
        Args:
            ptbxl_ratio: Ratio of PTB-XL samples in combined dataset
            use_cache: Whether to use cached data
            
        Returns:
            Combined dataset dictionary
        """
        print("🚀 PREPARING COMBINED DATASET")
        print("=" * 60)
        
        # Load both datasets
        ptbxl_data = self.load_ptbxl_complete(use_cache=use_cache)
        # arrhythmia_data = self.load_arrhythmia_dataset()  # Uncomment when ready
        
        # For now, just return PTB-XL data
        # TODO: Implement proper dataset combination
        
        return ptbxl_data
    
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
        
        print(f"✅ Split data: {len(train_idx)} train, {len(test_idx)} test")
        
        return train_data, test_data
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of loaded data"""
        print("\n" + "=" * 60)
        print("📊 DATASET SUMMARY")
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


if __name__ == "__main__":
    # Test the dataset manager
    manager = DatasetManager()
    
    # Load a small subset for testing
    results = manager.load_ptbxl_complete(max_records=100)
    
    if results['stats']['total_records'] > 0:
        print("\n✅ Dataset manager test successful!")
        print(f"   Loaded {results['stats']['total_records']} records")
    else:
        print("\n❌ Dataset manager test failed!")