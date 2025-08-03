#!/usr/bin/env python3
"""
Optimized ECG Arrhythmia Dataset Loader
Efficiently processes 45,152 WFDB records with progress monitoring
"""

import os
import time
import pickle
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

class OptimizedArrhythmiaLoader:
    """Optimized loader for ECG Arrhythmia dataset with caching and progress monitoring"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or 'data/raw/ecg-arrhythmia-dataset/WFDBRecords')
        self.cache_dir = Path('data/cache')
        self.cache_dir.mkdir(exist_ok=True)
        
    def scan_available_records(self) -> List[str]:
        """Quickly scan available WFDB records"""
        print("Scanning ECG Arrhythmia dataset structure...")
        
        cache_file = self.cache_dir / 'arrhythmia_record_list.pkl'
        
        # Check cache first
        if cache_file.exists():
            print("   Loading from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Scan filesystem
        record_paths = []
        
        if self.base_path.exists():
            # Look for .hea files in all subdirectories
            for hea_file in self.base_path.rglob('*.hea'):
                # Convert to record path (without extension)
                record_path = str(hea_file).replace('.hea', '')
                record_paths.append(record_path)
        
        print(f"   Found {len(record_paths)} WFDB records")
        
        # Cache the results
        with open(cache_file, 'wb') as f:
            pickle.dump(record_paths, f)
        
        return record_paths
    
    def load_sample_batch(self, max_records: int = 100, sampling_rate: int = 100) -> Dict[str, Any]:
        """Load a small batch for testing and immediate use"""
        print(f"Loading sample batch ({max_records} records)...")
        
        record_paths = self.scan_available_records()
        
        if not record_paths:
            raise ValueError("No WFDB records found")
        
        # Take first max_records for sample
        sample_paths = record_paths[:max_records]
        
        signals = []
        labels = []
        metadata = []
        
        print("   Processing sample records...")
        for i, record_path in enumerate(tqdm(sample_paths, desc="Loading")):
            try:
                # Read WFDB record
                record = wfdb.rdrecord(record_path)
                
                # Extract signal data
                signal = record.p_signal
                
                # Resample if needed
                if record.fs != sampling_rate:
                    # Simple resampling - take every nth sample
                    resample_factor = record.fs // sampling_rate
                    signal = signal[::resample_factor]
                
                # Ensure 12-lead format
                if signal.shape[1] >= 12:
                    signal = signal[:, :12]  # Take first 12 leads
                else:
                    # Pad if fewer than 12 leads
                    padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
                    signal = np.hstack([signal, padding])
                
                # Fixed length (10 seconds at target sampling rate)
                target_length = sampling_rate * 10
                if len(signal) > target_length:
                    signal = signal[:target_length]
                elif len(signal) < target_length:
                    # Pad with zeros
                    padding = np.zeros((target_length - len(signal), 12))
                    signal = np.vstack([signal, padding])
                
                signals.append(signal)
                
                # Extract diagnostic codes and metadata
                dx_codes = []
                patient_info = {}
                
                for comment in record.comments:
                    if comment.startswith('Dx:'):
                        dx_codes = comment.replace('Dx: ', '').split(',')
                    elif comment.startswith('Age:'):
                        patient_info['age'] = comment.replace('Age: ', '')
                    elif comment.startswith('Sex:'):
                        patient_info['sex'] = comment.replace('Sex: ', '')
                
                # Map diagnostic codes to conditions using comprehensive mapper
                from app.utils.comprehensive_mapper import comprehensive_mapper
                mapped_conditions = comprehensive_mapper.map_arrhythmia_snomed_codes(dx_codes)
                primary_condition = comprehensive_mapper.get_primary_condition(mapped_conditions)
                
                labels.append(primary_condition)
                metadata.append({
                    'record_id': Path(record_path).name,
                    'dx_codes': dx_codes,
                    'mapped_conditions': mapped_conditions,
                    'primary_condition': primary_condition,
                    'original_fs': record.fs,
                    'patient_info': patient_info
                })
                
            except Exception as e:
                print(f"   Warning: Error processing {record_path}: {e}")
                continue
        
        if not signals:
            raise ValueError("No valid records loaded")
        
        # Convert to numpy arrays
        X = np.array(signals)
        y = np.array(labels)
        
        # Create results dictionary
        results = {
            'X': X,
            'labels': y,
            'metadata': metadata,
            'stats': {
                'total_records': len(X),
                'shape': X.shape,
                'sampling_rate': sampling_rate,
                'unique_conditions': list(set(y)),
                'condition_counts': pd.Series(y).value_counts().to_dict()
            }
        }
        
        print(f"Loaded {len(X)} records successfully")
        print(f"   Shape: {X.shape}")
        print(f"   Conditions found: {list(set(y))}")
        
        return results
    
    def load_full_dataset(self, max_records: int = None, batch_size: int = 1000) -> Dict[str, Any]:
        """Load full dataset with progress monitoring and batching"""
        print(f"🔄 Loading full ECG Arrhythmia dataset...")
        
        cache_file = self.cache_dir / f'arrhythmia_full_{max_records or "all"}.pkl'
        
        # Check cache
        if cache_file.exists():
            print("   Loading from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        record_paths = self.scan_available_records()
        
        if max_records:
            record_paths = record_paths[:max_records]
        
        print(f"   Processing {len(record_paths)} records in batches of {batch_size}...")
        
        all_signals = []
        all_labels = []
        all_metadata = []
        
        # Process in batches to manage memory
        for batch_start in range(0, len(record_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(record_paths))
            batch_paths = record_paths[batch_start:batch_end]
            
            print(f"   Processing batch {batch_start//batch_size + 1}/{(len(record_paths)-1)//batch_size + 1}")
            
            try:
                batch_results = self._process_batch(batch_paths)
                all_signals.extend(batch_results['signals'])
                all_labels.extend(batch_results['labels'])
                all_metadata.extend(batch_results['metadata'])
            except Exception as e:
                print(f"   Warning: Error in batch {batch_start}-{batch_end}: {e}")
                continue
        
        if not all_signals:
            raise ValueError("No valid records loaded from full dataset")
        
        # Combine results
        X = np.array(all_signals)
        y = np.array(all_labels)
        
        results = {
            'X': X,
            'labels': y,
            'metadata': all_metadata,
            'stats': {
                'total_records': len(X),
                'shape': X.shape,
                'unique_conditions': list(set(y)),
                'condition_counts': pd.Series(y).value_counts().to_dict()
            }
        }
        
        # Cache results
        print("   Caching results for future use...")
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"✅ Full dataset loaded: {X.shape}")
        return results
    
    def _process_batch(self, record_paths: List[str], sampling_rate: int = 100) -> Dict[str, List]:
        """Process a batch of records"""
        signals = []
        labels = []
        metadata = []
        
        for record_path in tqdm(record_paths, desc="Batch progress"):
            try:
                # Read and process record (same as load_sample_batch)
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal
                
                # Resample and format
                if record.fs != sampling_rate:
                    resample_factor = record.fs // sampling_rate
                    signal = signal[::resample_factor]
                
                if signal.shape[1] >= 12:
                    signal = signal[:, :12]
                else:
                    padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
                    signal = np.hstack([signal, padding])
                
                target_length = sampling_rate * 10
                if len(signal) > target_length:
                    signal = signal[:target_length]
                elif len(signal) < target_length:
                    padding = np.zeros((target_length - len(signal), 12))
                    signal = np.vstack([signal, padding])
                
                signals.append(signal)
                
                # Extract metadata and map conditions
                dx_codes = []
                for comment in record.comments:
                    if comment.startswith('Dx:'):
                        dx_codes = comment.replace('Dx: ', '').split(',')
                        break
                
                from app.utils.comprehensive_mapper import comprehensive_mapper
                mapped_conditions = comprehensive_mapper.map_arrhythmia_snomed_codes(dx_codes)
                primary_condition = comprehensive_mapper.get_primary_condition(mapped_conditions)
                
                labels.append(primary_condition)
                metadata.append({
                    'record_id': Path(record_path).name,
                    'dx_codes': dx_codes,
                    'mapped_conditions': mapped_conditions,
                    'primary_condition': primary_condition
                })
                
            except Exception as e:
                continue  # Skip problematic records
        
        return {'signals': signals, 'labels': labels, 'metadata': metadata}

def quick_test():
    """Quick test of optimized loader"""
    print("🧪 TESTING OPTIMIZED ECG ARRHYTHMIA LOADER")
    print("=" * 60)
    
    loader = OptimizedArrhythmiaLoader()
    
    try:
        # Test with small sample first
        results = loader.load_sample_batch(max_records=10)
        
        print(f"✅ SUCCESS!")
        print(f"   Loaded: {results['X'].shape}")
        print(f"   Conditions: {results['stats']['unique_conditions']}")
        print(f"   Condition counts: {results['stats']['condition_counts']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    quick_test()