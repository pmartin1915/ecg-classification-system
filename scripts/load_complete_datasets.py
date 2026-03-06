#!/usr/bin/env python3
"""
Complete Dataset Loading Script
High-performance loading of all available ECG datasets with progress tracking and optimization
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.utils.optimized_dataset_loader import OptimizedDatasetLoader, DatasetStats
import numpy as np
import pandas as pd

def print_banner():
    """Print professional banner for dataset loading"""
    print("=" * 80)
    print("🚀 COMPREHENSIVE ECG DATASET LOADING")
    print("   Advanced Cardiac Analysis - Professional Edition")
    print("=" * 80)
    print()

def print_progress_callback(progress: float):
    """Progress callback for dataset loading"""
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: |{bar}| {progress:.1%}', end='', flush=True)

def display_dataset_statistics(loader: OptimizedDatasetLoader):
    """Display comprehensive dataset statistics"""
    print("📊 DATASET AVAILABILITY ASSESSMENT")
    print("-" * 60)
    
    stats = loader.get_dataset_statistics()
    
    # Arrhythmia dataset
    arrhythmia = stats['arrhythmia_dataset']
    status = "✅ AVAILABLE" if arrhythmia['available'] else "❌ NOT FOUND"
    print(f"ECG Arrhythmia Dataset: {status}")
    if arrhythmia['available']:
        print(f"   • Directory count: {arrhythmia['total_directories']}")
        print(f"   • Estimated records: ~{arrhythmia['estimated_records']:,}")
        print(f"   • Path: {arrhythmia['path']}")
    
    print()
    
    # PTB-XL dataset
    ptbxl = stats['ptbxl_dataset']
    status = "✅ AVAILABLE" if ptbxl['available'] else "❌ NOT FOUND"
    print(f"PTB-XL Dataset: {status}")
    if ptbxl['available']:
        print(f"   • Total records: {ptbxl['total_records']:,}")
        print(f"   • Metadata available: {'✅' if ptbxl['metadata_available'] else '❌'}")
        print(f"   • Path: {ptbxl['path']}")
    
    print()
    
    # Cache status
    cache = stats['cache_status']
    print(f"Cache System:")
    print(f"   • Cached files: {cache['cached_files']}")
    print(f"   • Cache size: {cache['cache_size_mb']:.1f} MB")
    print(f"   • Cache directory: {cache['cache_dir']}")
    
    print()

def load_arrhythmia_dataset(loader: OptimizedDatasetLoader, 
                          max_records: int = None,
                          sampling_rate: int = 100) -> tuple:
    """Load the complete arrhythmia dataset"""
    print("🔄 LOADING ECG ARRHYTHMIA DATASET")
    print("-" * 60)
    print(f"Target records: {max_records or 'ALL AVAILABLE'}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Expected processing time: {(max_records or 45000) / 1000 * 2:.1f} minutes")
    print()
    
    start_time = time.time()
    
    signals, labels, ids, stats = loader.load_complete_arrhythmia_dataset(
        max_records=max_records,
        sampling_rate=sampling_rate,
        progress_callback=print_progress_callback
    )
    
    print()  # New line after progress bar
    
    # Display results
    print("✅ ARRHYTHMIA DATASET LOADED")
    print(f"   • Total processed: {stats.processed_records:,}")
    print(f"   • Failed records: {stats.failed_records:,}")
    print(f"   • Success rate: {(stats.processed_records / stats.total_records * 100):.1f}%")
    print(f"   • Processing time: {stats.processing_time:.1f} seconds")
    print(f"   • Memory usage: {stats.memory_usage_mb:.1f} MB")
    print(f"   • Signal shape: {signals.shape if signals.size > 0 else 'Empty'}")
    
    # Display condition distribution
    if stats.conditions_found:
        print(f"   • Conditions found: {len(stats.conditions_found)}")
        top_conditions = sorted(stats.conditions_found.items(), key=lambda x: x[1], reverse=True)[:10]
        for condition, count in top_conditions:
            print(f"     - {condition}: {count:,} records")
    
    print()
    return signals, labels, ids, stats

def load_ptbxl_dataset(loader: OptimizedDatasetLoader,
                      max_records: int = None,
                      sampling_rate: int = 100) -> tuple:
    """Load the optimized PTB-XL dataset"""
    print("🔄 LOADING OPTIMIZED PTB-XL DATASET")
    print("-" * 60)
    print(f"Target records: {max_records or 'ALL AVAILABLE'}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print()
    
    signals, labels, ids, stats = loader.optimize_ptbxl_loading(
        max_records=max_records,
        sampling_rate=sampling_rate,
        use_cache=True
    )
    
    # Display results
    print("✅ PTB-XL DATASET OPTIMIZED")
    print(f"   • Total processed: {stats.processed_records:,}")
    print(f"   • Failed records: {stats.failed_records:,}")
    print(f"   • Success rate: {(stats.processed_records / stats.total_records * 100):.1f}%")
    print(f"   • Processing time: {stats.processing_time:.1f} seconds")
    print(f"   • Memory usage: {stats.memory_usage_mb:.1f} MB")
    print(f"   • Signal shape: {signals.shape if signals.size > 0 else 'Empty'}")
    
    # Display condition distribution
    if stats.conditions_found:
        print(f"   • Conditions found: {len(stats.conditions_found)}")
        top_conditions = sorted(stats.conditions_found.items(), key=lambda x: x[1], reverse=True)[:10]
        for condition, count in top_conditions:
            print(f"     - {condition}: {count:,} records")
    
    print()
    return signals, labels, ids, stats

def combine_datasets(arrhythmia_data: tuple, ptbxl_data: tuple) -> tuple:
    """Combine both datasets intelligently"""
    print("🔗 COMBINING DATASETS")
    print("-" * 60)
    
    arr_signals, arr_labels, arr_ids, arr_stats = arrhythmia_data
    ptb_signals, ptb_labels, ptb_ids, ptb_stats = ptbxl_data
    
    # Combine signals
    if arr_signals.size > 0 and ptb_signals.size > 0:
        # Ensure compatible shapes
        min_leads = min(arr_signals.shape[1], ptb_signals.shape[1])
        min_samples = min(arr_signals.shape[2], ptb_signals.shape[2])
        
        arr_signals_resized = arr_signals[:, :min_leads, :min_samples]
        ptb_signals_resized = ptb_signals[:, :min_leads, :min_samples]
        
        combined_signals = np.concatenate([arr_signals_resized, ptb_signals_resized], axis=0)
        combined_labels = arr_labels + ptb_labels
        combined_ids = arr_ids + [f"ptb_{id}" for id in ptb_ids]
        
    elif arr_signals.size > 0:
        combined_signals = arr_signals
        combined_labels = arr_labels
        combined_ids = arr_ids
    elif ptb_signals.size > 0:
        combined_signals = ptb_signals
        combined_labels = ptb_labels
        combined_ids = ptb_ids
    else:
        combined_signals = np.array([])
        combined_labels = []
        combined_ids = []
    
    # Combined statistics
    total_records = arr_stats.processed_records + ptb_stats.processed_records
    total_memory = arr_stats.memory_usage_mb + ptb_stats.memory_usage_mb
    
    print(f"✅ DATASETS COMBINED")
    print(f"   • Total records: {total_records:,}")
    print(f"   • Arrhythmia contribution: {arr_stats.processed_records:,}")
    print(f"   • PTB-XL contribution: {ptb_stats.processed_records:,}")
    print(f"   • Combined shape: {combined_signals.shape if combined_signals.size > 0 else 'Empty'}")
    print(f"   • Total memory usage: {total_memory:.1f} MB")
    
    # Combined condition distribution
    combined_conditions = {}
    for condition, count in arr_stats.conditions_found.items():
        combined_conditions[condition] = combined_conditions.get(condition, 0) + count
    for condition, count in ptb_stats.conditions_found.items():
        combined_conditions[condition] = combined_conditions.get(condition, 0) + count
    
    if combined_conditions:
        print(f"   • Unique conditions: {len(combined_conditions)}")
        top_conditions = sorted(combined_conditions.items(), key=lambda x: x[1], reverse=True)[:15]
        for condition, count in top_conditions:
            print(f"     - {condition}: {count:,} records")
    
    print()
    return combined_signals, combined_labels, combined_ids, combined_conditions

def main():
    """Main function for complete dataset loading"""
    parser = argparse.ArgumentParser(description='Load complete ECG datasets with optimization')
    parser.add_argument('--arrhythmia-records', type=int, default=None,
                       help='Maximum arrhythmia records to load (default: all)')
    parser.add_argument('--ptbxl-records', type=int, default=None,
                       help='Maximum PTB-XL records to load (default: all)')
    parser.add_argument('--sampling-rate', type=int, default=100,
                       help='Target sampling rate in Hz (default: 100)')
    parser.add_argument('--no-multiprocessing', action='store_true',
                       help='Disable multiprocessing')
    parser.add_argument('--stats-only', action='store_true',
                       help='Show statistics only, do not load data')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Initialize optimized loader
    loader = OptimizedDatasetLoader(
        use_multiprocessing=not args.no_multiprocessing,
        max_workers=None  # Auto-detect
    )
    
    # Display statistics
    display_dataset_statistics(loader)
    
    if args.stats_only:
        print("📊 Statistics display complete. Use without --stats-only to load data.")
        return
    
    # Load datasets
    arrhythmia_data = None
    ptbxl_data = None
    
    # Load arrhythmia dataset
    try:
        arrhythmia_data = load_arrhythmia_dataset(
            loader, 
            max_records=args.arrhythmia_records,
            sampling_rate=args.sampling_rate
        )
    except Exception as e:
        print(f"⚠️ Arrhythmia dataset loading failed: {e}")
        arrhythmia_data = (np.array([]), [], [], DatasetStats())
    
    # Load PTB-XL dataset
    try:
        ptbxl_data = load_ptbxl_dataset(
            loader,
            max_records=args.ptbxl_records,
            sampling_rate=args.sampling_rate
        )
    except Exception as e:
        print(f"⚠️ PTB-XL dataset loading failed: {e}")
        ptbxl_data = (np.array([]), [], [], DatasetStats())
    
    # Combine datasets
    if arrhythmia_data and ptbxl_data:
        combined_data = combine_datasets(arrhythmia_data, ptbxl_data)
    
    print("🎉 COMPLETE DATASET LOADING FINISHED")
    print("=" * 80)
    print("Your system now has access to the complete cardiac database!")
    print("Use your Professional Launcher to explore all features.")
    print("=" * 80)

if __name__ == "__main__":
    main()