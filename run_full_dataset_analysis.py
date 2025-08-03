#!/usr/bin/env python3
"""
Full Dataset Analysis - Load ALL available ECG records for comprehensive cardiac analysis
"""

from app.utils.dataset_manager import run_combined_dataset_loading
import time
import pandas as pd

def run_comprehensive_analysis():
    """Load and analyze the complete dataset with all available records"""
    
    print("🫀 COMPREHENSIVE ECG CARDIAC ANALYSIS")
    print("="*80)
    
    print("\n🔍 Loading FULL datasets...")
    print("- PTB-XL: ~21,388 records")  
    print("- ECG Arrhythmia: ~45,152 records")
    print("- Total: ~66,540 clinical records")
    
    start_time = time.time()
    
    # Load maximum available records
    print("\n⚡ Starting comprehensive data loading...")
    X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
        ptbxl_max_records=21388,      # ALL PTB-XL records
        arrhythmia_max_records=45152, # ALL Arrhythmia records  
        target_mi_records=5000,       # Ensure strong MI representation
        sampling_rate=100,            # Standard clinical sampling
        use_cache=True               # Cache for future runs
    )
    
    load_time = time.time() - start_time
    
    print(f"\n✅ SUCCESS! Loaded {X.shape[0]} total records")
    print(f"📊 Data shape: {X.shape}")
    print(f"⏱️  Load time: {load_time:.1f} seconds")
    print(f"💾 Memory usage: {X.nbytes / 1e9:.2f} GB")
    
    # Analyze condition distribution
    print(f"\n🎯 Target conditions: {target_conditions}")
    condition_counts = pd.Series(labels).value_counts()
    print("\n📈 Condition Distribution:")
    for condition, count in condition_counts.items():
        percentage = (count / len(labels)) * 100
        print(f"   {condition}: {count:,} records ({percentage:.1f}%)")
    
    # Dataset statistics
    print(f"\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return X, labels, ids, metadata, target_conditions, stats

if __name__ == "__main__":
    X, labels, ids, metadata, target_conditions, stats = run_comprehensive_analysis()
    print(f"\n🎉 Complete dataset ready for advanced cardiac analysis!")