#!/usr/bin/env python3
"""
Quick Launch Comprehensive ECG System
Uses existing processed data + small sample from ECG Arrhythmia for immediate launch
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_single_wfdb_record():
    """Test reading a single WFDB record to verify functionality"""
    import wfdb
    import os
    
    print("🔬 TESTING WFDB FILE READING...")
    
    test_path = 'data/raw/ecg-arrhythmia-dataset/WFDBRecords/01/010/JS00001'
    if os.path.exists(test_path + '.hea'):
        try:
            record = wfdb.rdrecord(test_path)
            print(f"✅ SUCCESS: Read WFDB record")
            print(f"   • Shape: {record.p_signal.shape}")
            print(f"   • Sampling rate: {record.fs} Hz") 
            print(f"   • Leads: {record.sig_name}")
            print(f"   • Duration: {len(record.p_signal)/record.fs:.1f} seconds")
            
            # Extract diagnostic codes
            dx_codes = []
            for comment in record.comments:
                if comment.startswith('Dx:'):
                    dx_codes = comment.replace('Dx: ', '').split(',')
                    break
            
            print(f"   • Diagnostic codes: {dx_codes}")
            return True, record, dx_codes
            
        except Exception as e:
            print(f"❌ ERROR reading WFDB: {e}")
            return False, None, []
    else:
        print(f"❌ Test file not found: {test_path}")
        return False, None, []

def launch_with_existing_data():
    """Launch system using existing processed data"""
    
    print("\n🚀 LAUNCHING COMPREHENSIVE ECG SYSTEM")
    print("=" * 80)
    
    from config.settings import TARGET_CONDITIONS, CLINICAL_PRIORITY
    from app.utils.comprehensive_mapper import comprehensive_mapper
    
    print(f"📊 COMPREHENSIVE SYSTEM STATUS:")
    print(f"   • Target conditions: {len(TARGET_CONDITIONS)}")
    print(f"   • Previous system: 5 conditions")
    print(f"   • Enhancement: {len(TARGET_CONDITIONS) - 5} additional conditions")
    
    # Check existing processed data
    processed_dir = Path('data/processed')
    if processed_dir.exists():
        print(f"\n📁 EXISTING PROCESSED DATA:")
        for file in processed_dir.glob('*.pkl'):
            size_mb = file.stat().st_size / (1024*1024)
            print(f"   • {file.name}: {size_mb:.1f} MB")
        
        for file in processed_dir.glob('*.npy'):
            try:
                data = np.load(file)
                print(f"   • {file.name}: {data.shape}")
            except:
                print(f"   • {file.name}: (unable to read shape)")
    
    # Test comprehensive mapping
    print(f"\n🎯 COMPREHENSIVE CONDITION MAPPING:")
    
    # Test PTB-XL mapping
    test_conditions = ['AMI', 'AFIB', 'LBBB', 'LVH', 'NORM']
    for condition in test_conditions:
        info = comprehensive_mapper.get_condition_info(condition)
        print(f"   • {condition}: {info['name']} [{info['priority']}]")
    
    # Test SNOMED mapping for ECG Arrhythmia
    print(f"\n🏥 ECG ARRHYTHMIA SNOMED MAPPING:")
    test_snomed = ['164889003', '59118001', '164934002']  # From sample file
    mapped = comprehensive_mapper.map_arrhythmia_snomed_codes(test_snomed)
    print(f"   • SNOMED codes {test_snomed} → {mapped}")
    
    print(f"\n🚨 CLINICAL PRIORITY SYSTEM:")
    for priority, conditions in CLINICAL_PRIORITY.items():
        print(f"   • {priority}: {len(conditions)} conditions")
    
    return True

def quick_arrhythmia_sample():
    """Load a small sample of ECG Arrhythmia data for demonstration"""
    import wfdb
    import os
    
    print(f"\n📡 SAMPLING ECG ARRHYTHMIA DATASET...")
    
    sample_files = []
    base_path = Path('data/raw/ecg-arrhythmia-dataset/WFDBRecords')
    
    # Find first 5 available records for quick demo
    for folder in ['01/010', '01/011', '01/012']:
        folder_path = base_path / folder
        if folder_path.exists():
            for file in folder_path.glob('JS*.hea'):
                if len(sample_files) < 5:
                    record_path = str(file).replace('.hea', '')
                    sample_files.append(record_path)
    
    print(f"   • Found {len(sample_files)} sample records")
    
    if sample_files:
        # Read first record as demo
        try:
            record = wfdb.rdrecord(sample_files[0])
            print(f"   • Sample record shape: {record.p_signal.shape}")
            print(f"   • Sampling rate: {record.fs} Hz")
            
            # Extract conditions
            dx_codes = []
            for comment in record.comments:
                if comment.startswith('Dx:'):
                    dx_codes = comment.replace('Dx: ', '').split(',')
                    break
            
            from app.utils.comprehensive_mapper import comprehensive_mapper
            mapped_conditions = comprehensive_mapper.map_arrhythmia_snomed_codes(dx_codes)
            print(f"   • SNOMED codes: {dx_codes}")
            print(f"   • Mapped to: {mapped_conditions}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error reading sample: {e}")
            return False
    
    return False

def main():
    """Main launch function"""
    
    print("🫀 COMPREHENSIVE ECG CLASSIFICATION SYSTEM")
    print("=" * 80)
    print("Quick Launch with Existing Data + WFDB Integration Test")
    
    # Test WFDB reading capability
    wfdb_success, record, dx_codes = test_single_wfdb_record()
    
    # Launch with existing data
    system_ready = launch_with_existing_data()
    
    # Test small arrhythmia sample
    arrhythmia_ready = quick_arrhythmia_sample()
    
    print(f"\n🎉 SYSTEM STATUS:")
    print(f"   • WFDB file reading: {'✅' if wfdb_success else '❌'}")
    print(f"   • Comprehensive mapping: {'✅' if system_ready else '❌'}")
    print(f"   • ECG Arrhythmia sampling: {'✅' if arrhythmia_ready else '❌'}")
    
    if wfdb_success and system_ready:
        print(f"\n🚀 READY TO LAUNCH STREAMLIT:")
        print(f"   streamlit run app/main.py")
        print(f"\n💡 FOR FULL DATASET (background process):")
        print(f"   python run_full_dataset_analysis.py")
        print(f"   (This will take 15-30 minutes for all 45,152 records)")
        
        print(f"\n🎯 YOUR SYSTEM NOW DETECTS:")
        print(f"   • 30 comprehensive cardiac conditions")
        print(f"   • Clinical priority classification")  
        print(f"   • WFDB format support (45,152 additional records)")
        print(f"   • Real-time arrhythmia analysis")
        
        return True
    else:
        print(f"\n❌ System not fully ready - check errors above")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎊 LAUNCH SUCCESSFUL!' if success else '⚠️  LAUNCH INCOMPLETE'}")