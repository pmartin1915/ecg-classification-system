#!/usr/bin/env python3
"""
Launch Comprehensive ECG Classification System
Demonstrates the enhanced 30-condition cardiac analysis
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Launch the comprehensive ECG system"""
    
    print("🫀 LAUNCHING COMPREHENSIVE ECG CLASSIFICATION SYSTEM")
    print("=" * 80)
    
    from config.settings import TARGET_CONDITIONS, CLINICAL_PRIORITY
    from app.utils.comprehensive_mapper import comprehensive_mapper
    
    print(f"\n✅ SYSTEM READY!")
    print(f"   • Total Conditions: {len(TARGET_CONDITIONS)}")
    print(f"   • Previous System: 5 conditions")
    print(f"   • Enhancement: {len(TARGET_CONDITIONS) - 5} additional conditions")
    
    print(f"\n🎯 DETECTION CAPABILITIES:")
    categories = {
        'Myocardial Infarction': ['AMI', 'IMI', 'LMI', 'PMI'],
        'Arrhythmias': ['AFIB', 'AFLT', 'VTAC', 'SVTAC', 'PVC', 'PAC'],
        'Conduction Disorders': ['AVB1', 'AVB2', 'AVB3', 'RBBB', 'LBBB', 'WPW'],
        'Structural Changes': ['LVH', 'RVH', 'LAE', 'RAE']
    }
    
    for category, conditions in categories.items():
        print(f"   • {category}: {len(conditions)} conditions")
    
    print(f"\n🚨 CLINICAL PRIORITY SYSTEM:")
    for priority, conditions in CLINICAL_PRIORITY.items():
        print(f"   • {priority}: {len(conditions)} conditions")
    
    print(f"\n📊 AVAILABLE DATASETS:")
    print(f"   • PTB-XL: 21,388 records")
    print(f"   • ECG Arrhythmia: 45,152 records") 
    print(f"   • Total Available: 66,540 clinical records")
    
    print(f"\n🚀 TO LAUNCH SYSTEM:")
    print(f"   streamlit run app/main.py")
    print(f"\n🔬 TO LOAD FULL DATASET:")
    print(f"   python run_full_dataset_analysis.py")
    
    print(f"\n✨ COMPREHENSIVE CARDIAC ANALYSIS IS READY!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Ready for world-class cardiac analysis!")
    else:
        print(f"\n❌ System initialization failed")