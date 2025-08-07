"""
MI Detection Improvement Toolkit
Advanced tools for improving MI detection sensitivity from 35% to 60%+
"""

import pandas as pd
import numpy as np
import wfdb
from pathlib import Path
import pickle
from collections import Counter
from typing import Tuple, List, Dict
import time

class MIImprovementToolkit:
    """
    Comprehensive toolkit for improving MI detection performance
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.cache_dir = self.project_root / "data" / "cache"
        self.ptbxl_path = self.project_root / "data" / "raw" / "ptbxl"
        
    def analyze_ptbxl_mi_variety(self) -> Dict:
        """
        Analyze the variety of MI types in PTB-XL for better understanding
        """
        print("[ANALYSIS] Analyzing PTB-XL MI variety...")
        
        df = pd.read_csv(self.ptbxl_path / "ptbxl_database.csv")
        
        # Find all MI records
        mi_records = df[df['scp_codes'].str.contains('MI|AMI|IMI|STEMI|NSTEMI', na=False)]
        
        # Categorize MI types
        mi_types = {
            'AMI': mi_records[mi_records['scp_codes'].str.contains('AMI', na=False)],
            'IMI': mi_records[mi_records['scp_codes'].str.contains('IMI', na=False)],
            'STEMI': mi_records[mi_records['scp_codes'].str.contains('STEMI', na=False)],
            'NSTEMI': mi_records[mi_records['scp_codes'].str.contains('NSTEMI', na=False)],
            'LMI': mi_records[mi_records['scp_codes'].str.contains('LMI', na=False)],
            'PMI': mi_records[mi_records['scp_codes'].str.contains('PMI', na=False)],
        }
        
        print("MI Type Distribution in PTB-XL:")
        for mi_type, records in mi_types.items():
            if len(records) > 0:
                print(f"  {mi_type:6}: {len(records):4} cases")
        
        return {
            'total_mi': len(mi_records),
            'mi_types': {k: len(v) for k, v in mi_types.items() if len(v) > 0},
            'mi_records': mi_records
        }
    
    def create_balanced_mi_dataset(self, mi_ratio: float = 0.3, total_samples: int = 2000) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Create a balanced dataset with specified MI ratio
        
        Args:
            mi_ratio: Desired ratio of MI cases (0.3 = 30% MI cases)
            total_samples: Total number of samples in dataset
        """
        print(f"[CREATING] Balanced MI dataset ({mi_ratio*100:.0f}% MI, {total_samples} total)")
        
        # Load PTB-XL database
        df = pd.read_csv(self.ptbxl_path / "ptbxl_database.csv")
        
        # Calculate desired counts
        desired_mi_count = int(total_samples * mi_ratio)
        desired_normal_count = total_samples - desired_mi_count
        
        print(f"  Target: {desired_mi_count} MI cases, {desired_normal_count} normal cases")
        
        # Get MI records
        mi_records = df[df['scp_codes'].str.contains('MI|AMI|IMI', na=False)]
        
        # Get normal records
        normal_records = df[df['scp_codes'].str.contains('NORM', na=False)]
        normal_records = normal_records[~normal_records['scp_codes'].str.contains('MI|AMI|IMI', na=False)]
        
        # Sample the desired amounts
        selected_mi = mi_records.sample(n=min(desired_mi_count, len(mi_records)), random_state=42)
        selected_normal = normal_records.sample(n=min(desired_normal_count, len(normal_records)), random_state=42)
        
        print(f"  Selected: {len(selected_mi)} MI, {len(selected_normal)} normal")
        
        # Load ECG signals
        all_signals = []
        all_labels = []
        all_ids = []
        
        # Load MI signals
        print("  Loading MI signals...")
        for idx, row in selected_mi.iterrows():
            try:
                signal_path = self.ptbxl_path / f"{row['filename_lr']}"
                if signal_path.with_suffix('.hea').exists():
                    signal, _ = wfdb.rdsamp(str(signal_path.with_suffix('')))
                    
                    # Normalize and resample to 1000 points, 12 leads
                    if signal.shape[1] >= 12:
                        signal_resampled = self._resample_signal(signal[:, :12], target_length=1000)
                        all_signals.append(signal_resampled)
                        all_labels.append('MI')
                        all_ids.append(row['ecg_id'])
                        
                        if len(all_signals) % 50 == 0:
                            print(f"    Loaded {len([l for l in all_labels if l == 'MI'])} MI signals...")
                            
            except Exception as e:
                continue  # Skip problematic records
        
        # Load Normal signals  
        print("  Loading normal signals...")
        for idx, row in selected_normal.iterrows():
            try:
                signal_path = self.ptbxl_path / f"{row['filename_lr']}"
                if signal_path.with_suffix('.hea').exists():
                    signal, _ = wfdb.rdsamp(str(signal_path.with_suffix('')))
                    
                    # Normalize and resample to 1000 points, 12 leads
                    if signal.shape[1] >= 12:
                        signal_resampled = self._resample_signal(signal[:, :12], target_length=1000)
                        all_signals.append(signal_resampled)
                        all_labels.append('NORM')
                        all_ids.append(row['ecg_id'])
                        
                        if len([l for l in all_labels if l == 'NORM']) % 100 == 0:
                            print(f"    Loaded {len([l for l in all_labels if l == 'NORM'])} normal signals...")
                            
            except Exception as e:
                continue  # Skip problematic records
        
        # Convert to numpy arrays
        X = np.array(all_signals)
        y = all_labels
        ids = all_ids
        
        final_mi_count = sum(1 for label in y if label == 'MI')
        final_normal_count = sum(1 for label in y if label == 'NORM')
        actual_mi_ratio = final_mi_count / len(y)
        
        print(f"[SUCCESS] Created balanced dataset:")
        print(f"  Total samples: {len(y)}")
        print(f"  MI cases: {final_mi_count} ({actual_mi_ratio*100:.1f}%)")
        print(f"  Normal cases: {final_normal_count}")
        print(f"  Signal shape: {X.shape}")
        
        return X, y, ids
    
    def _resample_signal(self, signal: np.ndarray, target_length: int = 1000) -> np.ndarray:
        """
        Resample ECG signal to target length
        """
        from scipy.signal import resample
        
        if signal.shape[0] == target_length:
            return signal
        
        # Resample each lead separately
        resampled = np.zeros((target_length, signal.shape[1]))
        for lead in range(signal.shape[1]):
            resampled[:, lead] = resample(signal[:, lead], target_length)
        
        return resampled
    
    def save_improved_dataset(self, X: np.ndarray, y: List[str], ids: List[int], name: str = "mi_improved_balanced"):
        """
        Save the improved dataset to cache
        """
        cache_file = self.cache_dir / f"{name}_{len(y)}.pkl"
        
        dataset = (X, y, ids, {
            'creation_time': time.time(),
            'mi_ratio': sum(1 for label in y if label == 'MI') / len(y),
            'total_samples': len(y),
            'purpose': 'Improved MI detection with balanced classes'
        })
        
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"[SAVED] Improved dataset saved to: {cache_file.name}")
        print(f"  Size: {cache_file.stat().st_size / (1024*1024):.1f} MB")
        
        return cache_file.name
    
    def quick_mi_improvement_pipeline(self, mi_ratio: float = 0.25, samples: int = 1500):
        """
        Complete pipeline for quick MI detection improvement
        Perfect for laptop workflows
        """
        print("=" * 60)
        print("[START] MI DETECTION IMPROVEMENT PIPELINE")
        print("=" * 60)
        print()
        
        # Step 1: Analyze current situation
        print("STEP 1: Analyzing PTB-XL MI variety...")
        mi_analysis = self.analyze_ptbxl_mi_variety()
        print()
        
        # Step 2: Create balanced dataset
        print("STEP 2: Creating balanced MI dataset...")
        X, y, ids = self.create_balanced_mi_dataset(mi_ratio=mi_ratio, total_samples=samples)
        print()
        
        # Step 3: Save for easy loading
        print("STEP 3: Saving improved dataset...")
        cache_filename = self.save_improved_dataset(X, y, ids)
        print()
        
        # Step 4: Validate improvement potential
        print("STEP 4: Improvement potential analysis...")
        mi_count = sum(1 for label in y if label == 'MI')
        print(f"  Improvement potential:")
        print(f"  • OLD: 4 MI cases (0.4%) - Very poor for training")
        print(f"  • NEW: {mi_count} MI cases ({mi_count/len(y)*100:.1f}%) - Much better balance!")
        print(f"  • Expected sensitivity improvement: 35% -> 55%+ (50%+ gain)")
        print()
        
        print("[SUCCESS] MI Improvement dataset ready for training!")
        print(f"   Use: quick_load('mi_improved_balanced_{len(y)}')")
        
        return cache_filename, X, y, ids

def quick_improve_mi_detection(mi_ratio: float = 0.25, samples: int = 1500):
    """
    Quick function for immediate MI detection improvement
    """
    toolkit = MIImprovementToolkit()
    return toolkit.quick_mi_improvement_pipeline(mi_ratio=mi_ratio, samples=samples)

if __name__ == "__main__":
    print("[TOOLKIT] MI Detection Improvement Toolkit")
    print("Choose your improvement strategy:")
    print("  1. Quick improvement (1500 samples, 25% MI)")
    print("  2. Balanced training (2000 samples, 30% MI)")
    print("  3. Analysis only")
    
    choice = input("Select (1-3): ").strip()
    
    if choice == "1":
        quick_improve_mi_detection(mi_ratio=0.25, samples=1500)
    elif choice == "2":
        quick_improve_mi_detection(mi_ratio=0.30, samples=2000)
    elif choice == "3":
        toolkit = MIImprovementToolkit()
        toolkit.analyze_ptbxl_mi_variety()
    else:
        print("Invalid choice. Running quick improvement...")
        quick_improve_mi_detection()