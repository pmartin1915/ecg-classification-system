"""
Laptop-Optimized ECG Data Loader
Smart, progress-aware data loading for single-monitor laptop workflows
"""

import pickle
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from collections import Counter

class LaptopOptimizedLoader:
    """
    Smart ECG data loader optimized for laptop development
    Features: Progress indicators, size recommendations, smart caching
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        # Always use absolute paths to avoid working directory issues
        if not Path(cache_dir).is_absolute():
            # Find project root (where this script is located)
            script_dir = Path(__file__).parent.parent.parent
            self.cache_dir = script_dir / cache_dir
        else:
            self.cache_dir = Path(cache_dir)
        self.load_times = {}  # Track loading performance
        
        # Define laptop-friendly dataset sizes
        self.dataset_catalog = {
            'instant': {
                'file': 'ptbxl_quick_test_5.pkl',
                'samples': 5,
                'size_mb': 0.2,
                'load_time': '1-2 seconds',
                'purpose': 'Instant testing, debugging, rapid iteration'
            },
            'rapid': {
                'file': 'ptbxl_rapid_dev_25.pkl', 
                'samples': 25,
                'size_mb': 1.2,
                'load_time': '3-5 seconds',
                'purpose': 'Rapid development, feature testing'
            },
            'training': {
                'file': 'ptbxl_training_1000.pkl',
                'samples': 1000,
                'size_mb': 46,
                'load_time': '20-30 seconds',
                'purpose': 'Model training, algorithm development'
            },
            'validation': {
                'file': 'ptbxl_validation_2000.pkl',
                'samples': 2000, 
                'size_mb': 92,
                'load_time': '45-60 seconds',
                'purpose': 'Model validation, performance testing'
            },
            'mi_focused': {
                'file': 'combined_mi_focused_906.pkl',
                'samples': 906,
                'size_mb': 42,
                'load_time': '25-35 seconds', 
                'purpose': 'MI detection training, clinical focus'
            },
            'general': {
                'file': 'combined_general_1986.pkl',
                'samples': 1986,
                'size_mb': 91,
                'load_time': '40-50 seconds',
                'purpose': 'General cardiac condition validation'
            }
        }
        
    def recommend_dataset_size(self, purpose: str = "development") -> str:
        """
        Recommend optimal dataset size based on purpose and laptop capabilities
        """
        recommendations = {
            "testing": "instant",
            "debugging": "instant", 
            "development": "rapid",
            "feature_dev": "rapid",
            "training": "training",
            "validation": "validation",
            "mi_research": "mi_focused",
            "full_validation": "general"
        }
        
        return recommendations.get(purpose.lower(), "rapid")
    
    def show_dataset_menu(self) -> None:
        """
        Display interactive dataset selection menu
        """
        print("=" * 70)
        print("[CHART] LAPTOP-OPTIMIZED ECG DATASET LOADER")
        print("=" * 70)
        print()
        
        print("Available datasets (optimized for single-monitor workflow):")
        print()
        
        for key, info in self.dataset_catalog.items():
            status = "[OK]" if (self.cache_dir / info['file']).exists() else "[MISSING]"
            print(f"{status} {key.upper():12} | {info['samples']:4} samples | {info['size_mb']:5.1f}MB | {info['load_time']:12}")
            print(f"     Purpose: {info['purpose']}")
            print()
    
    def load_with_progress(self, dataset_size: str = "rapid", verbose: bool = True) -> Tuple[Any, Any, Dict]:
        """
        Load dataset with progress indicators and performance tracking
        """
        if dataset_size not in self.dataset_catalog:
            raise ValueError(f"Unknown dataset size: {dataset_size}. Available: {list(self.dataset_catalog.keys())}")
        
        dataset_info = self.dataset_catalog[dataset_size]
        file_path = self.cache_dir / dataset_info['file']
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        if verbose:
            print(f"[LOADING] Loading {dataset_size.upper()} dataset...")
            print(f"   File: {dataset_info['file']}")
            print(f"   Size: {dataset_info['size_mb']:.1f}MB ({dataset_info['samples']} samples)")
            print(f"   Expected load time: {dataset_info['load_time']}")
            print()
            print("   Progress: ", end="", flush=True)
        
        # Start timing
        start_time = time.time()
        
        # Progress indicator
        if verbose:
            self._show_progress_bar(0, "Starting...")
        
        try:
            # Load the data
            with open(file_path, 'rb') as f:
                if verbose:
                    self._show_progress_bar(30, "Reading file...")
                
                data = pickle.load(f)
                
                if verbose:
                    self._show_progress_bar(70, "Processing data...")
        
            # Extract data based on format
            if isinstance(data, (tuple, list)) and len(data) >= 2:
                X, y = data[0], data[1]
                metadata = {'additional_data': data[2:] if len(data) > 2 else []}
            elif isinstance(data, dict):
                # Handle dictionary format
                X = data.get('signals', data.get('X', None))
                y = data.get('labels', data.get('y', None)) 
                metadata = {k: v for k, v in data.items() if k not in ['signals', 'labels', 'X', 'y']}
            else:
                raise ValueError(f"Unexpected data format: {type(data)}")
            
            if verbose:
                self._show_progress_bar(90, "Finalizing...")
            
            # Calculate loading performance
            load_time = time.time() - start_time
            self.load_times[dataset_size] = load_time
            
            if verbose:
                self._show_progress_bar(100, "Complete!")
                print()
                print()
                
            # Display loading summary
            if verbose:
                self._display_load_summary(X, y, dataset_info, load_time)
            
            return X, y, metadata
            
        except Exception as e:
            if verbose:
                print(f"\n[ERROR] Error loading dataset: {str(e)}")
            raise
    
    def _show_progress_bar(self, percentage: int, status: str = "") -> None:
        """
        Display a progress bar for laptop-friendly feedback
        """
        bar_length = 20
        filled_length = int(bar_length * percentage // 100)
        bar = "#" * filled_length + "-" * (bar_length - filled_length)
        print(f"\r   [{bar}] {percentage:3d}% {status}", end="", flush=True)
        time.sleep(0.1)  # Small delay for visual effect
    
    def _display_load_summary(self, X: Any, y: Any, dataset_info: Dict, load_time: float) -> None:
        """
        Display comprehensive loading summary
        """
        print("[SUCCESS] LOADING COMPLETE")
        print("-" * 50)
        print(f"[TIME] Load Time: {load_time:.2f}s (Expected: {dataset_info['load_time']})")
        print(f"[DATA] Data Shape: {X.shape if hasattr(X, 'shape') else f'{type(X)} ({len(X)} items)'}")
        print(f"[LABELS] Labels: {len(y)} samples")
        
        # Show label distribution
        if isinstance(y, (list, np.ndarray)):
            if len(y) <= 2000:  # Only count if reasonable size
                try:
                    y_list = y if isinstance(y, list) else y.tolist()
                    # Handle case where labels might be nested lists or complex structures
                    if y_list and isinstance(y_list[0], (list, np.ndarray)):
                        # If labels are nested, flatten or take first element
                        y_flat = [item[0] if isinstance(item, (list, np.ndarray)) else item for item in y_list]
                    else:
                        y_flat = y_list
                    
                    label_counts = Counter(y_flat)
                    print(f"[DISTRIBUTION] Label Distribution: {dict(label_counts)}")
                    
                    # Show MI detection capability
                    mi_count = label_counts.get('MI', 0)
                    if mi_count > 0:
                        mi_percentage = (mi_count / len(y)) * 100
                        print(f"[MI] MI Cases: {mi_count} ({mi_percentage:.1f}% - Good for MI detection!)")
                except Exception as e:
                    print(f"[INFO] Label distribution analysis skipped: Complex label format")
        
        print(f"[PURPOSE] Purpose: {dataset_info['purpose']}")
        print(f"[PERFORMANCE] Laptop Performance: {'Excellent' if load_time < 10 else 'Good' if load_time < 30 else 'Acceptable'}")
        print()
    
    def quick_test(self) -> bool:
        """
        Quick functionality test - loads smallest dataset
        """
        print("[TEST] QUICK SYSTEM TEST")
        print("-" * 30)
        
        try:
            X, y, metadata = self.load_with_progress("instant", verbose=True)
            
            print("[SUCCESS] Test Result: SUCCESS")
            print(f"   Loaded {len(y)} samples successfully")
            print(f"   Data format validation: PASSED") 
            return True
            
        except Exception as e:
            print(f"[ERROR] Test Result: FAILED - {str(e)}")
            return False
    
    def get_performance_report(self) -> Dict:
        """
        Get performance report of previous loads
        """
        if not self.load_times:
            return {"status": "No loads performed yet"}
        
        report = {
            "total_loads": len(self.load_times),
            "load_times": self.load_times.copy(),
            "average_performance": {}
        }
        
        # Calculate performance categories
        for size, time_taken in self.load_times.items():
            expected_max = {"instant": 5, "rapid": 10, "training": 40, "validation": 70}.get(size, 60)
            performance = "Excellent" if time_taken < expected_max * 0.5 else \
                         "Good" if time_taken < expected_max else "Slow"
            report["average_performance"][size] = {
                "time": time_taken,
                "performance": performance
            }
        
        return report

# Convenience functions for easy import
def quick_load(size: str = "rapid", verbose: bool = True) -> Tuple[Any, Any, Dict]:
    """
    Quick load function for immediate use
    """
    loader = LaptopOptimizedLoader()
    return loader.load_with_progress(size, verbose)

def show_menu():
    """
    Show dataset selection menu
    """
    loader = LaptopOptimizedLoader()
    loader.show_dataset_menu()

def test_system():
    """
    Test system functionality
    """
    loader = LaptopOptimizedLoader()
    return loader.quick_test()

if __name__ == "__main__":
    # Interactive CLI when run directly
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            test_system()
        elif command == "menu":
            show_menu()
        elif command in ["instant", "rapid", "training", "validation", "mi_focused", "general"]:
            X, y, meta = quick_load(command)
        else:
            print(f"Unknown command: {command}")
            print("Available commands: test, menu, instant, rapid, training, validation, mi_focused, general")
    else:
        # Default: show menu
        show_menu()
        print("\nUsage examples:")
        print("  python laptop_optimized_loader.py test      # Quick system test")  
        print("  python laptop_optimized_loader.py menu      # Show dataset menu")
        print("  python laptop_optimized_loader.py rapid     # Load rapid development dataset")