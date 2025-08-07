"""
Example: Using the Laptop-Optimized Data Loader
Perfect workflow for single-monitor development
"""

# Import the new laptop-optimized loader
from app.utils.laptop_optimized_loader import LaptopOptimizedLoader, quick_load, show_menu

def example_rapid_development():
    """
    Example: Rapid development workflow
    Perfect for coding on laptop with limited resources
    """
    print("=== EXAMPLE: RAPID DEVELOPMENT WORKFLOW ===")
    print()
    
    # Quick development with 25 samples (loads in 3-5 seconds)
    print("Step 1: Loading small dataset for rapid development...")
    X, y, metadata = quick_load("rapid", verbose=True)
    
    print(f"Step 2: Quick analysis on {len(y)} samples...")
    print(f"   Signal shape per sample: {X[0].shape}")
    print(f"   Available labels: {set(y)}")
    print()
    
    # Example: Quick feature extraction test
    print("Step 3: Quick feature extraction test...")
    mean_hr = X[:, :, 0].mean(axis=1)  # Simple heart rate proxy
    print(f"   Average heart rate proxy: {mean_hr.mean():.2f} ± {mean_hr.std():.2f}")
    
    print("✅ Rapid development complete! Ready for full training.")
    print()

def example_training_workflow():
    """
    Example: Training workflow with progress tracking
    """
    print("=== EXAMPLE: TRAINING WORKFLOW ===")
    print()
    
    # Initialize loader for performance tracking
    loader = LaptopOptimizedLoader()
    
    print("Step 1: Loading training dataset...")
    X_train, y_train, meta_train = loader.load_with_progress("training", verbose=True)
    
    print(f"Step 2: Loading validation dataset...")  
    X_val, y_val, meta_val = loader.load_with_progress("validation", verbose=True)
    
    print("Step 3: Performance report...")
    performance = loader.get_performance_report()
    print(f"   Total datasets loaded: {performance['total_loads']}")
    for dataset, perf in performance['average_performance'].items():
        print(f"   {dataset}: {perf['time']:.2f}s ({perf['performance']})")
    
    print("✅ Ready for model training!")
    print()

def example_mi_research():
    """
    Example: MI-focused research workflow
    """
    print("=== EXAMPLE: MI RESEARCH WORKFLOW ===")
    print()
    
    print("Step 1: Loading MI-focused dataset...")
    X_mi, y_mi, meta_mi = quick_load("mi_focused", verbose=True)
    
    print("Step 2: MI-specific analysis...")
    mi_indices = [i for i, label in enumerate(y_mi) if label == 'MI']
    norm_indices = [i for i, label in enumerate(y_mi) if label == 'NORM']
    
    if mi_indices:
        print(f"   Found {len(mi_indices)} MI cases for analysis")
        print(f"   Found {len(norm_indices)} normal cases for comparison")
        
        # Simple comparison
        mi_signals = X_mi[mi_indices]
        norm_signals = X_mi[norm_indices]
        
        print(f"   MI signal average: {mi_signals.mean():.4f}")
        print(f"   Normal signal average: {norm_signals.mean():.4f}")
    
    print("✅ MI research analysis complete!")
    print()

if __name__ == "__main__":
    print("🚀 LAPTOP-OPTIMIZED ECG WORKFLOW EXAMPLES")
    print("=" * 60)
    print()
    
    # Show available datasets first
    print("Available datasets:")
    show_menu()
    print()
    
    # Run examples
    example_rapid_development()
    example_mi_research()
    
    print("💡 WORKFLOW TIPS:")
    print("   • Start with 'rapid' (25 samples) for quick testing")
    print("   • Use 'training' (1000 samples) for algorithm development") 
    print("   • Use 'validation' (2000 samples) for final validation")
    print("   • Use 'mi_focused' for specific MI detection research")
    print("   • All datasets optimized for laptop performance!")
    print()
    print("📚 Usage in your code:")
    print("   from app.utils.laptop_optimized_loader import quick_load")
    print("   X, y, meta = quick_load('rapid')  # Fast 25-sample load")
    print("   X, y, meta = quick_load('training')  # 1000-sample training set")