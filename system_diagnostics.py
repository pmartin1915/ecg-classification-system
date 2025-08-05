"""
System Diagnostics Script
Checks system requirements and component availability
"""
import sys
import os
from pathlib import Path

def run_diagnostics():
    """Run comprehensive system diagnostics"""
    
    print("----------------------------------------------------------------------")
    print("                           SYSTEM DIAGNOSTICS")
    print("----------------------------------------------------------------------")
    
    # Python version
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Package checks
    packages = [
        ('streamlit', 'Streamlit'),
        ('wfdb', 'WFDB'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('plotly', 'Plotly')
    ]
    
    for package, display_name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"{display_name}: {version}")
        except ImportError:
            print(f"{display_name}: NOT INSTALLED")
    
    # Directory checks
    print(f"Project Directory: {'YES' if os.path.exists('app/main.py') else 'NO'}")
    print(f"Enhanced Features: {'YES' if os.path.exists('app/enhanced_main.py') else 'NO'}")
    print(f"Data Directory: {'YES' if os.path.exists('data') else 'NO'}")
    
    # Enhanced MI Model Status
    print()
    print("Enhanced MI Model Status:")
    try:
        models_dir = Path('models/trained_models')
        if models_dir.exists():
            enhanced_models = list(models_dir.glob('enhanced_mi_model_*.pkl'))
            print(f"Enhanced Models: {len(enhanced_models)} found")
            if enhanced_models:
                for model in enhanced_models:
                    print(f"  - {model.name}")
        else:
            print("Enhanced Models: Models directory not found")
    except Exception as e:
        print(f"Enhanced Models: Error checking - {e}")
    
    # Component status
    print()
    print("Component Status:")
    components = [
        ('app/enhanced_main.py', 'Enhanced Main Interface'),
        ('app/components/enhanced_explainability.py', 'AI Explainability'),
        ('app/utils/fast_prediction_pipeline.py', 'Fast Prediction Pipeline'),
        ('app/components/performance_monitor.py', 'Performance Monitor')
    ]
    
    for component_path, name in components:
        status = "Available" if os.path.exists(component_path) else "Missing"
        print(f"  {name}: {status}")

if __name__ == "__main__":
    run_diagnostics()