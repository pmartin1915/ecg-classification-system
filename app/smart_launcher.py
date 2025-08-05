"""
Smart Launcher for ECG Classification System
Automatically selects enhanced or standard interface based on available models
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_enhanced_capabilities():
    """Check if enhanced MI models are available"""
    try:
        # Check for enhanced model files
        enhanced_model_paths = [
            project_root / "models" / "trained_models" / "enhanced_mi_model_enhanced_rf.pkl",
            project_root / "models" / "trained_models" / "enhanced_mi_model_xgboost_mi.pkl", 
            project_root / "models" / "trained_models" / "enhanced_mi_model_ensemble.pkl"
        ]
        
        for model_path in enhanced_model_paths:
            if model_path.exists():
                return True
        
        return False
        
    except Exception:
        return False

def launch_application():
    """Launch the appropriate application version"""
    
    print("🫀 ECG CLASSIFICATION SYSTEM - SMART LAUNCHER")
    print("=" * 60)
    
    enhanced_available = check_enhanced_capabilities()
    
    if enhanced_available:
        print("✅ Enhanced MI detection models found!")
        print("🚀 Launching Enhanced ECG Classification System...")
        print("   Features: Advanced MI detection with 70%+ sensitivity")
        print("   Models: Ensemble ML with 150+ clinical features")
        
        try:
            import enhanced_main
            enhanced_main.main()
        except Exception as e:
            print(f"❌ Enhanced version failed to load: {e}")
            print("🔄 Falling back to standard version...")
            fallback_to_standard()
    else:
        print("📊 Standard configuration detected")
        print("🚀 Launching Standard ECG Classification System...")
        print("   Note: Enhanced MI models available after training")
        print("   Run: python enhanced_mi_detection_system.py")
        
        fallback_to_standard()

def fallback_to_standard():
    """Launch standard application"""
    try:
        import main
        main.main()
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        print("🔧 Please check your installation and dependencies")

if __name__ == "__main__":
    launch_application()