"""
Simple test script for Phase 2 to verify imports and basic functionality
"""
print("Starting Phase 2 Test Script...")
print("=" * 60)

import sys
from pathlib import Path

# Print current location
print(f"Script location: {Path(__file__).absolute()}")
print(f"Current working directory: {Path.cwd()}")

# Add project root to path
project_root = Path(__file__).parent
if project_root.name == 'app':
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
print(f"Project root: {project_root}")
print(f"Python path includes project root: {project_root in [Path(p) for p in sys.path]}")

# Try to import modules one by one
print("\n1. Testing imports...")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except Exception as e:
    print(f"❌ Pandas import failed: {e}")
    sys.exit(1)

try:
    from config.settings import DATA_DIR
    print("✅ Config imported successfully")
    print(f"   DATA_DIR: {DATA_DIR}")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    sys.exit(1)

try:
    from app.utils.dataset_manager import DatasetManager
    print("✅ DatasetManager imported successfully")
except Exception as e:
    print(f"❌ DatasetManager import failed: {e}")
    sys.exit(1)

print("\n2. Testing data loading...")
try:
    # Try to load just 1 sample
    manager = DatasetManager()
    results = manager.load_ptbxl_complete(max_records=1)
    print(f"✅ Loaded {len(results['X'])} sample(s)")
    print(f"   Shape: {results['X'].shape}")
    print(f"   Labels: {results['labels']}")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing preprocessing imports...")
try:
    from models.preprocessing.preprocessing_pipeline import PreprocessingPipeline
    print("✅ PreprocessingPipeline imported successfully")
except Exception as e:
    print(f"❌ PreprocessingPipeline import failed: {e}")
    # Try individual imports
    try:
        from models.preprocessing import signal_quality
        print("   ✅ signal_quality module found")
    except:
        print("   ❌ signal_quality module not found")
    
    try:
        from models.preprocessing import signal_filters
        print("   ✅ signal_filters module found")
    except:
        print("   ❌ signal_filters module not found")
        
    try:
        from models.preprocessing import preprocessing_pipeline
        print("   ✅ preprocessing_pipeline module found")
    except Exception as e2:
        print(f"   ❌ preprocessing_pipeline module not found: {e2}")
        
    # Check if models.preprocessing exists
    try:
        import models.preprocessing
        print(f"   ℹ️ models.preprocessing __file__: {models.preprocessing.__file__}")
        print(f"   ℹ️ models.preprocessing contents: {dir(models.preprocessing)}")
    except Exception as e3:
        print(f"   ❌ Can't inspect models.preprocessing: {e3}")

print("\n" + "=" * 60)
print("Test script completed!")
print("=" * 60)