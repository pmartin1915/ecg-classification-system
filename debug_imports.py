#!/usr/bin/env python3
"""Debug script to test imports for test_phase4.py"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("Testing imports for test_phase4.py...")
print("=" * 50)

# Test basic imports
try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import pickle
    print("✅ pickle imported successfully")
except ImportError as e:
    print(f"❌ pickle import failed: {e}")

# Test project-specific imports
try:
    from models.training.phase4_model_training import Phase4Pipeline
    print("✅ Phase4Pipeline imported successfully")
except ImportError as e:
    print(f"❌ Phase4Pipeline import failed: {e}")
except Exception as e:
    print(f"❌ Phase4Pipeline import error: {e}")

try:
    from config.model_config import ModelTrainingConfig
    print("✅ ModelTrainingConfig imported successfully")
except ImportError as e:
    print(f"❌ ModelTrainingConfig import failed: {e}")
except Exception as e:
    print(f"❌ ModelTrainingConfig import error: {e}")

try:
    from config.settings import DATA_DIR
    print("✅ DATA_DIR imported successfully")
    print(f"   DATA_DIR path: {DATA_DIR}")
except ImportError as e:
    print(f"❌ DATA_DIR import failed: {e}")
except Exception as e:
    print(f"❌ DATA_DIR import error: {e}")

try:
    from models.training import ModelTrainer
    print("✅ ModelTrainer imported successfully")
except ImportError as e:
    print(f"❌ ModelTrainer import failed: {e}")
except Exception as e:
    print(f"❌ ModelTrainer import error: {e}")

# Check if required files exist
print("\nChecking required files...")
print("-" * 30)

required_files = [
    "models/training/phase4_model_training.py",
    "config/model_config.py",
    "config/settings.py",
    "models/training/model_trainer.py",
    "models/training/__init__.py"
]

for file_path in required_files:
    path = Path(file_path)
    if path.exists():
        print(f"✅ {file_path} exists")
    else:
        print(f"❌ {file_path} missing")

# Check data directory
print("\nChecking data directories...")
print("-" * 30)

try:
    from config.settings import DATA_DIR
    data_paths = [
        DATA_DIR / 'processed',
        DATA_DIR / 'processed' / 'feature_extraction_results.pkl',
        DATA_DIR / 'processed' / 'phase3_features.pkl'
    ]
    
    for path in data_paths:
        if path.exists():
            print(f"✅ {path} exists")
        else:
            print(f"❌ {path} missing")
            
except Exception as e:
    print(f"❌ Error checking data paths: {e}")

print("\nDiagnosis complete!")
