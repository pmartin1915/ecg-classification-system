#!/usr/bin/env python3
"""Simple test to identify the failing import"""

print("Testing individual imports...")

try:
    import pandas as pd
    print("✅ pandas OK")
except Exception as e:
    print(f"❌ pandas failed: {e}")

try:
    import numpy as np
    print("✅ numpy OK")
except Exception as e:
    print(f"❌ numpy failed: {e}")

try:
    import sklearn
    print("✅ sklearn OK")
except Exception as e:
    print(f"❌ sklearn failed: {e}")

try:
    from imblearn.over_sampling import SMOTE
    print("✅ imbalanced-learn OK")
except Exception as e:
    print(f"❌ imbalanced-learn failed: {e}")

try:
    import psutil
    print("✅ psutil OK")
except Exception as e:
    print(f"❌ psutil failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ matplotlib OK")
except Exception as e:
    print(f"❌ matplotlib failed: {e}")

try:
    import seaborn as sns
    print("✅ seaborn OK")
except Exception as e:
    print(f"❌ seaborn failed: {e}")

print("\nTesting the actual problematic import...")
try:
    from models.training.phase4_model_training import Phase4Pipeline
    print("✅ Phase4Pipeline import successful!")
except Exception as e:
    print(f"❌ Phase4Pipeline import failed: {e}")
    import traceback
    traceback.print_exc()
