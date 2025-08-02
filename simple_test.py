#!/usr/bin/env python3
"""Simple test without Unicode characters"""

print("Testing individual imports...")

try:
    import pandas as pd
    print("OK: pandas")
except Exception as e:
    print(f"FAIL: pandas - {e}")

try:
    import numpy as np
    print("OK: numpy")
except Exception as e:
    print(f"FAIL: numpy - {e}")

try:
    import sklearn
    print("OK: sklearn")
except Exception as e:
    print(f"FAIL: sklearn - {e}")

try:
    from imblearn.over_sampling import SMOTE
    print("OK: imbalanced-learn")
except Exception as e:
    print(f"FAIL: imbalanced-learn - {e}")

try:
    import psutil
    print("OK: psutil")
except Exception as e:
    print(f"FAIL: psutil - {e}")

try:
    import matplotlib.pyplot as plt
    print("OK: matplotlib")
except Exception as e:
    print(f"FAIL: matplotlib - {e}")

try:
    import seaborn as sns
    print("OK: seaborn")
except Exception as e:
    print(f"FAIL: seaborn - {e}")

print("\nTesting the actual problematic import...")
try:
    from models.training.phase4_model_training import Phase4Pipeline
    print("OK: Phase4Pipeline import successful!")
except Exception as e:
    print(f"FAIL: Phase4Pipeline import - {e}")
    import traceback
    traceback.print_exc()
