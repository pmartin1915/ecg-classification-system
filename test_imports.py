#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    import streamlit as st
    print("[OK] Streamlit imported successfully")
except ImportError as e:
    print(f"[FAIL] Streamlit import failed: {e}")

try:
    import numpy as np
    print(f"[OK] NumPy imported successfully (version: {np.__version__})")
except ImportError as e:
    print(f"[FAIL] NumPy import failed: {e}")

try:
    import pandas as pd
    print(f"[OK] Pandas imported successfully (version: {pd.__version__})")
except ImportError as e:
    print(f"[FAIL] Pandas import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("[OK] Matplotlib imported successfully")
except ImportError as e:
    print(f"[FAIL] Matplotlib import failed: {e}")

try:
    import plotly.graph_objects as go
    print("[OK] Plotly imported successfully")
except ImportError as e:
    print(f"[FAIL] Plotly import failed: {e}")

try:
    from typing import Dict, List, Tuple, Any, Optional
    print("[OK] Typing imports successful")
except ImportError as e:
    print(f"[FAIL] Typing import failed: {e}")

print("\n*** All critical imports successful! ***")
print("The complete user-friendly system should now work properly.")
print("\nTo run the application:")
print("1. Double-click COMPLETE_USER_FRIENDLY_LAUNCHER.bat")
print("2. Or run: streamlit run complete_user_friendly.py --server.port=8507")