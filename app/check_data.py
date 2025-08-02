# Save this as check_data.py in your project root
from pathlib import Path
import os

print("Checking for data files...\n")

# Check data directories
data_dir = Path('data')
processed_dir = data_dir / 'processed'
raw_dir = data_dir / 'raw'

print("📁 Raw data files:")
if raw_dir.exists():
    for file in raw_dir.iterdir():
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.1f} MB)")
else:
    print("   - No raw directory found")

print("\n📁 Processed data files:")
if processed_dir.exists():
    for file in processed_dir.iterdir():
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   - {file.name} ({size_mb:.1f} MB)")
else:
    print("   - No processed directory found")

# Check for specific Phase 3 outputs
phase3_files = [
    'data/processed/feature_extraction_results.pkl',
    'data/processed/phase3_features.pkl',
    'data/processed/features.pkl'
]

print("\n🔍 Looking for Phase 3 output files:")
found_any = False
for file_path in phase3_files:
    if Path(file_path).exists():
        print(f"   ✅ Found: {file_path}")
        found_any = True
    else:
        print(f"   ❌ Not found: {file_path}")

if not found_any:
    print("\n⚠️  No Phase 3 results found. You need to either:")
    print("   1. Run Phase 3 to extract features from your ECG data")
    print("   2. Use the test script with synthetic data")