# Save as check_phase3_results.py
from pathlib import Path
import pickle
import os

print("🔍 Searching for Phase 3 feature extraction results...\n")

# Common locations for Phase 3 output
possible_locations = [
    'data/processed/feature_extraction_results.pkl',
    'data/processed/phase3_features.pkl',
    'data/processed/features.pkl',
    'data/processed/extracted_features.pkl',
    'data/features/extracted_features.pkl',
    'data/features/features.pkl',
    # Check in feature_extraction module output
    'models/feature_extraction/output/features.pkl',
    'models/feature_extraction/results/features.pkl',
]

# Also check for any .pkl files in processed directory
processed_dir = Path('data/processed')
if processed_dir.exists():
    possible_locations.extend([str(f) for f in processed_dir.glob('*.pkl')])

# Check features directory
features_dir = Path('data/features')
if features_dir.exists():
    possible_locations.extend([str(f) for f in features_dir.glob('*.pkl')])

found_files = []

for location in possible_locations:
    path = Path(location)
    if path.exists():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        found_files.append((path, size_mb))
        print(f"✅ Found: {path} ({size_mb:.2f} MB)")
        
        # Try to peek inside
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                print(f"   Contents: {list(data.keys())}")
                if 'X_features' in data or 'X' in data or 'features' in data:
                    print(f"   ⭐ This looks like Phase 3 output!")
                    
                    # Get data shape
                    X = data.get('X_features', data.get('X', data.get('features')))
                    if X is not None:
                        print(f"   Feature matrix shape: {X.shape if hasattr(X, 'shape') else 'Unknown'}")
                        
        except Exception as e:
            print(f"   Could not read file: {e}")
        
        print()

if not found_files:
    print("❌ No Phase 3 results found.\n")
    print("Do you need to run Phase 3? Looking for feature extraction script...")
    
    # Check if feature extraction pipeline exists
    fe_pipeline = Path('models/feature_extraction/feature_extraction_pipeline.py')
    if fe_pipeline.exists():
        print(f"\n✅ Found feature extraction pipeline at: {fe_pipeline}")
        print("\nTo run Phase 3, you might need to:")
        print("1. Run the feature extraction pipeline")
        print("2. Or run the phase3_feature_extraction.ipynb notebook")
else:
    print(f"\n📊 Found {len(found_files)} potential Phase 3 output files")
    print("\nIf none of these are the correct Phase 3 output, you may need to:")
    print("1. Run Phase 3 feature extraction")
    print("2. Or rename/move the correct file to: data/processed/feature_extraction_results.pkl")

# Check notebooks
print("\n📓 Checking notebooks:")
notebooks_dir = Path('notebooks')
if notebooks_dir.exists():
    for nb in notebooks_dir.glob('*.ipynb'):
        print(f"   - {nb.name}")