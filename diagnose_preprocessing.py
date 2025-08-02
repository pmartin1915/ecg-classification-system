"""
Diagnose where preprocessing import is hanging
"""
print("Starting preprocessing import diagnosis...")
print("=" * 60)

import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
if project_root.name == 'app':
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

print("Testing individual preprocessing module imports...\n")

# Test 1: Check if the preprocessing package exists
print("1. Checking preprocessing package...")
try:
    import models.preprocessing
    print("✅ models.preprocessing package found")
    print(f"   Location: {models.preprocessing.__file__}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 2: Import signal_quality
print("\n2. Importing signal_quality...")
sys.stdout.flush()  # Force output
try:
    from models.preprocessing import signal_quality
    print("✅ signal_quality imported")
except Exception as e:
    print(f"❌ Error importing signal_quality: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Import signal_filters
print("\n3. Importing signal_filters...")
sys.stdout.flush()  # Force output
try:
    from models.preprocessing import signal_filters
    print("✅ signal_filters imported")
except Exception as e:
    print(f"❌ Error importing signal_filters: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check preprocessing_pipeline exists
print("\n4. Checking if preprocessing_pipeline.py exists...")
pipeline_path = Path(project_root) / "models" / "preprocessing" / "preprocessing_pipeline.py"
if pipeline_path.exists():
    print(f"✅ File exists: {pipeline_path}")
    print(f"   File size: {pipeline_path.stat().st_size} bytes")
    
    # Read first few lines to check for issues
    print("\n   First 10 lines of preprocessing_pipeline.py:")
    with open(pipeline_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"   {i+1}: {line.rstrip()}")
else:
    print(f"❌ File not found: {pipeline_path}")

# Test 5: Try importing with verbose error catching
print("\n5. Attempting to import preprocessing_pipeline with detailed error catching...")
sys.stdout.flush()

try:
    # Set a flag to see if we even start the import
    print("   Starting import...")
    sys.stdout.flush()
    
    # Try to import
    from models.preprocessing import preprocessing_pipeline
    
    print("✅ preprocessing_pipeline imported successfully!")
    
except SyntaxError as e:
    print(f"❌ Syntax Error in preprocessing_pipeline: {e}")
    print(f"   File: {e.filename}, Line: {e.lineno}")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnosis complete!")
print("=" * 60)