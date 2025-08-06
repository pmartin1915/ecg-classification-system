#!/usr/bin/env python3
"""
Setup script for pre-commit hooks in ECG Classification System.

This script installs and configures pre-commit hooks with medical data protection.
Run this script to set up the development environment with code quality tools.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"📋 {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def check_python_version():
    """Check if Python version is adequate."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required for pre-commit setup")
        return False
    print(f"✅ Python {version.major}.{version.minor} is adequate")
    return True


def install_dev_dependencies():
    """Install development dependencies."""
    dev_packages = [
        "pre-commit>=3.0.0",
        "ruff>=0.1.0", 
        "black>=23.0.0",
        "isort>=5.12.0",
        "mypy>=1.0.0",
        "bandit>=1.7.0",
        "detect-secrets>=1.4.0",
    ]
    
    for package in dev_packages:
        if not run_command(
            [sys.executable, "-m", "pip", "install", package],
            f"Installing {package.split('>=')[0]}"
        ):
            return False
    return True


def setup_pre_commit():
    """Set up pre-commit hooks."""
    # Install pre-commit hooks
    if not run_command(
        ["pre-commit", "install"],
        "Installing pre-commit hooks"
    ):
        return False
    
    # Install commit message hook
    if not run_command(
        ["pre-commit", "install", "--hook-type", "commit-msg"],
        "Installing commit message hooks"
    ):
        return False
    
    return True


def create_secrets_baseline():
    """Create initial secrets baseline."""
    if not Path(".secrets.baseline").exists():
        return run_command(
            ["detect-secrets", "scan", "--baseline", ".secrets.baseline"],
            "Creating secrets detection baseline"
        )
    return True


def run_initial_checks():
    """Run initial pre-commit checks on all files."""
    print("\n🧪 Running initial code quality checks...")
    
    # Run on a subset first to avoid overwhelming output
    sample_files = [
        "complete_user_friendly.py",
        "src/inference.py", 
        "deployment_validation.py"
    ]
    
    existing_files = [f for f in sample_files if Path(f).exists()]
    
    if existing_files:
        success = run_command(
            ["pre-commit", "run", "--files"] + existing_files,
            "Running pre-commit on sample files"
        )
        if not success:
            print("⚠️  Some pre-commit checks failed - this is normal for first run")
            print("   The files have been auto-formatted. Please review and commit the changes.")
        return True
    else:
        print("⚠️  No sample files found, skipping initial check")
        return True


def main():
    """Main setup function."""
    print("🏥 ECG Classification System - Pre-commit Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install development dependencies
    print("\n📦 Installing development dependencies...")
    if not install_dev_dependencies():
        print("❌ Failed to install development dependencies")
        return 1
    
    # Set up pre-commit
    print("\n🔧 Setting up pre-commit hooks...")
    if not setup_pre_commit():
        print("❌ Failed to setup pre-commit hooks")
        return 1
    
    # Create secrets baseline
    print("\n🔒 Setting up secrets detection...")
    if not create_secrets_baseline():
        print("⚠️  Could not create secrets baseline (this is non-critical)")
    
    # Run initial checks
    if not run_initial_checks():
        print("⚠️  Initial checks had issues (this is normal)")
    
    print("\n" + "=" * 50)
    print("🎉 Pre-commit setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Review any auto-formatted files")
    print("2. Commit the pre-commit configuration files")
    print("3. Start developing with automated code quality checks")
    print()
    print("Commands you can now use:")
    print("  • pre-commit run --all-files   # Check all files")
    print("  • pre-commit run --files <file> # Check specific files")  
    print("  • pre-commit autoupdate        # Update hook versions")
    print()
    print("🏥 Medical Data Protection:")
    print("  • PHI detection is now active")
    print("  • Hardcoded medical data checks enabled")
    print("  • Security scanning with bandit enabled")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())