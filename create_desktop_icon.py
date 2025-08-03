#!/usr/bin/env python3
"""
Create Desktop Icon for Comprehensive ECG Classification System
Professional deployment with Windows shortcut creation
"""

import os
import sys
import winshell
from pathlib import Path
from win32com.client import Dispatch

def create_desktop_shortcut():
    """Create a professional desktop shortcut for the ECG system"""
    
    print("🫀 Creating Desktop Shortcut for Comprehensive ECG System")
    print("=" * 70)
    
    try:
        # Get current directory (project root)
        project_dir = Path(__file__).parent.absolute()
        launcher_path = project_dir / "LAUNCH_ECG_COMPREHENSIVE.bat"
        
        # Verify launcher exists
        if not launcher_path.exists():
            print(f"❌ ERROR: Launcher not found at {launcher_path}")
            return False
        
        # Get desktop path
        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, "Comprehensive ECG Classification System.lnk")
        
        # Create shortcut
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        
        # Configure shortcut properties
        shortcut.Targetpath = str(launcher_path)
        shortcut.WorkingDirectory = str(project_dir)
        shortcut.Description = "Comprehensive ECG Classification System - Advanced 30-Condition Cardiac Analysis"
        
        # Try to set an icon (use Windows default if custom not available)
        try:
            # Use a medical/heart icon if available in Windows
            shortcut.IconLocation = "shell32.dll,23"  # Medical cross icon
        except:
            pass  # Use default icon if custom fails
        
        # Save the shortcut
        shortcut.save()
        
        print(f"✅ SUCCESS: Desktop shortcut created!")
        print(f"   Location: {shortcut_path}")
        print(f"   Target: {launcher_path}")
        print(f"   Working Directory: {project_dir}")
        
        # Verify shortcut creation
        if os.path.exists(shortcut_path):
            print(f"✅ Shortcut verified and ready to use!")
            return True
        else:
            print(f"❌ ERROR: Shortcut creation failed - file not found")
            return False
        
    except ImportError as e:
        print(f"❌ ERROR: Missing required modules - {e}")
        print(f"   Please install: pip install pywin32 winshell")
        return False
    
    except Exception as e:
        print(f"❌ ERROR: Failed to create shortcut - {e}")
        return False

def create_project_summary():
    """Create a summary of the project for desktop deployment"""
    
    summary_path = Path("DESKTOP_DEPLOYMENT_SUMMARY.md")
    
    content = """# 🫀 Comprehensive ECG Classification System
## Desktop Deployment Summary

### 🚀 Quick Launch
**Double-click the desktop icon:** `Comprehensive ECG Classification System`

### 🎯 System Capabilities
- **30 Cardiac Conditions** detection
- **66,540 Clinical Records** available
- **Real-time Analysis** (<3 seconds)
- **Clinical Priority Alerts**
- **Professional Medical Interface**

### 📊 Condition Categories
1. **Myocardial Infarction** (4 types): AMI, IMI, LMI, PMI
2. **Arrhythmias** (6 types): AFIB, AFLT, VTAC, SVTAC, PVC, PAC
3. **Conduction Disorders** (9 types): AV Blocks, Bundle Blocks, WPW
4. **Structural Changes** (11 types): Hypertrophy, Ischemia, ST-T changes

### 🏥 Clinical Applications
- **Medical Education**: Advanced ECG training
- **Healthcare Research**: Cardiac analysis studies
- **Clinical Decision Support**: Professional diagnostic assistance

### 💻 System Requirements
- **Python 3.11+** installed
- **8GB RAM** minimum (16GB recommended)
- **5GB Storage** available
- **Windows 10/11**

### 🔧 Troubleshooting
1. **Desktop Icon**: Run `create_desktop_icon.py` to recreate
2. **Dependencies**: Run `pip install -r requirements.txt`
3. **Browser Access**: http://localhost:8501
4. **Support**: Check DEPLOYMENT.md for detailed help

---
**🎉 Ready for Professional Cardiac Analysis!**
"""
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"📄 Created deployment summary: {summary_path}")

def main():
    """Main function to set up desktop deployment"""
    
    print("🫀 COMPREHENSIVE ECG SYSTEM - DESKTOP DEPLOYMENT SETUP")
    print("=" * 70)
    
    # Check if running on Windows
    if sys.platform != "win32":
        print("❌ ERROR: Desktop shortcut creation is only supported on Windows")
        print("   For other systems, manually create a shortcut to LAUNCH_ECG_COMPREHENSIVE.bat")
        return False
    
    # Create desktop shortcut
    shortcut_success = create_desktop_shortcut()
    
    # Create deployment summary
    create_project_summary()
    
    if shortcut_success:
        print("\n🎉 DESKTOP DEPLOYMENT COMPLETE!")
        print("=" * 50)
        print("✅ Desktop shortcut created")
        print("✅ Deployment documentation ready")
        print("✅ System ready for one-click launch")
        print("\n🚀 TO USE:")
        print("   1. Double-click desktop icon")
        print("   2. Select option 1 (Full System)")
        print("   3. System opens in browser automatically")
        print("\n📚 For detailed help, see DEPLOYMENT.md")
        return True
    else:
        print("\n⚠️  DESKTOP DEPLOYMENT INCOMPLETE")
        print("   Manual shortcut creation may be required")
        return False

if __name__ == "__main__":
    success = main()
    input(f"\nPress Enter to exit...")
    sys.exit(0 if success else 1)