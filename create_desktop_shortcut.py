"""
Create Desktop Shortcut for ECG Classification System
Professional shortcut with icon and description
"""
import os
import winshell
from win32com.client import Dispatch
from pathlib import Path

def create_desktop_shortcut():
    """Create a professional desktop shortcut"""
    print("CREATING ECG CLASSIFICATION SYSTEM DESKTOP SHORTCUT")
    print("=" * 60)
    
    try:
        # Get desktop path
        desktop = winshell.desktop()
        print(f"Desktop location: {desktop}")
        
        # Get current directory
        current_dir = Path(__file__).parent.absolute()
        bat_file = current_dir / "launch_ecg_system.bat"
        
        # Create shortcut
        shortcut_path = os.path.join(desktop, "ECG Classification System.lnk")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        
        # Set shortcut properties
        shortcut.Targetpath = str(bat_file)
        shortcut.WorkingDirectory = str(current_dir)
        shortcut.Description = "ECG Classification System - Enhanced MI Detection for Healthcare"
        shortcut.IconLocation = "shell32.dll,23"  # Medical/health icon
        
        # Save shortcut
        shortcut.save()
        
        print(f"SUCCESS: Desktop shortcut created!")
        print(f"Location: {shortcut_path}")
        print(f"Target: {bat_file}")
        print("\nShortcut Features:")
        print("- Double-click to launch ECG system")
        print("- Automatic system status check")
        print("- Professional clinical interface")
        print("- Enhanced MI detection capabilities")
        
        return True
        
    except Exception as e:
        print(f"ERROR creating shortcut: {e}")
        print("\nAlternative: You can manually create a shortcut to:")
        print(f"Target: {current_dir}\\launch_ecg_system.bat")
        return False

if __name__ == "__main__":
    success = create_desktop_shortcut()
    
    if success:
        print("\n" + "=" * 60)
        print("DESKTOP SHORTCUT READY!")
        print("=" * 60)
        print("Your ECG Classification System is now accessible from:")
        print("- Desktop shortcut: 'ECG Classification System'")
        print("- One-click launch with professional interface")
        print("- Automatic system validation")
        print("\nDouble-click the desktop icon to start your clinical AI system!")
    else:
        print("\nManual setup: Right-click desktop > New > Shortcut")
        print("Then point to: launch_ecg_system.bat")