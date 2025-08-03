"""
Manual Desktop Shortcut Instructions
Creates the shortcut files and provides instructions
"""
import os
from pathlib import Path

def create_shortcut_instructions():
    """Provide manual shortcut creation instructions"""
    print("ECG CLASSIFICATION SYSTEM - DESKTOP SHORTCUT SETUP")
    print("=" * 60)
    
    current_dir = Path(__file__).parent.absolute()
    bat_file = current_dir / "launch_ecg_system.bat"
    
    print("\nOPTION 1: AUTOMATIC (if you have administrator rights)")
    print("Run this command in an admin command prompt:")
    print(f'powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut(\'%userprofile%\\Desktop\\ECG Classification System.lnk\'); $s.TargetPath=\'{bat_file}\'; $s.Save()"')
    
    print("\nOPTION 2: MANUAL (easiest method)")
    print("1. Right-click on your Desktop")
    print("2. Select 'New' > 'Shortcut'")
    print(f"3. In the location box, paste this path:")
    print(f"   {bat_file}")
    print("4. Click 'Next'")
    print("5. Name it: 'ECG Classification System'")
    print("6. Click 'Finish'")
    
    print("\nOPTION 3: COPY AND PASTE")
    print("1. Navigate to this folder in Windows Explorer:")
    print(f"   {current_dir}")
    print("2. Right-click 'launch_ecg_system.bat'")
    print("3. Select 'Send to' > 'Desktop (create shortcut)'")
    print("4. Rename the shortcut to 'ECG Classification System'")
    
    print(f"\nYour launch file is ready at:")
    print(f"{bat_file}")
    
    print("\n" + "=" * 60)
    print("SHORTCUT BENEFITS:")
    print("=" * 60)
    print("✓ One-click launch of your ECG system")
    print("✓ Automatic system status check")
    print("✓ Professional clinical interface startup")
    print("✓ Enhanced MI detection ready")
    print("✓ Perfect for demonstrations")
    
    return str(bat_file)

if __name__ == "__main__":
    bat_path = create_shortcut_instructions()
    
    print("\nTEST YOUR LAUNCHER:")
    print(f"Double-click: {Path(bat_path).name}")
    print("This will start your professional ECG classification system!")