Set WshShell = CreateObject("WScript.Shell")
Set oShellLink = WshShell.CreateShortcut(WshShell.SpecialFolders("Desktop") & "\Professional ECG Classification System.lnk")

' Get the current directory where this script is located
strCurrentDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Set shortcut properties
oShellLink.TargetPath = strCurrentDir & "\STREAMLINED_LAUNCHER.bat"
oShellLink.WindowStyle = 1
oShellLink.IconLocation = "shell32.dll,253"
oShellLink.Description = "Professional ECG Classification System - Medical Education Platform"
oShellLink.WorkingDirectory = strCurrentDir
oShellLink.Save

' Show success message
WScript.Echo "Desktop shortcut created successfully!" & vbCrLf & vbCrLf & _
            "Shortcut Name: Professional ECG Classification System" & vbCrLf & _
            "Location: Desktop" & vbCrLf & _
            "Target: STREAMLINED_LAUNCHER.bat" & vbCrLf & vbCrLf & _
            "You can now launch your professional ECG system from the desktop!"