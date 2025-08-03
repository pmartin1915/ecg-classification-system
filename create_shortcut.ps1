
 = New-Object -ComObject WScript.Shell
 = .CreateShortcut("C:\Users\perry\Desktop\Comprehensive ECG Classification System.lnk")
.TargetPath = "C:\ecg-classification-system-pc\ecg-classification-system\LAUNCH_ECG_COMPREHENSIVE.bat"
.WorkingDirectory = "C:\ecg-classification-system-pc\ecg-classification-system"
.Description = "Comprehensive ECG Classification System - Advanced 30-Condition Cardiac Analysis"
.Save()
