@echo off
REM ====================================================================
REM Desktop Shortcut Creator for Professional ECG Classification System
REM ====================================================================

title Install Desktop Shortcut

echo.
echo ======================================================================
echo          PROFESSIONAL ECG CLASSIFICATION SYSTEM
echo                Desktop Shortcut Installation
echo ======================================================================
echo.

echo Creating professional desktop shortcut...
echo.

REM Run the VBS script to create shortcut
cscript //nologo "Create_Desktop_Shortcut.vbs"

echo.
echo ======================================================================
echo                        INSTALLATION COMPLETE
echo ======================================================================
echo.
echo Your Professional ECG Classification System is now available on
echo your desktop with a professional medical application icon.
echo.
echo Desktop Icon Name: "Professional ECG Classification System"
echo.
echo Double-click the desktop icon to launch your medical training platform!
echo.
echo ======================================================================
echo.

pause