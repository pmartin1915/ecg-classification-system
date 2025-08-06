@echo off
REM ====================================================================
REM ECG System - Professional Launcher (Fixed for Presentation)
REM ====================================================================

title ECG Heart Attack Detection System

REM Set environment
set PYTHONWARNINGS=ignore
set MPLBACKEND=Agg

REM Change to project directory
cd /d "%~dp0"

cls
echo.
echo ======================================================================
echo                    ECG HEART ATTACK DETECTION
echo                Professional Clinical AI System
echo ======================================================================
echo.
echo [STARTING] Initializing ECG classification system...
echo.

REM Try to find Python executable
set PYTHON_EXE=
if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" (
    set PYTHON_EXE="%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
) else if exist "C:\Python313\python.exe" (
    set PYTHON_EXE="C:\Python313\python.exe"
) else (
    set PYTHON_EXE=python
)

echo [PYTHON] Using: %PYTHON_EXE%

REM Test Python
%PYTHON_EXE% --version
if errorlevel 1 (
    echo [ERROR] Python not found! Please ensure Python 3.8+ is installed.
    pause
    exit /b 1
)

echo [OK] Python detected
echo [STARTING] Launching Streamlit server...
echo.
echo This will open in your default browser at: http://localhost:8501
echo.
echo If browser doesn't open automatically:
echo   1. Open your browser
echo   2. Go to: http://localhost:8501
echo   3. Wait 10-15 seconds for full startup
echo.
echo Press Ctrl+C to stop the system
echo.

REM Start Streamlit directly (simpler approach)
%PYTHON_EXE% -m streamlit run complete_user_friendly.py

echo.
echo [STOPPED] ECG system has been stopped.
pause