@echo off
REM ====================================================================
REM Complete User-Friendly ECG System - All Advanced Features
REM Professional clinical interface with comprehensive functionality
REM ====================================================================

title ECG Heart Attack Detection - Complete User-Friendly System

REM Set environment to suppress warnings
set MPLBACKEND=Agg
set PYTHONWARNINGS=ignore

REM Change to project directory
cd /d "%~dp0"

REM Clear screen for professional appearance
cls

echo.
echo ======================================================================
echo                    ECG HEART ATTACK DETECTION
echo                Complete User-Friendly AI System
echo ======================================================================
echo.
echo    [HEART] All Features        [BRAIN] AI Explainability  [BOOK] Clinical Training
echo    [CHECK] Batch Processing    [SPEED] Performance Monitor [HELP] User-Friendly
echo    [SMILE] Complete System     [SAFE] Educational Focus   [OK] Advanced Features
echo.
echo ======================================================================
echo.

REM Use Python 3.13 explicitly to avoid version conflicts
set PYTHON_EXE="%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"

REM Check if Python 3.13 is available
%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.13 not found! 
    echo Expected location: %PYTHON_EXE%
    echo Please install Python 3.13+ from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python 3.13 detected - Starting complete user-friendly system...
echo.
echo Complete Feature Set:
echo   - User-friendly onboarding and interface
echo   - Advanced ECG analysis with clinical reasoning
echo   - Heart attack detection with territory mapping
echo   - Complete AI explainability system
echo   - Comprehensive clinical training modules
echo   - Full batch processing capabilities
echo   - Real-time performance monitoring
echo   - Educational content for all experience levels
echo.
echo Starting Streamlit server...
echo This may take 10-15 seconds to fully initialize
echo.

REM Start Streamlit in background
start /B cmd /c %PYTHON_EXE% -m streamlit run complete_user_friendly.py --server.port=8507 --server.headless=true

echo Waiting for server to start up...
timeout /t 8 >nul

echo.
echo Opening: http://localhost:8507
echo.
echo If the page shows 404 or "site can't be reached":
echo   1. Wait another 10-15 seconds for full startup
echo   2. Refresh the browser page (F5)
echo   3. Check that no antivirus is blocking the connection
echo.
echo Press Ctrl+C to stop the system
echo.

start "" "http://localhost:8507"

REM Keep the window open
pause

echo.
echo Complete user-friendly system stopped.
echo This version combines advanced AI capabilities with intuitive design!
echo.
pause