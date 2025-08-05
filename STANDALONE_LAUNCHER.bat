@echo off
REM ====================================================================
REM Standalone User-Friendly ECG System - No Import Dependencies
REM Guaranteed to work without any existing app structure issues
REM ====================================================================

title ECG Heart Attack Detection - Standalone User-Friendly Interface

REM Set environment to suppress warnings
set MPLBACKEND=Agg
set PYTHONWARNINGS=ignore

REM Change to project directory
cd /d "%~dp0"

REM Clear screen
cls

echo.
echo ======================================================================
echo                    ECG HEART ATTACK DETECTION
echo                   Standalone User-Friendly Interface
echo ======================================================================
echo.
echo    [HEART] No Import Issues     [BRAIN] AI Explanations    [BOOK] Learn ECGs
echo    [CHECK] Clinical Workflow    [SPEED] 3-Second Analysis  [HELP] Step-by-Step
echo    [SMILE] Beginner Friendly    [SAFE] Educational Focus   [OK] Sample ECGs
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

echo [OK] Python 3.13 detected - Starting standalone interface...
echo.
echo This version is completely isolated and guaranteed to work!
echo.
echo Features:
echo   - Welcome guide for new users
echo   - Step-by-step clinical workflow  
echo   - Plain language explanations
echo   - Interactive sample ECGs
echo   - No dependency issues
echo.
echo Opening: http://localhost:8509
echo Press Ctrl+C to stop the system
echo.

timeout /t 3 >nul
start "" "http://localhost:8509"
%PYTHON_EXE% -m streamlit run standalone_user_friendly.py --server.port=8509 --server.headless=false

echo.
echo Standalone interface stopped.
echo This version works independently of any other app components!
echo.
pause