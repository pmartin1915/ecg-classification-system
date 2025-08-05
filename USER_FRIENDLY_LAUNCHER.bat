@echo off
REM ====================================================================
REM User-Friendly ECG Classification System - Intuitive Interface
REM Designed for non-experts with clear clinical workflow
REM ====================================================================

title ECG Heart Attack Detection - User-Friendly Interface

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
echo                   User-Friendly AI Interface
echo ======================================================================
echo.
echo    [HEART] Intuitive Design      [BRAIN] AI Explanations    [BOOK] Learn ECGs
echo    [CHECK] Clinical Workflow     [SPEED] 3-Second Analysis  [HELP] Step-by-Step
echo    [SMILE] Non-Expert Friendly   [SAFE] Educational Focus   [OK] Sample ECGs
echo.
echo ======================================================================
echo.

REM Quick Python check
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.11+
    echo    Download from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python detected - Starting user-friendly interface...
echo.
echo Features:
echo   - Welcome guide for new users
echo   - Step-by-step clinical workflow
echo   - Plain language explanations
echo   - Sample ECGs to try
echo   - Interactive onboarding
echo.
echo Opening: http://localhost:8503
echo Press Ctrl+C to stop the system
echo.

timeout /t 3 >nul
start "" "http://localhost:8503"
streamlit run app/user_friendly_main_fixed.py --server.port=8503 --server.headless=false

echo.
echo User-friendly interface stopped.
echo Run this launcher again anytime for intuitive ECG analysis!
echo.
pause