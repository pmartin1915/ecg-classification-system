@echo off
REM ====================================================================
REM Professional ECG Classification System - Medical Education Platform
REM ====================================================================

title Professional ECG Classification System

REM Set environment to suppress matplotlib warnings
set MPLBACKEND=Agg
set PYTHONWARNINGS=ignore

REM Change to project directory
cd /d "%~dp0"

REM Clear screen for professional appearance
cls

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    PROFESSIONAL ECG CLASSIFICATION SYSTEM                   ║
echo ║                     Medical Education Training Platform                      ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo    30 Cardiac Conditions    Clinical Training    AI Explainability
echo    66,540 Medical Records   Batch Processing    Professional Grade
echo.
echo ══════════════════════════════════════════════════════════════════════════════
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

echo System Ready - Professional Medical Education Platform
echo.

REM Professional launch options
echo +------------------------------------------------------------------------------+
echo ^|                              LAUNCH OPTIONS                                  ^|
echo +------------------------------------------------------------------------------+
echo ^|  1. Launch Professional System (Recommended)                                ^|
echo ^|  2. Quick Demo Launch                                                       ^|
echo ^|  3. Background Dataset Loading                                              ^|
echo ^|  4. System Diagnostics                                                      ^|
echo ^|  5. Exit                                                                    ^|
echo +------------------------------------------------------------------------------+
echo.
set /p choice="Select option (1-5): "

if "%choice%"=="1" goto launch_professional
if "%choice%"=="2" goto launch_demo
if "%choice%"=="3" goto load_dataset
if "%choice%"=="4" goto diagnostics
if "%choice%"=="5" goto exit
goto invalid_choice

:launch_professional
echo.
echo +------------------------------------------------------------------------------+
echo ^|  Launching Professional ECG Classification System                          ^|
echo ^|                                                                             ^|
echo ^|  30-Condition Detection Active                                             ^|
echo ^|  Clinical Training Interface Ready                                         ^|
echo ^|  AI Explainability Enabled                                                 ^|
echo ^|  Batch Processing Available                                                ^|
echo ^|                                                                             ^|
echo ^|  Opening: http://localhost:8501                                            ^|
echo ^|  Press Ctrl+C to stop the system                                          ^|
echo +------------------------------------------------------------------------------+
echo.
timeout /t 2 >nul
start "" "http://localhost:8501"
streamlit run app/main.py --server.port=8501 --server.headless=false
goto end

:launch_demo
echo.
echo Launching Quick Demo System...
echo Opening: http://localhost:8502
echo.
start "" "http://localhost:8502"
streamlit run app/main.py --server.port=8502 --server.headless=false
goto end

:load_dataset
echo.
echo Starting Background Dataset Loading...
echo This will take 15-30 minutes for all 45,152 records
echo System will be available while processing
echo.
start "" python run_full_dataset_analysis.py
timeout /t 3 >nul
goto launch_professional

:diagnostics
echo.
echo +------------------------------------------------------------------------------+
echo ^|                            SYSTEM DIAGNOSTICS                               ^|
echo +------------------------------------------------------------------------------+
python -c "import sys; print('Python Version:', sys.version.split()[0])"
python -c "try: import streamlit; print('Streamlit:', streamlit.__version__); except: print('Streamlit: NOT INSTALLED')"
python -c "try: import wfdb; print('WFDB:', wfdb.__version__); except: print('WFDB: NOT INSTALLED')"
python -c "try: import numpy; print('NumPy:', numpy.__version__); except: print('NumPy: NOT INSTALLED')"
python -c "try: import pandas; print('Pandas:', pandas.__version__); except: print('Pandas: NOT INSTALLED')"
python -c "import os; print('Project Directory:', 'YES' if os.path.exists('app/main.py') else 'NO')"
python -c "import os; print('Data Directory:', 'YES' if os.path.exists('data') else 'NO')"
echo +------------------------------------------------------------------------------+
echo.
pause
goto start

:invalid_choice
echo.
echo Invalid choice. Please select 1-5.
timeout /t 2 >nul
goto start

:exit
echo.
echo Thank you for using Professional ECG Classification System!
echo Advancing Medical Education with AI Technology
echo.
timeout /t 2 >nul
exit /b 0

:end
echo.
echo System stopped. 
echo Run this launcher again anytime for professional ECG analysis
echo.
pause

:start
cls
goto :eof