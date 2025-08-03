@echo off
REM ====================================================================
REM Comprehensive ECG Classification System Launcher
REM Professional cardiac analysis with 30-condition detection
REM ====================================================================

title Comprehensive ECG Classification System

echo.
echo ======================================================================
echo    COMPREHENSIVE ECG CLASSIFICATION SYSTEM
echo    Advanced Cardiac Analysis - 30 Conditions Detection
echo ======================================================================
echo.

REM Change to project directory
cd /d "%~dp0"

echo [INFO] Starting Comprehensive ECG System...
echo [INFO] Project Directory: %CD%
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.11 or higher.
    echo [INFO] Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Python found: 
python --version

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] No virtual environment found. Using system Python.
)

REM Check required packages
echo [INFO] Checking dependencies...
python -c "import streamlit, wfdb, numpy, pandas" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
)

echo [INFO] Dependencies verified!
echo.

REM Display system capabilities
echo ======================================================================
echo    SYSTEM CAPABILITIES
echo ======================================================================
python -c "
from config.settings import TARGET_CONDITIONS, CLINICAL_PRIORITY
print(f'• Total Conditions: {len(TARGET_CONDITIONS)}')
print(f'• Clinical Priority Levels: {len(CLINICAL_PRIORITY)}')
print(f'• PTB-XL Dataset: 21,388 records')
print(f'• ECG Arrhythmia Dataset: 45,152 records')
print(f'• WFDB Format Support: YES')
print(f'• Real-time Analysis: YES')
"
echo.

REM Launch options menu
echo ======================================================================
echo    LAUNCH OPTIONS
echo ======================================================================
echo    1. Launch Full Comprehensive System (Recommended)
echo    2. Launch with Sample Data (Quick Start)
echo    3. Load Full Dataset First (Background Process)
echo    4. System Diagnostics
echo    5. Exit
echo.
set /p choice="Select option (1-5): "

if "%choice%"=="1" goto launch_full
if "%choice%"=="2" goto launch_quick
if "%choice%"=="3" goto load_dataset
if "%choice%"=="4" goto diagnostics
if "%choice%"=="5" goto exit
goto invalid_choice

:launch_full
echo.
echo [INFO] Launching Comprehensive ECG Classification System...
echo [INFO] Opening browser at: http://localhost:8501
echo [INFO] Press Ctrl+C to stop the system
echo.
start "" "http://localhost:8501"
streamlit run app/main.py --server.port=8501
goto end

:launch_quick
echo.
echo [INFO] Quick launch with existing processed data...
echo [INFO] Opening browser at: http://localhost:8502
echo.
start "" "http://localhost:8502"
streamlit run app/main.py --server.port=8502
goto end

:load_dataset
echo.
echo [INFO] Loading full dataset in background...
echo [INFO] This will take 15-30 minutes for all 45,152 records
echo [INFO] You can use the system while this processes
echo.
start "" python run_full_dataset_analysis.py
timeout /t 3 >nul
goto launch_full

:diagnostics
echo.
echo ======================================================================
echo    SYSTEM DIAGNOSTICS
echo ======================================================================
python -c "
import sys
print(f'Python Version: {sys.version}')
try:
    import streamlit; print(f'Streamlit: {streamlit.__version__}')
except: print('Streamlit: NOT INSTALLED')
try:
    import wfdb; print(f'WFDB: {wfdb.__version__}')
except: print('WFDB: NOT INSTALLED')
try:
    import numpy; print(f'NumPy: {numpy.__version__}')
except: print('NumPy: NOT INSTALLED')
try:
    import pandas; print(f'Pandas: {pandas.__version__}')
except: print('Pandas: NOT INSTALLED')

import os
print(f'Working Directory: {os.getcwd()}')
print(f'Project Files Present: {\"YES\" if os.path.exists(\"app/main.py\") else \"NO\"}')
print(f'Data Directory: {\"YES\" if os.path.exists(\"data\") else \"NO\"}')
print(f'Config Files: {\"YES\" if os.path.exists(\"config/settings.py\") else \"NO\"}')
"
echo.
pause
goto main_menu

:invalid_choice
echo [ERROR] Invalid choice. Please select 1-5.
timeout /t 2 >nul
goto main_menu

:main_menu
cls
goto start

:exit
echo.
echo [INFO] Thank you for using Comprehensive ECG Classification System!
timeout /t 2 >nul
exit /b 0

:end
echo.
echo [INFO] System stopped. Window will close in 5 seconds...
timeout /t 5 >nul

:start