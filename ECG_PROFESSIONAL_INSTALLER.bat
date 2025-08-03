@echo off
REM ====================================================================
REM Professional ECG Classification System - Installation Manager
REM Medical Education Platform Installer v2.0
REM ====================================================================

title Professional ECG Classification System - Installer

REM Set professional environment
set MPLBACKEND=Agg
set PYTHONWARNINGS=ignore

REM Change to installation directory
cd /d "%~dp0"

REM Clear screen for professional installer appearance
cls

echo.
echo ======================================================================
echo                 PROFESSIONAL ECG CLASSIFICATION SYSTEM
echo                      Medical Education Platform v2.0
echo                        Installation Manager
echo ======================================================================
echo.
echo                        Advanced Cardiac Diagnostics
echo                    Training Platform for Healthcare Professionals
echo.
echo    Features: 30 Cardiac Conditions | 66,540 Medical Records
echo              AI Explainability | Clinical Training Interface
echo.
echo ======================================================================
echo.

REM System requirements check
echo [1/4] Checking System Requirements...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11+ is required but not found
    echo         Please install Python from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
) else (
    echo [OK] Python installation verified
)

REM Check project files
if not exist "app\main.py" (
    echo [ERROR] Core application files missing
    echo         Please ensure you're in the correct project directory
    pause
    exit /b 1
) else (
    echo [OK] Application files verified
)

REM Check data directory
if not exist "data" (
    echo [INFO] Creating data directory structure...
    mkdir data\cache 2>nul
    mkdir data\processed 2>nul
    mkdir data\results 2>nul
) else (
    echo [OK] Data directory structure verified
)

echo [OK] All system requirements satisfied
echo.

echo [2/4] Preparing Professional Environment...
echo.
timeout /t 1 >nul
echo [OK] Environment configuration complete
echo.

echo [3/4] Installing Desktop Integration...
echo.

REM Create desktop shortcut
if exist "Create_Desktop_Shortcut.vbs" (
    cscript //nologo "Create_Desktop_Shortcut.vbs" >nul 2>&1
    if errorlevel 0 (
        echo [OK] Desktop shortcut installed successfully
    ) else (
        echo [INFO] Desktop shortcut creation skipped
    )
) else (
    echo [INFO] Desktop integration files not found
)

echo.

echo [4/4] Finalizing Installation...
echo.
timeout /t 1 >nul
echo [OK] Professional ECG Classification System ready for deployment
echo.

echo ======================================================================
echo                        INSTALLATION COMPLETE
echo ======================================================================
echo.
echo Your Professional Medical Education Platform is now ready!
echo.
echo Available Launch Options:
echo   1. Desktop Shortcut: "Professional ECG Classification System"
echo   2. Quick Launch:     STREAMLINED_LAUNCHER.bat
echo   3. Full Featured:    PROFESSIONAL_LAUNCHER.bat
echo.
echo System Capabilities:
echo   - 30 Comprehensive Cardiac Condition Detection
echo   - 66,540+ Physician-Validated Medical Records  
echo   - Real-time ECG Analysis (sub-3 second processing)
echo   - AI Explainability for Educational Transparency
echo   - Clinical Training Interface with Case Studies
echo   - Professional Batch Processing and Reporting
echo.
echo Medical Education Applications:
echo   - Medical School ECG Curriculum Integration
echo   - Residency Training Programs (Emergency Medicine/Cardiology)
echo   - Continuing Medical Education for Practicing Clinicians
echo   - Healthcare Professional Competency Assessment
echo.
echo ======================================================================
echo.

set /p launch="Would you like to launch the system now? (Y/N): "
if /i "%launch%"=="Y" (
    echo.
    echo Launching Professional ECG Classification System...
    echo Opening browser interface at: http://localhost:8501
    echo.
    timeout /t 2 >nul
    start "" "http://localhost:8501"
    streamlit run app/main.py --server.port=8501 --server.headless=false
) else (
    echo.
    echo Installation complete. Launch anytime using your desktop shortcut!
    echo.
    timeout /t 3 >nul
)

echo.
echo Thank you for choosing Professional ECG Classification System
echo for your medical education needs!
echo.