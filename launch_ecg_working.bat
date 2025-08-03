@echo off
title ECG Classification System
echo ================================================================================
echo    ECG CLASSIFICATION SYSTEM - CLINICAL AI PLATFORM
echo ================================================================================
echo    Enhanced MI Detection for Healthcare Professionals
echo    Real Medical Data ^| Professional Interface ^| Clinical Impact
echo ================================================================================
echo.
echo Starting your ECG Classification System...
echo.

cd /d "C:\ecg-classification-system-pc\ecg-classification-system"

echo Checking system status...
python clean_status_check.py

echo.
echo ================================================================================
echo    LAUNCHING CLINICAL WEB INTERFACE
echo ================================================================================
echo    Your professional ECG classification system is starting...
echo    The web interface will open automatically in your browser.
echo    
echo    Key Features Available:
echo    - Upload ECG files for instant analysis
echo    - Real-time heart attack detection
echo    - Professional clinical reports
echo    - 5-class cardiac condition classification
echo.
echo    Press Ctrl+C anytime to close the system
echo ================================================================================
echo.

streamlit run app/simple_main.py

pause