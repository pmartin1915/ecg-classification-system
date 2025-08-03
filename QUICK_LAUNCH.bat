@echo off
REM Quick Launch - Bypass dependency checks
title Comprehensive ECG Classification System - Quick Launch

echo.
echo ======================================================================
echo    COMPREHENSIVE ECG CLASSIFICATION SYSTEM - QUICK LAUNCH
echo    Advanced Cardiac Analysis - 30 Conditions Detection
echo ======================================================================
echo.

cd /d "%~dp0"

echo [INFO] Quick launching Comprehensive ECG System...
echo [INFO] Opening browser at: http://localhost:8501
echo [INFO] Press Ctrl+C to stop the system
echo.

REM Launch browser first
start "" "http://localhost:8501"

REM Start Streamlit directly
streamlit run app/main.py --server.port=8501

pause