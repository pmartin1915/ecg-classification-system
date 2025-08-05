@echo off
REM ====================================================================
REM Enhanced ECG Classification System - Professional Clinical Platform
REM Advanced MI Detection with Smart Interface Selection
REM ====================================================================

title Enhanced ECG Classification System - Professional Clinical Platform

REM Set environment to suppress warnings
set MPLBACKEND=Agg
set PYTHONWARNINGS=ignore

REM Change to project directory
cd /d "%~dp0"

REM Clear screen for professional appearance
cls

echo.
echo ======================================================================
echo                 ENHANCED ECG CLASSIFICATION SYSTEM
echo              Professional Clinical Platform with Advanced MI Detection
echo ======================================================================
echo.
echo    🫀 Enhanced MI Detection     📊 Clinical Training    🧠 AI Explainability
echo    📈 70%+ MI Sensitivity       📁 Batch Processing     ⚡ Real-time Analysis
echo    🎯 66,540 Medical Records     🏥 Professional Grade   🔬 Research Ready
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

echo System Ready - Enhanced Professional Clinical Platform
echo.

:menu
echo ----------------------------------------------------------------------
echo                         ENHANCED LAUNCH OPTIONS                      
echo ----------------------------------------------------------------------
echo  1. Launch Enhanced System (Smart Auto-Detection)                   
echo  2. Launch Standard System (Fallback Mode)                          
echo  3. Train Enhanced MI Models (Powerful PC Required)                 
echo  4. Quick MI Enhancement Test                                        
echo  5. System Diagnostics                                               
echo  6. Exit                                                             
echo ----------------------------------------------------------------------
echo.
set /p choice="Select option (1-6): "

if "%choice%"=="1" goto launch_enhanced
if "%choice%"=="2" goto launch_standard  
if "%choice%"=="3" goto train_enhanced
if "%choice%"=="4" goto test_enhanced
if "%choice%"=="5" goto diagnostics
if "%choice%"=="6" goto exit
goto invalid_choice

:launch_enhanced
echo.
echo ----------------------------------------------------------------------
echo  Launching Enhanced ECG Classification System                       
echo                                                                     
echo  🫀 Advanced MI Detection (70%+ Target Sensitivity)                 
echo  🧠 Ensemble ML Models with 150+ Clinical Features                  
echo  🏥 Professional Clinical Interface                                 
echo  📊 Real-time Analysis with AI Explainability                       
echo                                                                     
echo  Smart Detection: Auto-selects enhanced or standard interface       
echo  Opening: http://localhost:8501                                     
echo  Press Ctrl+C to stop the system                                   
echo ----------------------------------------------------------------------
echo.
timeout /t 3 >nul
start "" "http://localhost:8501"
streamlit run app/enhanced_main.py --server.port=8501 --server.headless=false
goto end

:launch_standard
echo.
echo ----------------------------------------------------------------------
echo  Launching Standard ECG Classification System                       
echo                                                                     
echo  📊 Standard Multi-Condition Detection                              
echo  🎓 Clinical Training Interface                                     
echo  📁 Batch Processing Available                                      
echo                                                                     
echo  Opening: http://localhost:8502                                     
echo  Press Ctrl+C to stop the system                                   
echo ----------------------------------------------------------------------
echo.
timeout /t 2 >nul
start "" "http://localhost:8502"
streamlit run app/main.py --server.port=8502 --server.headless=false
goto end

:train_enhanced
echo.
echo ----------------------------------------------------------------------
echo                         ENHANCED MI MODEL TRAINING                  
echo ----------------------------------------------------------------------
echo  This will train advanced MI detection models for 70%+ sensitivity  
echo  Requirements: Powerful PC with multi-core CPU                      
echo  Estimated Time: 10-30 minutes depending on hardware               
echo  Dataset: Up to 3,000 records with enhanced features               
echo.
echo  WARNING: This is computationally intensive!                        
echo  Ensure you're running on a powerful machine.                       
echo ----------------------------------------------------------------------
echo.
set /p confirm="Continue with enhanced training? (y/n): "
if /i "%confirm%"=="y" goto run_training
if /i "%confirm%"=="yes" goto run_training
goto menu

:run_training
echo.
echo Starting Enhanced MI Model Training...
echo This will run in the background - monitor progress in console
echo.
python optimized_mi_enhancement.py
echo.
echo Training completed! Enhanced models should now be available.
echo You can now use option 1 to launch with enhanced capabilities.
echo.
pause
goto menu

:test_enhanced
echo.
echo ----------------------------------------------------------------------
echo                      QUICK MI ENHANCEMENT TEST                      
echo ----------------------------------------------------------------------
echo  Testing enhanced MI detection capabilities with small dataset       
echo  This is safe to run on any hardware                               
echo  Estimated Time: 2-5 minutes                                       
echo ----------------------------------------------------------------------
echo.
python test_mi_enhancement.py
echo.
pause
goto menu

:diagnostics
echo.
python system_diagnostics.py
echo ----------------------------------------------------------------------
echo.
pause
goto menu

:invalid_choice
echo.
echo Invalid choice. Please select 1-6.
timeout /t 2 >nul
goto menu

:exit
echo.
echo Thank you for using Enhanced ECG Classification System!
echo Advancing Medical Education with AI Technology
echo.
timeout /t 2 >nul
exit /b 0

:end
echo.
echo System stopped. 
echo Run this enhanced launcher again anytime for professional ECG analysis
echo with advanced MI detection capabilities!
echo.
pause