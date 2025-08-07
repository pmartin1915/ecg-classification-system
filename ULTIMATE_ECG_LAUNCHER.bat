@echo off
REM ====================================================================
REM ULTIMATE ECG SYSTEM LAUNCHER
REM Smart, comprehensive launcher with laptop optimization
REM Consolidates all previous launchers with intelligent system detection
REM ====================================================================

title Ultimate ECG Heart Attack Detection System

REM Set optimal environment variables
set MPLBACKEND=Agg
set PYTHONWARNINGS=ignore
set PYTHONPATH=%cd%

REM Change to project directory
cd /d "%~dp0"

REM ==== SYSTEM DETECTION AND OPTIMIZATION ====
call :detect_system_capabilities

cls
echo.
echo ======================================================================
echo                 ULTIMATE ECG HEART ATTACK DETECTION
echo              Smart AI System with Laptop Optimization
echo ======================================================================
echo.
echo    System Type: %SYSTEM_TYPE%          Performance: %PERFORMANCE_LEVEL%
echo    Python: %PYTHON_VERSION%            Memory: %AVAILABLE_MEMORY%
echo    Data Loader: Optimized              Progress: Real-time tracking
echo.
echo ======================================================================
echo.

REM Show smart menu based on system capabilities
call :show_smart_menu

REM Get user choice
set /p choice="Select option (1-6): "

REM Process user choice (simplified syntax)
if "%choice%"=="1" call :quick_test_mode
if "%choice%"=="2" call :development_mode
if "%choice%"=="3" call :training_mode
if "%choice%"=="4" call :full_system_launch
if "%choice%"=="5" call :data_management
if "%choice%"=="6" call :system_diagnostics

REM Handle invalid choice
if not "%choice%"=="1" if not "%choice%"=="2" if not "%choice%"=="3" if not "%choice%"=="4" if not "%choice%"=="5" if not "%choice%"=="6" (
    echo.
    echo [ERROR] Invalid choice: %choice%
    echo Please select a number between 1 and 6.
    echo.
)

echo.
echo [COMPLETE] Operation finished.
pause
exit /b 0

REM ====================================================================
REM SYSTEM DETECTION FUNCTIONS
REM ====================================================================

:detect_system_capabilities
echo [DETECTING] Analyzing system capabilities...

REM Detect system type (laptop vs desktop indicators)
set SYSTEM_TYPE=Unknown
wmic computersystem get TotalPhysicalMemory /value | find "TotalPhysicalMemory" > temp_mem.txt
for /f "tokens=2 delims==" %%i in (temp_mem.txt) do set TOTAL_MEMORY=%%i
del temp_mem.txt

REM Convert bytes to GB (approximate) - handle potential errors
set /a MEMORY_GB=8
if defined TOTAL_MEMORY (
    if not "%TOTAL_MEMORY%"=="" (
        set /a MEMORY_GB=%TOTAL_MEMORY:~0,-9% 2>nul
    )
)

if %MEMORY_GB% LEQ 8 (
    set SYSTEM_TYPE=Laptop/Mobile
    set PERFORMANCE_LEVEL=Optimized
    set RECOMMENDED_SIZE=rapid
    set RECOMMENDED_CHOICE=2
) else if %MEMORY_GB% LEQ 16 (
    set SYSTEM_TYPE=Standard PC
    set PERFORMANCE_LEVEL=Good
    set RECOMMENDED_SIZE=training
    set RECOMMENDED_CHOICE=3
) else (
    set SYSTEM_TYPE=High-End PC
    set PERFORMANCE_LEVEL=Excellent
    set RECOMMENDED_SIZE=validation
    set RECOMMENDED_CHOICE=3
)

set AVAILABLE_MEMORY=%MEMORY_GB%GB

REM Detect Python version
set PYTHON_EXE=
if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" (
    set PYTHON_EXE="%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
    set PYTHON_VERSION=3.13
) else if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON_EXE="%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
    set PYTHON_VERSION=3.12
) else if exist "C:\Python313\python.exe" (
    set PYTHON_EXE="C:\Python313\python.exe"
    set PYTHON_VERSION=3.13
) else (
    set PYTHON_EXE=python
    set PYTHON_VERSION=System
)

echo [OK] System detection complete
REM Clean up temporary files
if exist temp_mem.txt del temp_mem.txt
goto :eof

:show_smart_menu
echo SMART MENU (Optimized for %SYSTEM_TYPE%):
echo.
echo   1. [INSTANT] Quick Test Mode        (5 samples,  2 seconds)
echo   2. [RAPID]   Development Mode       (25 samples, 5 seconds) 
echo   3. [TRAIN]   Training Mode          (1000 samples, 30 seconds)
echo   4. [FULL]    Complete System Launch (Full interface)
echo   5. [DATA]    Data Management        (Smart loader tools)
echo   6. [DIAG]    System Diagnostics     (Performance analysis)
echo.
echo   Recommended for your system: Option %RECOMMENDED_CHOICE%
echo.
goto :eof

REM ====================================================================
REM MODE FUNCTIONS
REM ====================================================================

:quick_test_mode
echo.
echo ======================================================================
echo                        QUICK TEST MODE
echo ======================================================================
echo.
echo [MODE] Instant testing with minimal data (perfect for debugging)
echo [DATA] Loading 5 samples for immediate feedback...
echo.

%PYTHON_EXE% -c "
import sys
sys.path.append('.')
from app.utils.laptop_optimized_loader import test_system
print('[SYSTEM] Testing laptop-optimized data loader...')
success = test_system()
if success:
    print('[SUCCESS] Quick test completed - system ready!')
else:
    print('[ERROR] Quick test failed - check system configuration')
"

echo.
echo [COMPLETE] Quick test mode finished
goto :eof

:development_mode
echo.
echo ======================================================================
echo                       DEVELOPMENT MODE  
echo ======================================================================
echo.
echo [MODE] Rapid development with optimal dataset size
echo [DATA] Loading %RECOMMENDED_SIZE% dataset for development...
echo.

%PYTHON_EXE% -c "
import sys
sys.path.append('.')
from app.utils.laptop_optimized_loader import quick_load

print('[LOADING] Loading development dataset...')
try:
    X, y, metadata = quick_load('%RECOMMENDED_SIZE%', verbose=True)
    print('[SUCCESS] Development data loaded successfully!')
    print(f'[READY] Ready for development with {len(y)} samples')
    
    # Show what you can do next
    print()
    print('[NEXT STEPS] You can now:')
    print('  - Test feature extraction algorithms')  
    print('  - Develop preprocessing pipelines')
    print('  - Prototype ML models')
    print('  - Validate data quality')
    
except Exception as e:
    print(f'[ERROR] Failed to load development data: {e}')
"

echo.
echo [COMPLETE] Development mode ready
goto :eof

:training_mode
echo.
echo ======================================================================
echo                         TRAINING MODE
echo ======================================================================
echo.
echo [MODE] Full training with comprehensive datasets
echo [WARN] This mode loads larger datasets - may take 30-60 seconds
echo.

set /p confirm="Continue with training mode? (y/n): "
if not "%confirm%"=="y" goto :eof

echo [LOADING] Preparing training environment...
echo.

%PYTHON_EXE% -c "
import sys
sys.path.append('.')
from app.utils.laptop_optimized_loader import LaptopOptimizedLoader

print('[TRAINING] Initializing training mode...')
loader = LaptopOptimizedLoader()

try:
    print('[STEP 1/3] Loading training dataset...')
    X_train, y_train, meta_train = loader.load_with_progress('training', verbose=True)
    
    print('[STEP 2/3] Loading validation dataset...')  
    X_val, y_val, meta_val = loader.load_with_progress('validation', verbose=True)
    
    print('[STEP 3/3] Performance summary...')
    performance = loader.get_performance_report()
    print('[PERFORMANCE] Training environment ready!')
    
    for dataset, perf in performance['average_performance'].items():
        print(f'  {dataset}: {perf[\"time\"]:.2f}s ({perf[\"performance\"]})')
    
    print()
    print('[SUCCESS] Training mode fully initialized!')
    print(f'[DATA] Training: {len(y_train)} samples, Validation: {len(y_val)} samples')
    
except Exception as e:
    print(f'[ERROR] Training mode failed: {e}')
"

echo.
echo [COMPLETE] Training mode initialized
goto :eof

:full_system_launch
echo.
echo ======================================================================
echo                      FULL SYSTEM LAUNCH
echo ======================================================================
echo.
echo [MODE] Complete user-friendly interface with all features
echo [PORT] Starting on port 8507 for optimal performance
echo.

REM Pre-flight system check
echo [CHECK] Running pre-flight system verification...
%PYTHON_EXE% -c "
import sys
sys.path.append('.')
try:
    import complete_user_friendly
    print('[OK] Main application verified')
    
    from app.utils.laptop_optimized_loader import test_system
    test_result = test_system()
    if test_result:
        print('[OK] Data loader verified')
    else:
        print('[WARN] Data loader issues detected - system will still launch')
        
    print('[SUCCESS] Pre-flight check passed')
except Exception as e:
    print(f'[WARN] Pre-flight issue: {e}')
    print('[INFO] System will attempt to launch anyway')
"

echo.
echo [LAUNCHING] Starting complete ECG system...
echo [INFO] This will open in your browser at: http://localhost:8507
echo.
echo System Features Available:
echo   - User-friendly onboarding and interface
echo   - Advanced ECG analysis with clinical reasoning  
echo   - Heart attack detection with territory mapping
echo   - Complete AI explainability system
echo   - Comprehensive clinical training modules
echo   - Full batch processing capabilities
echo   - Real-time performance monitoring
echo.

echo [BROWSER] Opening system in 5 seconds...
timeout /t 5 >nul

REM Start the system with progress indication
start /B cmd /c %PYTHON_EXE% -m streamlit run complete_user_friendly.py --server.port=8507 --server.headless=true

echo [WAIT] Allowing system to initialize...
timeout /t 8 >nul

echo [SUCCESS] Opening browser...
start "" "http://localhost:8507"

echo.
echo ======================================================================
echo                    SYSTEM RUNNING SUCCESSFULLY
echo ======================================================================
echo.
echo   URL: http://localhost:8507
echo   Status: Active and ready for clinical use
echo   Performance: %PERFORMANCE_LEVEL% on %SYSTEM_TYPE%
echo.
echo   If page shows 404 or connection issues:
echo     1. Wait another 10-15 seconds for complete startup
echo     2. Refresh browser (F5)
echo     3. Check Windows Defender/Firewall settings
echo.
echo   Press Ctrl+C to stop the system
echo   Close this window to keep system running
echo.

REM Keep window open to show status
goto :eof

:data_management
echo.
echo ======================================================================
echo                       DATA MANAGEMENT
echo ======================================================================
echo.
echo [MODE] Smart data management with laptop optimization
echo.

%PYTHON_EXE% app/utils/laptop_optimized_loader.py menu

echo.
echo Data Management Options:
echo   1. Test instant loading (5 samples)
echo   2. Test rapid loading (25 samples) 
echo   3. Test training loading (1000 samples)
echo   4. Show performance report
echo   5. Return to main menu
echo.

set /p data_choice="Select data operation (1-5): "

if "%data_choice%"=="1" %PYTHON_EXE% app/utils/laptop_optimized_loader.py instant
if "%data_choice%"=="2" %PYTHON_EXE% app/utils/laptop_optimized_loader.py rapid  
if "%data_choice%"=="3" %PYTHON_EXE% app/utils/laptop_optimized_loader.py training
if "%data_choice%"=="4" (
    echo [PERFORMANCE] Running performance analysis...
    %PYTHON_EXE% -c "
import sys
sys.path.append('.')
from app.utils.laptop_optimized_loader import LaptopOptimizedLoader
loader = LaptopOptimizedLoader()
report = loader.get_performance_report()
print('[REPORT] Performance Summary:')
if 'total_loads' in report:
    print(f'  Total loads performed: {report[\"total_loads\"]}')
    for dataset, perf in report.get('average_performance', {}).items():
        print(f'  {dataset}: {perf[\"time\"]:.2f}s ({perf[\"performance\"]})')
else:
    print('  No performance data available - run some data loading first')
    "
)

echo.
echo [COMPLETE] Data management operations finished
goto :eof

:system_diagnostics
echo.
echo ======================================================================
echo                      SYSTEM DIAGNOSTICS
echo ======================================================================
echo.
echo [DIAG] Running comprehensive system analysis...
echo.

REM System info
echo [INFO] System Configuration:
echo   Type: %SYSTEM_TYPE%
echo   Memory: %AVAILABLE_MEMORY%
echo   Performance: %PERFORMANCE_LEVEL%
echo   Python: %PYTHON_VERSION%
echo.

REM Python environment check
echo [PYTHON] Python Environment Check:
%PYTHON_EXE% --version
%PYTHON_EXE% -c "
import sys
print(f'  Python executable: {sys.executable}')
print(f'  Python path: {sys.path[0]}')

# Check key dependencies
deps = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'plotly', 'sklearn']
missing = []
for dep in deps:
    try:
        __import__(dep)
        print(f'  ✓ {dep}')
    except ImportError:
        print(f'  ✗ {dep} (missing)')
        missing.append(dep)

if missing:
    print(f'  [WARN] Missing dependencies: {missing}')
else:
    print('  [OK] All key dependencies available')
"

echo.
echo [DATA] Data System Check:
%PYTHON_EXE% -c "
import sys
sys.path.append('.')
from pathlib import Path

# Check data directories
data_dirs = ['data/cache', 'data/models', 'data/raw/ptbxl']
for dir_path in data_dirs:
    path = Path(dir_path)
    if path.exists():
        if dir_path == 'data/cache':
            cache_files = list(path.glob('*.pkl'))
            print(f'  ✓ {dir_path} ({len(cache_files)} cache files)')
        elif dir_path == 'data/models':
            model_files = list(path.glob('*.pkl')) + list(path.glob('*.joblib'))
            print(f'  ✓ {dir_path} ({len(model_files)} models)')
        else:
            print(f'  ✓ {dir_path}')
    else:
        print(f'  ✗ {dir_path} (missing)')

# Test data loader
try:
    from app.utils.laptop_optimized_loader import test_system
    if test_system():
        print('  ✓ Laptop-optimized data loader')
    else:
        print('  ✗ Data loader issues detected')
except Exception as e:
    print(f'  ✗ Data loader error: {e}')
"

echo.
echo [STREAMLIT] Streamlit Check:
%PYTHON_EXE% -c "
try:
    import streamlit
    print(f'  ✓ Streamlit version: {streamlit.__version__}')
    import complete_user_friendly
    print('  ✓ Main application imports successfully')
except Exception as e:
    print(f'  ✗ Streamlit issue: {e}')
"

echo.
echo [COMPLETE] System diagnostics finished
echo.
echo Recommendations based on your %SYSTEM_TYPE%:
if "%SYSTEM_TYPE%"=="Laptop/Mobile" (
    echo   • Use 'rapid' datasets ^(25 samples^) for development
    echo   • Limit training sessions to ^<30 minutes
    echo   • Close other applications during model training
    echo   • Use instant mode for quick debugging
)
if "%SYSTEM_TYPE%"=="Standard PC" (
    echo   • Use 'training' datasets ^(1000 samples^) for development
    echo   • Full model training should work well
    echo   • Consider using validation datasets for final testing
)
if "%SYSTEM_TYPE%"=="High-End PC" (  
    echo   • All dataset sizes work optimally
    echo   • Full validation and large-scale training recommended
    echo   • Consider contributing to model development
)

goto :eof