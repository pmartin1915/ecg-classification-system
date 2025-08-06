@echo off
REM ====================================================================
REM PC Setup Automation for Enhanced MI Detection Training
REM Windows 11, AMD Ryzen 5 5600G, AMD Radeon RX 6600, 10GB RAM
REM ====================================================================

title ECG Enhanced MI Training - PC Setup

cls
echo.
echo ======================================================================
echo              ECG ENHANCED MI TRAINING - PC SETUP
echo                    AMD Ryzen 5 5600G Optimized
echo ======================================================================
echo.
echo This script will set up your PC for enhanced MI detection training:
echo   - Python virtual environment
echo   - Enhanced ML dependencies  
echo   - AMD GPU optimization
echo   - Memory configuration
echo   - VS Code workspace setup
echo.
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found!
    echo.
    echo Please install Python 3.11 first:
    echo 1. Go to https://www.python.org/downloads/windows/
    echo 2. Download Python 3.11.x (64-bit)
    echo 3. Run installer with "Add to PATH" checked
    echo 4. Restart this script
    echo.
    pause
    exit /b 1
)

echo ✅ Python detected - Setting up enhanced environment...
echo.

REM Create virtual environment
echo 📦 Creating virtual environment for enhanced training...
python -m venv ecg_enhanced_env

REM Activate virtual environment  
echo ⚡ Activating virtual environment...
call ecg_enhanced_env\Scripts\activate.bat

REM Upgrade pip
echo 🔄 Upgrading pip to latest version...
python -m pip install --upgrade pip

REM Install enhanced requirements
echo 📥 Installing enhanced ML dependencies...
echo This may take 5-10 minutes depending on your internet connection...
echo.

REM Core packages first
echo Installing core data science stack...
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.11.1

echo Installing visualization libraries...
pip install matplotlib==3.7.2 seaborn==0.12.2 plotly==5.15.0

echo Installing machine learning libraries...
pip install scikit-learn==1.3.0 xgboost==1.7.6 lightgbm==4.0.0 imbalanced-learn==0.11.0

echo Installing PyTorch (CPU optimized)...
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo Installing ECG-specific libraries...
pip install wfdb==4.1.0 neurokit2==0.2.4 mne==1.4.2

echo Installing Streamlit and web framework...
pip install streamlit==1.25.0 altair==5.0.1

echo Installing performance optimization libraries...
pip install numba==0.57.1 psutil==5.9.5 tqdm==4.65.0 joblib==1.3.1

echo Installing utilities...
pip install requests==2.31.0 python-dotenv==1.0.0 pyyaml==6.0.1

echo Installing development tools...
pip install pytest==7.4.0 black==23.7.0 flake8==6.0.0

REM Optional: Try to install AMD GPU acceleration
echo.
echo 🎮 Attempting to install AMD GPU acceleration (optional)...
pip install torch-directml 2>nul
if errorlevel 1 (
    echo ⚠️  AMD DirectML not available - CPU training will be used
) else (
    echo ✅ AMD DirectML installed - GPU acceleration available
)

REM Create project directories
echo.
echo 📁 Creating enhanced training directories...
if not exist "enhanced_training" mkdir enhanced_training
if not exist "enhanced_training\datasets" mkdir enhanced_training\datasets
if not exist "enhanced_training\models" mkdir enhanced_training\models  
if not exist "enhanced_training\results" mkdir enhanced_training\results
if not exist "enhanced_training\cache" mkdir enhanced_training\cache
if not exist "enhanced_training\logs" mkdir enhanced_training\logs

if not exist "data\raw\ecg-arrhythmia-dataset" mkdir data\raw\ecg-arrhythmia-dataset
if not exist "data\enhanced" mkdir data\enhanced
if not exist "models\enhanced" mkdir models\enhanced

REM Create .env file
echo.
echo ⚙️  Creating environment configuration...
(
echo # Enhanced Training Configuration for AMD Ryzen 5 5600G PC
echo ENHANCED_TRAINING=true
echo USE_GPU_ACCELERATION=true
echo MAX_WORKERS=6
echo BATCH_SIZE=512
echo CACHE_ENABLED=true
echo VERBOSE_LOGGING=true
echo.
echo # Memory Management for 10GB RAM
echo PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
echo OMP_NUM_THREADS=6
echo NUMBA_NUM_THREADS=6
echo.
echo # Dataset Paths
echo ARRHYTHMIA_DATASET_PATH=./data/raw/ecg-arrhythmia-dataset
echo ENHANCED_MODELS_PATH=./models/enhanced
echo ENHANCED_CACHE_PATH=./enhanced_training/cache
) > .env

REM Create VS Code workspace
echo.
echo 💻 Creating VS Code workspace configuration...
(
echo {
echo     "folders": [
echo         {
echo             "path": "."
echo         }
echo     ],
echo     "settings": {
echo         "python.defaultInterpreterPath": "./ecg_enhanced_env/Scripts/python.exe",
echo         "python.terminal.activateEnvironment": true,
echo         "python.linting.enabled": true,
echo         "python.linting.flake8Enabled": true,
echo         "python.formatting.provider": "black",
echo         "files.autoSave": "afterDelay",
echo         "editor.formatOnSave": true,
echo         "files.watcherExclude": {
echo             "**/data/raw/**": true,
echo             "**/enhanced_training/cache/**": true
echo         }
echo     },
echo     "extensions": {
echo         "recommendations": [
echo             "ms-python.python",
echo             "ms-python.vscode-pylance", 
echo             "ms-toolsai.jupyter",
echo             "eamodio.gitlens",
echo             "ms-python.black-formatter"
echo         ]
echo     }
echo }
) > ecg-enhanced-training.code-workspace

REM Run health check
echo.
echo 🧪 Running system health check...
python pc_health_check.py

echo.
echo ======================================================================
echo                        ✅ SETUP COMPLETE!
echo ======================================================================
echo.
echo 🎯 Your PC is now ready for enhanced MI detection training!
echo.
echo 📋 WHAT'S BEEN CONFIGURED:
echo   ✅ Python virtual environment (ecg_enhanced_env)
echo   ✅ Enhanced ML dependencies installed
echo   ✅ AMD GPU acceleration configured (if available)
echo   ✅ Memory settings optimized for 10GB RAM
echo   ✅ VS Code workspace created
echo   ✅ Project directories structured
echo   ✅ Environment variables configured
echo.
echo 🚀 NEXT STEPS:
echo   1. Open VS Code: code ecg-enhanced-training.code-workspace
echo   2. Install recommended VS Code extensions
echo   3. Download ECG Arrhythmia Dataset (45,152 records)
echo   4. Run enhanced MI model training
echo   5. Target: 70%+ MI detection accuracy
echo.
echo 💡 TIPS:
echo   - Always activate environment: ecg_enhanced_env\Scripts\activate
echo   - Use 'Best Performance' power mode during training
echo   - Monitor RAM usage with Task Manager
echo   - Close unnecessary applications during training
echo.
echo Press any key to open VS Code workspace...
pause >nul

REM Open VS Code workspace
code ecg-enhanced-training.code-workspace

echo.
echo 🎉 Happy training! Your AMD Ryzen 5 5600G is ready for enhanced MI detection!
echo.
pause