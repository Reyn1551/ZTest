@echo off
REM ATCS Vision - Run Script for Windows

echo ==========================================
echo 🚦 ATCS JOGJA - AI Traffic Surveillance
echo ==========================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    exit /b 1
)

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  FFmpeg not found, will use OpenCV fallback
)

REM Create output directories
if not exist "outputs\recordings" mkdir outputs\recordings
if not exist "outputs\violations" mkdir outputs\violations
if not exist "outputs\reports" mkdir outputs\reports
if not exist "outputs\snapshots" mkdir outputs\snapshots

REM Run based on argument
if "%1"=="--gradio" (
    echo Starting Gradio Web UI...
    python main.py --gradio %2 %3 %4 %5
) else if "%1"=="-g" (
    echo Starting Gradio Web UI...
    python main.py --gradio %2 %3 %4 %5
) else if "%1"=="--help" (
    python main.py --help
) else (
    echo Starting OpenCV mode...
    python main.py %1 %2 %3 %4 %5
)
