@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set ENV_NAME=sw_weather
set "PROJ_ROOT=%~dp0"

echo ============================================================
echo   weather_sw - One-time Setup
echo ============================================================

where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] conda not found. Please install Anaconda or Miniconda.
    pause
    exit /b 1
)

REM Find conda base path
for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
call "%CONDA_BASE%\condabin\conda.bat" activate base

REM Check if env already exists
conda env list | findstr /C:"%ENV_NAME%" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [*] Environment '%ENV_NAME%' already exists. Recreate? (y/n^)
    choice /c yn /n
    if !ERRORLEVEL! EQU 2 goto skip_env
    conda env remove -n %ENV_NAME% -y
)

echo [*] Creating conda environment '%ENV_NAME%' (Python 3.11) ...
conda create -n %ENV_NAME% python=3.11 -y
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create environment.
    pause
    exit /b 1
)

:skip_env
echo [*] Installing conda packages ...
conda install -n %ENV_NAME% numpy pandas scipy scikit-learn matplotlib -y

echo [*] Installing pip packages ...
conda run -n %ENV_NAME% pip install torch tensorboard meteostat pyyaml streamlit plotly

echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Next step: double-click run.bat to launch the app
echo ============================================================

pause
