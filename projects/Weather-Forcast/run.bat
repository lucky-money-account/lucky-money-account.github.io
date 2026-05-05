@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set ENV_NAME=sw_weather
set "PROJ_ROOT=%~dp0"

echo ============================================================
echo   weather_sw - Southwest China Weather Prediction
echo ============================================================

where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
if not exist "%CONDA_BASE%\condabin\conda.bat" (
    echo [ERROR] Cannot find conda at: %CONDA_BASE%
    pause
    exit /b 1
)
call "%CONDA_BASE%\condabin\conda.bat" activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Cannot activate conda env '%ENV_NAME%'.
    echo         Run setup.bat first.
    pause
    exit /b 1
)

echo.
echo [*] Refreshing weather data from Open-Meteo API ...
echo     (fetches only new days since last run)
python "%PROJ_ROOT%src\data_prep.py"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data download failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [*] Launching Streamlit web UI ...
echo.
echo     Open http://localhost:8501 in your browser
echo     Press Ctrl+C here to stop.
echo ============================================================
echo.

set KMP_DUPLICATE_LIB_OK=TRUE
streamlit run "%PROJ_ROOT%src\app.py"

pause
