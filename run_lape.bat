@echo off
setlocal enabledelayedexpansion

title LAPE (Leak Analysis ^& Prediction Engine) - DATATHON 2026
echo ======================================================================
echo   LAPE: Leak Analysis ^& Prediction Engine
echo   DATATHON 2026 Track 4: The Money Leak
echo   Evidence-Linked Financial Triage ^& Leak Physics Platform
echo ======================================================================
echo.

:: 1. Navigate to script directory
cd /d "%~dp0"
echo [*] Working Directory: %CD%

:: 2. Locate Verified Python 3.12+ Executable
set "PYTHON_EXE="

:: First attempt: Windows Python Launcher py -3.12 or py -3
where py >nul 2>&1
if !errorlevel! equ 0 (
    py -3.12 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py -3.12"
    ) else (
        py -3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=py -3"
        )
    )
)

:: Second attempt: Standard python executable on PATH
if "%PYTHON_EXE%"=="" (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=python"
        )
    )
)

:: Third attempt: Default local Windows user path fallback
if "%PYTHON_EXE%"=="" (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    ) else if exist "C:\Users\kalpesh\AppData\Local\Programs\Python\Python312\python.exe" (
        set "PYTHON_EXE=C:\Users\kalpesh\AppData\Local\Programs\Python\Python312\python.exe"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] No suitable Python 3.10+ environment found!
    echo Please install Python 3.10 or higher and make sure it is added to your PATH.
    pause
    exit /b 1
)

echo [*] Using Python: %PYTHON_EXE%
echo.

:: 3. Dependency Pre-Flight Check
echo [*] Checking core dependencies...
"%PYTHON_EXE%" -c "import pandas, numpy, sklearn, streamlit; print('  [+] Core packages verified: pandas, numpy, sklearn, streamlit')"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Missing required dependencies in the Python environment!
    echo Attempting to install requirements...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation failed. Please check your environment.
        pause
        exit /b 1
    )
)

:: 4. Verify Dataset Existence
if not exist "data\raw\transactions.csv" (
    echo [*] Initializing data directory structure from DATASET/...
    "%PYTHON_EXE%" scripts\audit_data.py
)

:: 5. Launch LAPE Streamlit Dashboard
echo.
echo ======================================================================
echo   Launching LAPE Streamlit Dashboard on http://localhost:8501
echo   Press Ctrl+C in this terminal window to stop the server.
echo ======================================================================
echo.

"%PYTHON_EXE%" -m streamlit run app.py --server.headless=false --server.port=8501

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Streamlit exited with an error code: %errorlevel%
    pause
)

endlocal
