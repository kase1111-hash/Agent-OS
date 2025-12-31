@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Agent-OS Windows Build Script
echo ============================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Display Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found %PYTHON_VERSION%

:: Check if virtual environment exists
if exist "venv" (
    echo [OK] Virtual environment found
) else (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install dependencies
echo.
echo [INFO] Installing dependencies (this may take a while)...
echo [INFO] PyTorch and CUDA libraries are ~4GB - please be patient
echo.
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

:: Install package in editable mode
echo [INFO] Installing Agent-OS in editable mode...
pip install -e . --quiet 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Editable install skipped (no pyproject.toml or setup.py issue)
)

:: Create .env if it doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        echo [INFO] Creating .env from .env.example...
        copy .env.example .env >nul
        echo [OK] Created .env file
    )
)

:: Create data directory
if not exist "data" (
    mkdir data
    echo [OK] Created data directory
)

echo.
echo ============================================
echo   Build Complete!
echo ============================================
echo.
echo To start Agent-OS, run:
echo   start.bat
echo.
echo Or manually:
echo   venv\Scripts\activate
echo   python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080
echo.
echo Access the web interface at: http://localhost:8080
echo.

pause
