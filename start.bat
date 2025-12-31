@echo off
echo ============================================
echo   Starting Agent-OS
echo ============================================
echo.

:: Check for virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found
    echo Please run build.bat first
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Check if Ollama is running (optional)
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Ollama is running
) else (
    echo [WARN] Ollama not detected at localhost:11434
    echo        Chat will show "network access restricted" without Ollama
    echo        Download from: https://ollama.com
)

echo.
echo [INFO] Starting web server on http://localhost:8080
echo [INFO] Press Ctrl+C to stop
echo.

python -m uvicorn src.web.app:get_app --factory --host 0.0.0.0 --port 8080

pause
