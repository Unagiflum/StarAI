@echo off
setlocal

set "GPU_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%GPU_PYTHON%" (
    echo GPU/training Python was not found: "%GPU_PYTHON%"
    echo Create .venv and install dependencies with: python -m pip install -r requirements-build.txt
    endlocal
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" -BuildName "StarAI_Train" -PythonPath "%GPU_PYTHON%" %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
