@echo off
setlocal

set "CPUAI_PYTHON=%~dp0.venv-cpuai\Scripts\python.exe"

if not exist "%CPUAI_PYTHON%" (
    echo CPU AI Python was not found: "%CPUAI_PYTHON%"
    echo Create it with: python -m venv .venv-cpuai
    echo Then install dependencies with: .\.venv-cpuai\Scripts\python.exe -m pip install -r requirements-cpuai.txt
    echo Then install build tooling with: .\.venv-cpuai\Scripts\python.exe -m pip install -r requirements-build-tools.txt
    endlocal
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" -BuildName "StarAI_CPUAI" -PythonPath "%CPUAI_PYTHON%" %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
