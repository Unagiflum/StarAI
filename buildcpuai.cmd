@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" -BuildName "StarAI_CPUAI" %*
exit /b %ERRORLEVEL%
