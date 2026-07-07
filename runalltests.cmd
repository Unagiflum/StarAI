@echo off
"%~dp0.venv\Scripts\python.exe" -m unittest discover -s tests
exit /b %ERRORLEVEL%
