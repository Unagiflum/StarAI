@echo off
setlocal
pushd "%~dp0" || exit /b 1
"%~dp0.venv\Scripts\python.exe" -m unittest discover -s tests
set "test_exit=%ERRORLEVEL%"
popd
exit /b %test_exit%
