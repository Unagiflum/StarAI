@echo off
setlocal enabledelayedexpansion

REM Prompt the user for old and new prefixes
set /p oldprefix=Enter the old prefix (without numbers): 
set /p newprefix=Enter the new prefix (without numbers): 

for %%F in (%oldprefix%*.png) do (
    set "filename=%%~nF"
    REM Remove old prefix to isolate numeric part
    set "num=!filename:%oldprefix%=!"

    REM Prepend a zero and then take the last two chars to ensure two-digit format
    set "nn=0!num!"
    set "nn=!nn:~-2!"

    REM Rename the file
    ren "%%F" "!newprefix!!nn!%%~xF"
)

endlocal
echo Done!
pause
