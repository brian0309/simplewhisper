@echo off
setlocal
cd /d "%~dp0"
title SimpleWhisper - local speech-to-text
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
    echo SimpleWhisper is not installed yet.
    echo.
    echo Run these commands once:
    echo   py -3.11 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e .
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m simplewhisper %*
set "CODE=%ERRORLEVEL%"

echo.
pause
endlocal & exit /b %CODE%
