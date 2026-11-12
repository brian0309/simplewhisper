@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo SimpleWhisper is not installed yet.
    echo.
    echo Run these commands once:
    echo   py -3.11 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e ".[gui]"
    echo.
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" -m simplewhisper gui
