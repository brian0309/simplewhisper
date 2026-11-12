@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem ---------------------------------------------------------------- options
set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "ONEFILE=0"
set "CLEAN=0"
set "INSTALL=1"
set "PAUSE=1"
set "HELP=0"

rem ------------------------------------------------------ environment knobs
rem  BUILD_MODE         standalone (default) or onefile
rem  BUILD_INSTALLER    1 (default) or 0 - wrap the build in an installer
rem  AUTO_INSTALL_INNO  1 (default) or 0 - install Inno Setup when missing
rem  SIGN_MODE          self or none (none on CI, self on a desktop)
if not defined BUILD_MODE set "BUILD_MODE=standalone"
if not defined BUILD_INSTALLER set "BUILD_INSTALLER=1"
if not defined AUTO_INSTALL_INNO set "AUTO_INSTALL_INNO=1"
if not defined SIGN_MODE (
    if defined CI (set "SIGN_MODE=none") else (set "SIGN_MODE=self")
)

rem ------------------------------------------------------------ build details
set "COMPANY=SimpleWhisper"
set "PRODUCT_NAME=SimpleWhisper"
set "INSTALLER_BASENAME=simplewhisper-setup"
set "ISS_FILE=%~dp0installer\simplewhisper.iss"
set "LICENSE_FILE=%~dp0LICENSE"
set "ICON_ICO=%~dp0assets\simplewhisper.ico"
set "CERT_SUBJECT=SimpleWhisper Self-Signed"
set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS_EXE%" set "PS_EXE=powershell"

rem ----------------------------------------------------------------- parse
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--onefile" (
    set "ONEFILE=1"
    set "BUILD_MODE=onefile"
    shift
    goto parse
)
if /i "%~1"=="--clean" (
    set "CLEAN=1"
    shift
    goto parse
)
if /i "%~1"=="--no-install" (
    set "INSTALL=0"
    shift
    goto parse
)
if /i "%~1"=="--no-pause" (
    set "PAUSE=0"
    shift
    goto parse
)
if /i "%~1"=="--help" (
    set "HELP=1"
    shift
    goto parse
)
if /i "%~1"=="-h" (
    set "HELP=1"
    shift
    goto parse
)
echo Unknown option: %~1
goto usage

:parsed
if "%HELP%"=="1" goto usage

if /i not "%BUILD_MODE%"=="standalone" if /i not "%BUILD_MODE%"=="onefile" (
    echo [ERROR] Invalid BUILD_MODE "%BUILD_MODE%". Use "standalone" or "onefile".
    goto usage
)
if "%ONEFILE%"=="1" set "BUILD_MODE=onefile"

rem -------------------------------------------------------------- locations
if "%ONEFILE%"=="1" (
    set "EXE_CLI=%~dp0dist\SimpleWhisper.exe"
    set "EXE_GUI=%~dp0dist\SimpleWhisper-GUI.exe"
) else (
    set "DIST_DIR=%~dp0dist\SimpleWhisper"
    set "EXE_CLI=%~dp0dist\SimpleWhisper\SimpleWhisper.exe"
    set "EXE_GUI=%~dp0dist\SimpleWhisper\SimpleWhisper-GUI.exe"
)

echo.
echo ==================================================
echo  SimpleWhisper - standalone build
echo ==================================================
echo.
echo   mode:       %BUILD_MODE%
echo   installer:  %BUILD_INSTALLER%
echo.

rem ------------------------------------------------------- virtual environment
if exist "%PY%" (
    echo [1/5] Using the existing virtual environment.
) else (
    echo [1/5] Creating the virtual environment...
    call :create_venv
    if errorlevel 1 goto failed
)

rem --------------------------------------------------------------- dependencies
if "%INSTALL%"=="1" (
    echo [2/5] Installing the build dependencies...
    "%PY%" -m pip install --upgrade pip >nul
    "%PY%" -m pip install -e ".[gui]" || goto failed
    "%PY%" -m pip install "pyinstaller>=6.6" || goto failed
) else (
    echo [2/5] Skipping the dependency install.
)

rem ------------------------------------------------------------ version resource
echo [3/5] Writing the version resource...
"%PY%" "%~dp0scripts\make_version_info.py" || goto failed

set "APP_VERSION="
set "VERSION_FILE=%TEMP%\simplewhisper-version.txt"
"%PY%" "%~dp0scripts\make_version_info.py" --print-version > "%VERSION_FILE%" || goto failed
set /p APP_VERSION=<"%VERSION_FILE%"
del /f /q "%VERSION_FILE%" >nul 2>nul
if not defined APP_VERSION (
    echo [ERROR] Could not read the version from pyproject.toml.
    goto failed
)
echo        version: %APP_VERSION%

rem ------------------------------------------------------------------ pyinstaller
echo [4/5] Running PyInstaller...
set "SIMPLEWHISPER_ONEFILE=%ONEFILE%"
if "%CLEAN%"=="1" (
    "%PY%" -m PyInstaller --noconfirm --clean "%~dp0SimpleWhisper.spec" || goto failed
) else (
    "%PY%" -m PyInstaller --noconfirm "%~dp0SimpleWhisper.spec" || goto failed
)

if not exist "%EXE_CLI%" (
    echo [ERROR] Expected executable not found: %EXE_CLI%
    goto failed
)
if not exist "%EXE_GUI%" (
    echo [ERROR] Expected executable not found: %EXE_GUI%
    goto failed
)

rem ---------------------------------------------------------------- code signing
if /i "%SIGN_MODE%"=="self" (
    echo [INFO] Self-signing the executables with a local certificate...
    call :sign "%EXE_CLI%"
    call :sign "%EXE_GUI%"
    if not exist "%~dp0build" mkdir "%~dp0build" >nul 2>nul
    "%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -Command ^
        "$subject='CN=%CERT_SUBJECT%';" ^
        "$cert=Get-ChildItem 'Cert:\CurrentUser\My' | Where-Object { $_.Subject -eq $subject -and $_.HasPrivateKey } | Select-Object -First 1;" ^
        "if($cert){ Export-Certificate -Cert $cert -FilePath '%~dp0build\simplewhisper-selfsign.cer' -Force | Out-Null };"
    echo        Certificate: %~dp0build\simplewhisper-selfsign.cer
    echo        Trust it locally with:
    echo          certutil -user -addstore TrustedPublisher "%~dp0build\simplewhisper-selfsign.cer"
    echo          certutil -user -addstore Root "%~dp0build\simplewhisper-selfsign.cer"
) else (
    echo [INFO] Signing skipped. SIGN_MODE=%SIGN_MODE%
)

rem ------------------------------------------------------------------ installer
if /i not "%BUILD_MODE%"=="standalone" (
    echo [5/5] Installer skipped: BUILD_MODE=%BUILD_MODE%.
    echo        The installer wraps the folder build.
    goto result
)
if /i not "%BUILD_INSTALLER%"=="1" (
    echo [5/5] Installer skipped: BUILD_INSTALLER=%BUILD_INSTALLER%.
    goto result
)

echo [5/5] Building the installer...
if not exist "%ISS_FILE%" (
    echo [ERROR] Inno Setup script not found: %ISS_FILE%
    goto failed
)

call :find_iscc
if not defined ISCC_EXE (
    echo [ERROR] Inno Setup compiler ^(ISCC.exe^) not found.
    echo         Install Inno Setup 6 or add ISCC.exe to PATH.
    echo         Set AUTO_INSTALL_INNO=0 to skip the automatic install,
    echo         or BUILD_INSTALLER=0 to skip the installer.
    goto failed
)
echo        Using ISCC: !ISCC_EXE!

set "INSTALLER_OUTDIR=%~dp0build\installer"
if not exist "%INSTALLER_OUTDIR%" mkdir "%INSTALLER_OUTDIR%" >nul 2>nul
set "SETUP_EXE=%INSTALLER_OUTDIR%\%INSTALLER_BASENAME%.exe"
if exist "%SETUP_EXE%" del /f /q "%SETUP_EXE%" >nul 2>nul

set "INSTALLER_ICON_DEFINE="
if exist "%ICON_ICO%" set INSTALLER_ICON_DEFINE="/DSetupIconFile=%ICON_ICO%"

if defined INSTALLER_ICON_DEFINE (
    "!ISCC_EXE!" /Qp ^
        "/DAppName=%PRODUCT_NAME%" ^
        "/DAppVersion=%APP_VERSION%" ^
        "/DPublisher=%COMPANY%" ^
        "/DSourceDir=%DIST_DIR%" ^
        "/DOutputDir=%INSTALLER_OUTDIR%" ^
        "/DOutputBaseFilename=%INSTALLER_BASENAME%" ^
        "/DLicenseFile=%LICENSE_FILE%" ^
        !INSTALLER_ICON_DEFINE! ^
        "%ISS_FILE%"
) else (
    "!ISCC_EXE!" /Qp ^
        "/DAppName=%PRODUCT_NAME%" ^
        "/DAppVersion=%APP_VERSION%" ^
        "/DPublisher=%COMPANY%" ^
        "/DSourceDir=%DIST_DIR%" ^
        "/DOutputDir=%INSTALLER_OUTDIR%" ^
        "/DOutputBaseFilename=%INSTALLER_BASENAME%" ^
        "/DLicenseFile=%LICENSE_FILE%" ^
        "%ISS_FILE%"
)

if errorlevel 1 (
    echo [ERROR] Installer build failed.
    goto failed
)

if not exist "%SETUP_EXE%" (
    echo [ERROR] Expected installer not found: %SETUP_EXE%
    goto failed
)

if /i "%SIGN_MODE%"=="self" (
    echo [INFO] Self-signing the installer...
    call :sign "%SETUP_EXE%"
)

rem --------------------------------------------------------------------- result
:result
echo.
echo Build finished.
echo.
echo   command line:  %EXE_CLI%
echo   desktop app:   %EXE_GUI%
if defined SETUP_EXE if exist "%SETUP_EXE%" echo   installer:     %SETUP_EXE%
echo.
echo The models are downloaded on the first run and are not part of the build.
echo ffmpeg is optional: without it the bundled PyAV decoder is used.
echo.
if "%PAUSE%"=="1" pause
endlocal & exit /b 0

rem ------------------------------------------------------------------- helpers
:create_venv
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -m venv "%VENV%" >nul 2>&1
    if errorlevel 1 py -3 -m venv "%VENV%"
) else (
    python -m venv "%VENV%"
)
if not exist "%PY%" (
    echo     Could not create the virtual environment.
    echo     Install Python 3.11 or newer and try again.
    exit /b 1
)
exit /b 0

rem Find ISCC.exe: PATH, the usual install folders, then winget or choco.
:find_iscc
set "ISCC_EXE="
where iscc >nul 2>nul && set "ISCC_EXE=iscc"
if not defined ISCC_EXE if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC_EXE if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC_EXE if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC_EXE if /i not "%AUTO_INSTALL_INNO%"=="1" exit /b 0

if not defined ISCC_EXE (
    echo        Inno Setup was not found, trying to install it...
    where winget >nul 2>nul
    if not errorlevel 1 (
        winget install --id JRSoftware.InnoSetup -e --silent --accept-package-agreements --accept-source-agreements
    ) else (
        where choco >nul 2>nul
        if not errorlevel 1 (
            choco install innosetup -y --no-progress
        ) else (
            echo [WARN] Neither winget nor choco is available for the automatic install.
        )
    )

    where iscc >nul 2>nul && set "ISCC_EXE=iscc"
    if not defined ISCC_EXE if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    if not defined ISCC_EXE if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles%\Inno Setup 6\ISCC.exe"
    if not defined ISCC_EXE if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)
exit /b 0

rem Sign one file with the local self-signed certificate. %1 = file path.
:sign
if /i not "%SIGN_MODE%"=="self" exit /b 0
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$subject='CN=%CERT_SUBJECT%';" ^
    "$store='Cert:\CurrentUser\My';" ^
    "$cert=Get-ChildItem $store | Where-Object { $_.Subject -eq $subject -and $_.HasPrivateKey } | Sort-Object NotAfter -Descending | Select-Object -First 1;" ^
    "if(-not $cert){ $cert=New-SelfSignedCertificate -Type CodeSigningCert -Subject $subject -CertStoreLocation $store -NotAfter (Get-Date).AddYears(3) };" ^
    "$null=Set-AuthenticodeSignature -FilePath '%~1' -Certificate $cert;"
if errorlevel 1 (
    echo [WARN] Self-sign failed for %~1. Continuing without a signature.
    exit /b 0
)
echo        Signed %~1
exit /b 0

:failed
echo.
echo The build failed. The output above says where it stopped.
if "%PAUSE%"=="1" pause
endlocal & exit /b 1

:usage
echo.
echo Usage: build.bat [options]
echo.
echo   --onefile     two single file executables instead of a folder
echo   --clean       rebuild everything from scratch
echo   --no-install  do not install the dependencies first
echo   --no-pause    do not wait for a key press at the end
echo   --help        show this text
echo.
echo Environment variables:
echo   BUILD_MODE         standalone ^(default^) or onefile
echo   BUILD_INSTALLER    1 ^(default^) or 0 - build the Inno Setup installer
echo   AUTO_INSTALL_INNO  1 ^(default^) or 0 - install Inno Setup when missing
echo   SIGN_MODE          self or none ^(none on CI, self on a desktop^)
echo.
if "%PAUSE%"=="1" pause
endlocal & exit /b 0