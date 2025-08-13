@echo off
echo Starting Legal AI Windows Services...

REM Check if running as administrator
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: This script requires administrator privileges
    echo Please run as Administrator
    pause
    exit /b 1
)

echo Checking Legal AI Windows Services status...
echo.

REM Check if services are installed
sc query "LegalAIManager" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Legal AI services are not installed
    echo Please run install-services.bat first
    echo.
    echo Would you like to install services now? (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        call install-services.bat
    ) else (
        echo Installation cancelled
        pause
        exit /b 1
    )
)

echo Starting Legal AI services in dependency order...
echo.

REM Start services in correct dependency order
echo [1/4] Starting Legal AI Manager Service...
sc start "LegalAIManager" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ LegalAIManager started successfully
) else (
    echo ❌ Failed to start LegalAIManager
)

REM Wait for manager to initialize
timeout /t 3 /nobreak >nul

echo [2/4] Starting Legal AI Database Service...
sc start "LegalAIDatabase" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ LegalAIDatabase started successfully
) else (
    echo ❌ Failed to start LegalAIDatabase
)

REM Wait for database to initialize
timeout /t 5 /nobreak >nul

echo [3/4] Starting Legal AI Vector Service...
sc start "LegalAIVector" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ LegalAIVector started successfully
) else (
    echo ❌ Failed to start LegalAIVector
)

REM Wait for vector service to initialize
timeout /t 3 /nobreak >nul

echo [4/4] Starting Legal AI Engine Service...
sc start "LegalAIEngine" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ LegalAIEngine started successfully
) else (
    echo ❌ Failed to start LegalAIEngine
)

echo.
echo Service startup completed!
echo.

REM Display service status
echo Current Legal AI service status:
echo.
for %%s in (LegalAIManager LegalAIDatabase LegalAIVector LegalAIEngine) do (
    for /f "tokens=4" %%a in ('sc query "%%s" ^| find "STATE"') do (
        if "%%a"=="RUNNING" (
            echo ✅ %%s: RUNNING
        ) else (
            echo ❌ %%s: %%a
        )
    )
)

echo.
echo Service endpoints:
echo   📊 Manager API: http://localhost:9000/status
echo   🗄️ Database API: http://localhost:9001/health
echo   🔍 Vector API: http://localhost:9002/health
echo   🤖 AI Engine API: http://localhost:9003/health
echo.
echo Service logs location: C:\ProgramData\LegalAI\Logs\
echo.
echo To stop services: stop-services.bat
echo To check status: check-services.bat
echo.
pause