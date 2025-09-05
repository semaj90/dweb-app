@echo off
echo Starting Node.js Cluster Manager for Legal AI system...

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js not found in PATH
    echo Please install Node.js and add it to your PATH
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

echo Starting Legal AI Node.js Cluster Manager...
echo.
echo Cluster Configuration:
echo   - Legal Workers: 3 instances (512MB each)
echo   - AI Workers: 2 instances (1GB each)  
echo   - Vector Workers: 2 instances (256MB each)
echo   - Database Workers: 3 instances (256MB each)
echo.
echo Management Server: http://localhost:3000
echo Health Check: http://localhost:3000/health
echo Cluster Status: http://localhost:3000/status
echo.

REM Start the cluster manager
node cluster-manager.js

echo.
echo Cluster manager stopped
pause