@echo off
echo ============================================================
echo         FILE MERGE SYSTEM - ENHANCED SETUP
echo ============================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo This script requires Administrator privileges.
    echo Please run as Administrator.
    pause
    exit /b 1
)

echo Starting Enhanced File Merge System Setup...
echo.

REM Navigate to the project directory
cd /d "%~dp0"

REM Run the enhanced PowerShell setup script with all features
echo Running setup with:
echo   - Secure configuration generation
echo   - Monitoring stack
echo   - Backup creation
echo   - Full logging
echo.

powershell.exe -ExecutionPolicy Bypass -File "scripts\setup-complete-system.ps1" -GenerateSecureConfig -EnableMonitoring -CreateBackup

if %errorLevel% neq 0 (
    echo.
    echo Setup failed! Please check the logs in the ./logs directory
    pause
    exit /b 1
)

echo.
echo ============================================================
echo         SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo   1. Run start-system.bat to start all services
echo   2. Open http://localhost:5173 in your browser
echo   3. Check FILE_MERGE_BEST_PRACTICES.md for documentation
echo.
echo Services URLs:
echo   - Frontend:       http://localhost:5173
echo   - Backend API:    http://localhost:8084
echo   - MinIO Console:  http://localhost:9001
echo   - Qdrant UI:      http://localhost:6333/dashboard
echo   - Prometheus:     http://localhost:9090
echo   - Grafana:        http://localhost:3000
echo.
pause
