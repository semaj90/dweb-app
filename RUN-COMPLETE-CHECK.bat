@echo off
REM RUN-COMPLETE-CHECK.bat
REM Comprehensive error checking for the entire system

echo ========================================================
echo    LEGAL AI SYSTEM - COMPLETE ERROR CHECK
echo    Checking Vite, Go, GPU, Redis, BullMQ, XState
echo ========================================================
echo.

REM First install dependencies for the checker
echo Installing validator dependencies...
npm install --no-save node-fetch ioredis bullmq neo4j-driver pg chalk >nul 2>&1

echo.
echo Running comprehensive system check...
echo.

REM Run the validation script
node check-system-integration.mjs

echo.
echo ========================================================
echo    CHECK COMPLETE
echo ========================================================
echo.
echo Quick Actions:
echo   1. Start all services: COMPLETE-AI-SYSTEM-STARTUP.bat
echo   2. Build Go server: build-go-server.bat
echo   3. Test integration: Test-Complete-Integration.ps1
echo   4. View logs: Check integration-report-*.json
echo.

pause
