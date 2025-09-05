@echo off
echo Starting Node.js Cluster Manager with Dynamic Port Allocation...
echo.

REM Set environment variables for dynamic port allocation
set MANAGER_PORT_AUTO=1
set FORCE_CLUSTER=1
set LOG_LEVEL=debug

echo Configuration:
echo   - Manager Port: Auto-detect starting from 3100
echo   - Force Override: Enabled
echo   - Log Level: Debug
echo.

REM Start cluster manager with dynamic port allocation
node cluster-manager.cjs --manager-port=3100 --legal-count=2 --ai-count=2 --vector-count=1 --database-count=2

echo.
echo Cluster startup complete!