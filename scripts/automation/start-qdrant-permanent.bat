@echo off
echo Starting Qdrant Vector Database Permanently
echo ==========================================

REM Kill any existing Qdrant processes
taskkill /F /IM qdrant.exe 2>nul

REM Create storage directory
if not exist "qdrant_storage" mkdir qdrant_storage

echo Starting Qdrant on http://localhost:6333
echo Dashboard: http://localhost:6333/dashboard
echo.

REM Start Qdrant in new window
start "Qdrant Vector DB" /MIN cmd /k "qdrant-windows\qdrant.exe --config-path qdrant-local-config.yaml"

REM Wait a moment for startup
timeout /t 3 >nul

echo Qdrant should now be running in a minimized window.
echo You can access the dashboard at: http://localhost:6333/dashboard
pause