@echo off
echo Starting Qdrant Vector Database Locally (No Docker)
echo ====================================================

REM Create storage directory if it doesn't exist
if not exist "qdrant_storage" (
    echo Creating Qdrant storage directory...
    mkdir qdrant_storage
)

REM Check if Qdrant executable exists
if not exist "qdrant-windows\qdrant.exe" (
    echo ERROR: Qdrant executable not found at qdrant-windows\qdrant.exe
    echo Please ensure Qdrant is extracted properly.
    pause
    exit /b 1
)

echo Starting Qdrant with local configuration...
echo Access URL: http://localhost:6333
echo Dashboard: http://localhost:6333/dashboard
echo.

REM Start Qdrant with local configuration
qdrant-windows\qdrant.exe --config-path qdrant-local-config.yaml

pause