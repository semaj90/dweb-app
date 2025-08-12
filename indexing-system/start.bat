@echo off
echo Starting Legal AI Indexing System...

REM Create logs directory
if not exist "logs" mkdir logs

REM Start PM2 services
echo Starting PM2 services...
pm2 start pm2.config.cjs

REM Wait for services
timeout /t 5 >nul

REM Show status
echo Service Status:
pm2 status

echo.
echo Access Points:
echo   Monitor Dashboard: http://localhost:8084
echo   Go Indexer API:    http://localhost:8081
echo   GPU Clustering:    http://localhost:8085
echo.
echo System startup complete!

pause