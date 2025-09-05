@echo off
REM Start Redis with Legal AI Platform Configuration
REM This script starts Redis server with the optimized configuration

echo Starting Redis for Legal AI Platform...
echo.

REM Check if Redis is installed
where redis-server.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: redis-server.exe not found in PATH
    echo Please install Redis or add it to your PATH
    pause
    exit /b 1
)

REM Check if config file exists
if not exist "%~dp0..\redis.conf" (
    echo ERROR: redis.conf not found at %~dp0..\redis.conf
    echo Please ensure redis.conf exists in the frontend directory
    pause
    exit /b 1
)

REM Create data directory if it doesn't exist
if not exist "%~dp0..\redis-data" (
    mkdir "%~dp0..\redis-data"
    echo Created redis-data directory
)

REM Start Redis with our configuration
echo Starting Redis server with legal AI configuration...
echo Config file: %~dp0..\redis.conf
echo Data directory: %~dp0..\redis-data
echo.

cd /d "%~dp0.."
redis-server.exe redis.conf

REM If Redis exits, show the exit code
echo.
echo Redis server stopped with exit code %errorlevel%
if %errorlevel% neq 0 (
    echo Check the redis.log file for error details
)
pause