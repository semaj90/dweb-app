@echo off
echo Starting NATS server for Legal AI system...

REM Check if NATS server is installed
where nats-server >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: nats-server not found in PATH
    echo Please install NATS server and add it to your PATH
    echo Download from: https://github.com/nats-io/nats-server/releases
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "C:\ProgramData\NATS\logs" mkdir "C:\ProgramData\NATS\logs"
if not exist "C:\ProgramData\NATS\jetstream" mkdir "C:\ProgramData\NATS\jetstream"

echo Starting NATS server with legal AI configuration...
nats-server -c "%~dp0nats-server.conf"

echo NATS server started
echo Monitoring available at: http://localhost:8222
pause