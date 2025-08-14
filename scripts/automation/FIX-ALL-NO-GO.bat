@echo off
REM FIX-ALL-NO-GO.bat
cls
echo FIXING WITHOUT GO...

REM Start Redis
start /min redis-server 2>nul || (
    echo Installing Redis...
    curl -L https://github.com/microsoftarchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.zip -o redis.zip
    tar -xf redis.zip
    start /min Redis-x64-3.2.100\redis-server.exe
)

REM Use existing compiled Go server
cd go-microservice
if exist main.exe (
    echo Using existing main.exe
    start /min main.exe
) else if exist server.exe (
    echo Using existing server.exe
    start /min server.exe
) else if exist legal-ai-server.exe (
    echo Using existing legal-ai-server.exe
    start /min legal-ai-server.exe
) else (
    echo No Go server found! Using Node.js fallback...
    cd ..
    node fallback-server.js
)

cd ..
timeout /t 3 >nul

REM Start Vite
start cmd /k npm run dev

echo.
echo READY: http://localhost:3130
pause