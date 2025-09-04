@echo off
echo ==========================================
echo Legal AI System Startup with NATS
echo ==========================================
echo.
echo Starting all Legal AI services...
echo.
echo 1. Starting NATS Server...
start "NATS Server" /min nats-server\start-nats.bat
timeout /t 3 /nobreak > nul

echo 2. Checking NATS status...
curl -s http://localhost:8222/varz 2>nul | findstr "server_name" >nul
if %errorlevel% equ 0 (
    echo ✅ NATS Server is running
) else (
    echo ❌ NATS Server failed to start
)

echo 3. Starting SvelteKit development server...
echo 🚀 Navigate to: http://localhost:5173/demos/nats-messaging
echo.
npm run dev
