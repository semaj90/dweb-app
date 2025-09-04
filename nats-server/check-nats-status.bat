@echo off
echo ==========================================
echo NATS Server Status Check
echo ==========================================
echo.
echo Checking NATS server status...
curl -s http://localhost:8222/varz 2>nul | findstr "server_name" >nul
if %errorlevel% equ 0 (
    echo ✅ NATS Server is running
    echo 📡 Main port: 4222
    echo 🌐 WebSocket port: 4223
    echo 📊 Monitoring: http://localhost:8222
    echo.
    echo 📈 Server stats:
    curl -s http://localhost:8222/varz 2>nul | findstr "connections\|in_msgs\|out_msgs"
) else (
    echo ❌ NATS Server is not running
    echo Run start-nats.bat to start the server
)
echo.
pause
