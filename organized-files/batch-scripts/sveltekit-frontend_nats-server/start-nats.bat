@echo off
echo ==========================================
echo Starting NATS Server for Legal AI
echo ==========================================
echo.
echo 🚀 NATS Server starting...
echo 📡 Main port: 4222
echo 🌐 WebSocket port: 4223  
echo 📊 HTTP monitoring: 8222
echo.
echo Press Ctrl+C to stop the server
echo.
nats-server.exe -c nats-server.conf
pause
