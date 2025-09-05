@echo off
echo ==========================================
echo Installing NATS Server for Legal AI Demo
echo ==========================================

:: Create NATS directory
if not exist "nats-server" mkdir nats-server
cd nats-server

:: Download NATS Server (Windows version)
echo 📥 Downloading NATS Server...
curl -L https://github.com/nats-io/nats-server/releases/download/v2.10.7/nats-server-v2.10.7-windows-amd64.zip -o nats-server.zip

:: Extract NATS Server
echo 📦 Extracting NATS Server...
powershell -command "Expand-Archive -Path nats-server.zip -DestinationPath . -Force"

:: Copy executable to main directory
copy nats-server-v2.10.7-windows-amd64\nats-server.exe .

:: Create NATS configuration file for Legal AI
echo 🔧 Creating NATS configuration...
(
echo # NATS Server Configuration for Legal AI Platform
echo # High-performance messaging with WebSocket support
echo
echo # Server settings
echo server_name: "legal-ai-nats-server"
echo port: 4222
echo
echo # Enable WebSocket for browser clients
echo websocket {
echo   port: 4223
echo   no_tls: true
echo   # Allow all origins for development
echo   same_origin: false
echo   allowed_origins: ["*"]
echo }
echo
echo # HTTP monitoring
echo http_port: 8222
echo
echo # Logging
echo log_file: "nats-server.log"
echo logtime: true
echo debug: false
echo trace: false
echo
echo # Performance settings
echo max_connections: 1000
echo max_subscriptions: 10000
echo max_payload: 64MB
echo
echo # Legal AI specific subjects and permissions
echo authorization {
echo   default_permissions: {
echo     publish: ["legal.>", "system.>", "_INBOX.>"]
echo     subscribe: ["legal.>", "system.>", "_INBOX.>"]
echo   }
echo   users: [
echo     {
echo       user: "legal_ai_client"
echo       password: "legal_ai_2024"
echo       permissions: {
echo         publish: ["legal.>", "system.>", "_INBOX.>"]
echo         subscribe: ["legal.>", "system.>", "_INBOX.>"]
echo       }
echo     }
echo   ]
echo }
echo
echo # Clustering (for future expansion)
echo cluster {
echo   name: "legal-ai-cluster"
echo   listen: "127.0.0.1:6222"
echo }
echo
echo # JetStream (for persistent messaging)
echo jetstream {
echo   store_dir: "./jetstream"
echo   max_mem: 1GB
echo   max_file: 10GB
echo }
) > nats-server.conf

:: Create startup script
echo 🚀 Creating startup script...
(
echo @echo off
echo echo ==========================================
echo echo Starting NATS Server for Legal AI
echo echo ==========================================
echo echo.
echo echo 🚀 NATS Server starting...
echo echo 📡 Main port: 4222
echo echo 🌐 WebSocket port: 4223  
echo echo 📊 HTTP monitoring: 8222
echo echo.
echo echo Press Ctrl+C to stop the server
echo echo.
echo nats-server.exe -c nats-server.conf
echo pause
) > start-nats.bat

:: Create status check script
echo 📊 Creating status check script...
(
echo @echo off
echo echo ==========================================
echo echo NATS Server Status Check
echo echo ==========================================
echo echo.
echo echo Checking NATS server status...
echo curl -s http://localhost:8222/varz 2^>nul ^| findstr "server_name" ^>nul
echo if %%errorlevel%% equ 0 (
echo     echo ✅ NATS Server is running
echo     echo 📡 Main port: 4222
echo     echo 🌐 WebSocket port: 4223
echo     echo 📊 Monitoring: http://localhost:8222
echo     echo.
echo     echo 📈 Server stats:
echo     curl -s http://localhost:8222/varz 2^>nul ^| findstr "connections\|in_msgs\|out_msgs"
echo ^) else (
echo     echo ❌ NATS Server is not running
echo     echo Run start-nats.bat to start the server
echo ^)
echo echo.
echo pause
) > check-nats-status.bat

:: Create test client script
echo 🧪 Creating test client script...
(
echo @echo off
echo echo ==========================================
echo echo NATS Test Client
echo echo ==========================================
echo echo.
echo echo This script tests NATS connectivity and legal AI subjects
echo echo.
echo echo Testing connection to NATS server...
echo curl -s http://localhost:8222/varz 2^>nul ^| findstr "server_name" ^>nul
echo if %%errorlevel%% neq 0 (
echo     echo ❌ NATS Server is not running
echo     echo Please start the server first with start-nats.bat
echo     pause
echo     exit /b 1
echo ^)
echo echo ✅ NATS Server is accessible
echo echo.
echo echo 📡 Available for Legal AI subjects:
echo echo    - legal.case.created
echo echo    - legal.document.uploaded  
echo echo    - legal.ai.analysis.completed
echo echo    - legal.search.query
echo echo    - legal.chat.message
echo echo    - system.health
echo echo.
echo echo 🌐 WebSocket endpoint: ws://localhost:4223
echo echo 📊 HTTP monitoring: http://localhost:8222
echo echo.
echo echo Ready for SvelteKit Legal AI integration!
echo pause
) > test-nats.bat

:: Create combined startup script for full system
echo 🔗 Creating combined system startup script...
(
echo @echo off
echo echo ==========================================
echo echo Legal AI System Startup with NATS
echo echo ==========================================
echo echo.
echo echo Starting all Legal AI services...
echo echo.
echo echo 1. Starting NATS Server...
echo start "NATS Server" /min nats-server\start-nats.bat
echo timeout /t 3 /nobreak ^> nul
echo.
echo echo 2. Checking NATS status...
echo curl -s http://localhost:8222/varz 2^>nul ^| findstr "server_name" ^>nul
echo if %%errorlevel%% equ 0 (
echo     echo ✅ NATS Server is running
echo ^) else (
echo     echo ❌ NATS Server failed to start
echo ^)
echo.
echo echo 3. Starting SvelteKit development server...
echo echo 🚀 Navigate to: http://localhost:5173/demos/nats-messaging
echo echo.
echo npm run dev
) > ..\start-legal-ai-with-nats.bat

echo.
echo ✅ NATS Server installation complete!
echo.
echo 📁 Installation directory: %CD%
echo 🔧 Configuration file: nats-server.conf
echo 🚀 Startup script: start-nats.bat
echo 📊 Status check: check-nats-status.bat
echo 🧪 Test client: test-nats.bat
echo.
echo ==========================================
echo Quick Start Commands:
echo ==========================================
echo.
echo 1. Start NATS Server:
echo    start-nats.bat
echo.
echo 2. Check server status:
echo    check-nats-status.bat
echo.
echo 3. Test connectivity:
echo    test-nats.bat
echo.
echo 4. Start full Legal AI system:
echo    ..\start-legal-ai-with-nats.bat
echo.
echo ==========================================
echo NATS Endpoints:
echo ==========================================
echo.
echo 📡 NATS Protocol: nats://localhost:4222
echo 🌐 WebSocket: ws://localhost:4223
echo 📊 HTTP Monitoring: http://localhost:8222
echo.
echo ==========================================
echo Legal AI Integration:
echo ==========================================
echo.
echo The NATS server is now configured with:
echo - Legal AI subject patterns (legal.*)
echo - WebSocket support for browser clients
echo - JetStream for persistent messaging
echo - HTTP monitoring and metrics
echo - Authentication for legal_ai_client
echo.
echo Ready for SvelteKit integration!
echo Navigate to: http://localhost:5173/demos/nats-messaging
echo.
pause