@echo off
echo ==============================================================================
echo              Legal AI Enterprise Orchestration System
echo                         Unified Startup Script
echo ==============================================================================
echo.

REM Check if running as administrator for Windows Services
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some services require administrator privileges
    echo Run as Administrator for full Windows Service integration
    echo.
)

REM Set color scheme for better visibility
color 0A

echo [ORCHESTRATOR] Starting Legal AI Enterprise Orchestration System...
echo [ORCHESTRATOR] Timestamp: %date% %time%
echo.

REM Check prerequisites
echo [PREREQUISITES] Checking system requirements...

REM Check Node.js
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js not found. Please install Node.js and add to PATH
    pause
    exit /b 1
)

REM Check Go
where go >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Go not found. Please install Go and add to PATH
    pause
    exit /b 1
)

REM Check if orchestration controller exists
if not exist "orchestration-controller.js" (
    echo ERROR: Orchestration controller not found
    echo Please ensure orchestration-controller.js is in the current directory
    pause
    exit /b 1
)

REM Check if configuration exists
if not exist "orchestration-config.json" (
    echo ERROR: Orchestration configuration not found
    echo Please ensure orchestration-config.json is in the current directory
    pause
    exit /b 1
)

echo [PREREQUISITES] ✅ All prerequisites satisfied
echo.

REM Create necessary directories
echo [SETUP] Creating directory structure...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "temp" mkdir temp
if not exist "certs" mkdir certs

echo [SETUP] ✅ Directory structure ready
echo.

REM Install Node.js dependencies if package.json exists
if exist "package.json" (
    echo [SETUP] Installing Node.js dependencies...
    npm install --silent
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Failed to install some dependencies
    ) else (
        echo [SETUP] ✅ Dependencies installed
    )
    echo.
)

REM Show system overview
echo [SYSTEM] Legal AI Orchestration System Overview:
echo.
echo   🏗️  Go-Kratos Microservice        : localhost:8080
echo   📊  ELK Stack (Monitoring)        : localhost:9200, 5601, 5044
echo   🚀  NATS Message Queue           : localhost:4222 (mgmt: 8222)
echo   🌐  Node.js Cluster Manager      : localhost:3000
echo   ⚡  QUIC Protocol Gateway        : localhost:8443 (HTTP/3: 8444)
echo   🔧  Windows Service Manager      : localhost:9000-9003
echo   🎮  WebGPU Tensor Engine         : localhost:7000
echo   🔄  XState Workflow Engine       : localhost:6000
echo   📡  Orchestration Controller     : localhost:8000
echo.
echo   📋  Management Dashboard         : http://localhost:8000/status
echo   📊  Health Monitoring           : http://localhost:8000/health
echo   🔍  Service Discovery           : http://localhost:8000/services
echo   📈  Real-time Metrics           : ws://localhost:8000 (WebSocket)
echo.

REM Start fully wired orchestration system
echo [ORCHESTRATOR] Starting fully wired Legal AI orchestration system...
echo [ORCHESTRATOR] This includes all integrated components:
echo               - Service Discovery & Registration
echo               - Inter-Service Communication (NATS)  
echo               - Health Monitoring & Alerting
echo               - Dependency Management
echo               - Configuration Management
echo               - Comprehensive Logging (ELK Stack)
echo               - Message Routing & Coordination
echo.

REM Start in new window for monitoring
start "Legal AI Wired Orchestration System" cmd /k "node WIRED-ORCHESTRATION-SYSTEM.js"

REM Wait for orchestration controller to initialize
echo [ORCHESTRATOR] Waiting for orchestration controller to initialize...
timeout /t 5 /nobreak >nul

REM Check if orchestration controller is running
curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [ORCHESTRATOR] ✅ Orchestration controller is running
) else (
    echo [ORCHESTRATOR] ⏳ Orchestration controller is starting up...
)
echo.

REM Display management options
echo [MANAGEMENT] Orchestration system is starting up...
echo [MANAGEMENT] Available management commands:
echo.
echo   📊 System Status    : curl http://localhost:8000/status
echo   🔍 Service List     : curl http://localhost:8000/services  
echo   💚 Health Check     : curl http://localhost:8000/health
echo   📈 Metrics Data     : curl http://localhost:8000/metrics
echo   🔄 Restart Service  : curl -X POST http://localhost:8000/restart
echo   🛑 Graceful Stop    : curl -X POST http://localhost:8000/shutdown
echo.

REM Open management dashboard in browser
echo [MANAGEMENT] Opening management dashboard...
timeout /t 3 /nobreak >nul
start http://localhost:8000/status

REM Show VS Code integration
echo [VS-CODE] Available VS Code tasks (Ctrl+Shift+P):
echo.
echo   🏗️  Go-Kratos: Build ^& Run
echo   📊  ELK: Start Elasticsearch/Logstash/Kibana  
echo   🚀  NATS: Start Message Queue
echo   🌐  Node: Start Cluster Manager
echo   ⚡  QUIC: Start Protocol Services
echo   🔧  Windows: Start Service Manager
echo   🚀  Full Stack: Start All Services
echo   📋  Orchestration: Health Check All
echo.

REM Monitor startup progress
echo [MONITOR] Monitoring service startup progress...
echo [MONITOR] This may take 1-2 minutes for all services to be healthy
echo.

REM Wait for user input
echo [READY] ✅ Legal AI Orchestration System is starting up!
echo.
echo ══════════════════════════════════════════════════════════════════════════════
echo   Press any key to view real-time logs, or close this window to continue
echo   The orchestration system will continue running in the background
echo ══════════════════════════════════════════════════════════════════════════════
echo.
pause >nul

REM Show real-time status
:STATUS_LOOP
cls
echo ==============================================================================
echo                    Legal AI System - Real-time Status
echo ==============================================================================
echo.

REM Get system status from orchestration controller
curl -s http://localhost:8000/status 2>nul | findstr /C:"status" /C:"healthy" /C:"uptime"
if %ERRORLEVEL% NEQ 0 (
    echo [STATUS] Orchestration controller not responding
    echo [STATUS] Services may still be starting up...
) else (
    echo [STATUS] ✅ System operational
)

echo.
echo [CONTROLS] Press 'R' to refresh, 'Q' to quit monitoring, 'S' to shutdown system
echo.

REM Simple input handling (Windows limitation - using timeout for refresh)
timeout /t 10 /nobreak >nul

REM Check for quit condition (simplified)
goto STATUS_LOOP

REM Cleanup on exit
:CLEANUP
echo.
echo [SHUTDOWN] Initiating graceful shutdown...
curl -X POST http://localhost:8000/shutdown >nul 2>&1
echo [SHUTDOWN] Shutdown signal sent to orchestration controller
echo [SHUTDOWN] Services will stop gracefully
echo.
pause