@echo off
title Legal AI - Windows Production Launcher
color 0A

echo.
echo  ██╗    ██╗██╗███╗   ██╗██████╗  ██████╗ ██╗    ██╗███████╗
echo  ██║    ██║██║████╗  ██║██╔══██╗██╔═══██╗██║    ██║██╔════╝
echo  ██║ █╗ ██║██║██╔██╗ ██║██║  ██║██║   ██║██║ █╗ ██║███████╗
echo  ██║███╗██║██║██║╚██╗██║██║  ██║██║   ██║██║███╗██║╚════██║
echo  ╚███╔███╔╝██║██║ ╚████║██████╔╝╚██████╔╝╚███╔███╔╝███████║
echo   ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝  ╚══╝╚══╝ ╚══════╝
echo.
echo  🚀 PRODUCTION PHASES LAUNCHER
echo  ═══════════════════════════════
echo.

:MENU
echo  PRODUCTION PHASES:
echo  [1] Phase 2 - Production Setup (Docker + Nginx)
echo  [2] Phase 3 - Performance Optimization
echo  [3] Phase 4 - Security Configuration  
echo  [4] Phase 5 - Launch Production (WSL)
echo  [5] Phase 6 - Production Monitoring
echo  [6] All Phases Setup
echo  [7] Health Check All Services
echo  [0] Exit
echo.
set /p choice="Select phase: "

if "%choice%"=="1" goto PHASE2
if "%choice%"=="2" goto PHASE3
if "%choice%"=="3" goto PHASE4
if "%choice%"=="4" goto PHASE5
if "%choice%"=="5" goto PHASE6
if "%choice%"=="6" goto ALL_PHASES
if "%choice%"=="7" goto HEALTH_CHECK
if "%choice%"=="0" goto EXIT
goto MENU

:PHASE2
echo Running Phase 2 - Production Setup...
powershell -ExecutionPolicy Bypass -File "phase2-production-setup-windows.ps1"
goto MENU

:PHASE3
echo Running Phase 3 - Performance Setup...
call PHASE3-PERFORMANCE-SETUP.bat
goto MENU

:PHASE4
echo Running Phase 4 - Security Setup...
call PHASE4-SECURITY-SETUP.bat
goto MENU

:PHASE5
echo Running Phase 5 - WSL Production Launch...
wsl bash launch-production-wsl.sh
goto MENU

:PHASE6
echo Setting up production monitoring...
mkdir logs\monitoring 2>nul
mkdir deployment\monitoring 2>nul

:: Create monitoring docker-compose
echo version: '3.8' > deployment\monitoring\docker-compose.monitoring.yml
echo. >> deployment\monitoring\docker-compose.monitoring.yml
echo services: >> deployment\monitoring\docker-compose.monitoring.yml
echo   prometheus: >> deployment\monitoring\docker-compose.monitoring.yml
echo     image: prom/prometheus:latest >> deployment\monitoring\docker-compose.monitoring.yml
echo     ports: >> deployment\monitoring\docker-compose.monitoring.yml
echo       - "9090:9090" >> deployment\monitoring\docker-compose.monitoring.yml
echo   grafana: >> deployment\monitoring\docker-compose.monitoring.yml
echo     image: grafana/grafana:latest >> deployment\monitoring\docker-compose.monitoring.yml
echo     ports: >> deployment\monitoring\docker-compose.monitoring.yml
echo       - "3001:3000" >> deployment\monitoring\docker-compose.monitoring.yml

echo ✅ Phase 6 monitoring setup complete!
goto MENU

:ALL_PHASES
echo Running all production phases...
echo.
echo [1/5] Phase 2 - Production Setup...
powershell -ExecutionPolicy Bypass -File "phase2-production-setup-windows.ps1"
echo.
echo [2/5] Phase 3 - Performance...
call PHASE3-PERFORMANCE-SETUP.bat
echo.
echo [3/5] Phase 4 - Security...
call PHASE4-SECURITY-SETUP.bat
echo.
echo [4/5] Phase 5 - Launch Production...
wsl bash launch-production-wsl.sh
echo.
echo [5/5] Phase 6 - Monitoring...
docker-compose -f deployment\monitoring\docker-compose.monitoring.yml up -d
echo.
echo ✅ All phases complete!
goto MENU

:HEALTH_CHECK
echo Running comprehensive health check...
echo.
echo Docker Desktop:
docker version

echo.
echo WSL Integration:
wsl docker ps

echo.
echo Production Services:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr legal-ai

echo.
echo Service Health:
curl -s http://localhost/health 2>nul && echo ✅ Frontend healthy || echo ❌ Frontend down
curl -s http://localhost:6333/health 2>nul && echo ✅ Qdrant healthy || echo ❌ Qdrant down
docker exec legal-ai-redis-prod redis-cli ping 2>nul && echo ✅ Redis healthy || echo ❌ Redis down

echo.
pause
goto MENU

:EXIT
echo Goodbye!
exit
