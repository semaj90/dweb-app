@echo off
title GPU Legal AI Control Panel
color 0A

:menu
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║         GPU-ACCELERATED LEGAL AI CONTROL PANEL              ║
echo ║                  RTX 3060 Ti Optimized                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo   [1] 🚀 Start AI Service (Full Stack)
echo   [2] ⚡ Optimize GPU Performance
echo   [3] 📊 Monitor System (Real-time)
echo   [4] 🧪 Run Load Tests
echo   [5] 🐳 Deploy with Docker
echo   [6] 📈 View Metrics Dashboard
echo   [7] 🔧 Check System Status
echo   [8] 🗑️ Clear GPU Cache
echo   [9] 📖 View Documentation
echo   [0] ❌ Exit
echo.
echo ════════════════════════════════════════════════════════════════
set /p choice="Select option (0-9): "

if "%choice%"=="1" goto start_service
if "%choice%"=="2" goto optimize_gpu
if "%choice%"=="3" goto monitor_system
if "%choice%"=="4" goto run_tests
if "%choice%"=="5" goto docker_deploy
if "%choice%"=="6" goto view_metrics
if "%choice%"=="7" goto check_status
if "%choice%"=="8" goto clear_cache
if "%choice%"=="9" goto view_docs
if "%choice%"=="0" goto exit_app

goto invalid_choice

:start_service
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                 STARTING GPU AI SERVICE                     ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
call START-GPU-LEGAL-AI-8084.bat
pause
goto menu

:optimize_gpu
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║               OPTIMIZING GPU PERFORMANCE                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
powershell -ExecutionPolicy Bypass -File optimize-gpu-legal-ai.ps1
pause
goto menu

:monitor_system
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              REAL-TIME SYSTEM MONITORING                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
powershell -ExecutionPolicy Bypass -File monitor-gpu-ai.ps1
goto menu

:run_tests
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    LOAD TESTING SUITE                       ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
call test-gpu-ai-load.bat
goto menu

:docker_deploy
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   DOCKER DEPLOYMENT                         ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo [1] Build and Start Services
echo [2] Stop All Services
echo [3] View Logs
echo [4] Scale Services
echo [5] Back to Main Menu
echo.
set /p docker_choice="Select option (1-5): "

if "%docker_choice%"=="1" (
    docker-compose up -d --build
    echo.
    echo Services started successfully!
    echo Access points:
    echo   - API: http://localhost:8084
    echo   - Grafana: http://localhost:3000
    echo   - Prometheus: http://localhost:9090
)
if "%docker_choice%"=="2" (
    docker-compose down
    echo Services stopped.
)
if "%docker_choice%"=="3" (
    docker-compose logs -f --tail=100
)
if "%docker_choice%"=="4" (
    set /p scale_count="Number of instances (1-5): "
    docker-compose up -d --scale legal-ai-service=%scale_count%
)
if "%docker_choice%"=="5" goto menu

pause
goto docker_deploy

:view_metrics
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    METRICS DASHBOARD                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Opening metrics endpoints in browser...
start http://localhost:8084/api/metrics
start http://localhost:8084/api/health
echo.
echo Press any key to return to menu...
pause >nul
goto menu

:check_status
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                     SYSTEM STATUS                           ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Checking GPU...
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader
echo.
echo Checking Services...
echo.

echo Legal AI Service:
curl -s http://localhost:8084/api/health >nul 2>&1
if %errorlevel%==0 (
    echo   [✓] Running on port 8084
) else (
    echo   [✗] Not running
)

echo.
echo Ollama:
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel%==0 (
    echo   [✓] Running on port 11434
) else (
    echo   [✗] Not running
)

echo.
echo Redis:
redis-cli ping >nul 2>&1
if %errorlevel%==0 (
    echo   [✓] Running on port 6379
) else (
    echo   [✗] Not running
)

echo.
echo PostgreSQL:
pg_isready >nul 2>&1
if %errorlevel%==0 (
    echo   [✓] Running on port 5432
) else (
    echo   [✗] Not running
)

echo.
pause
goto menu

:clear_cache
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    CLEARING GPU CACHE                       ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Resetting GPU...
nvidia-smi --gpu-reset
echo.
echo Clearing Redis cache...
redis-cli FLUSHALL >nul 2>&1
echo.
echo Cache cleared successfully!
pause
goto menu

:view_docs
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                      DOCUMENTATION                          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo [1] View README
echo [2] Open API Documentation (Browser)
echo [3] View Configuration Guide
echo [4] Back to Main Menu
echo.
set /p doc_choice="Select option (1-4): "

if "%doc_choice%"=="1" (
    type README-GPU-AI.md | more
    pause
)
if "%doc_choice%"=="2" (
    start http://localhost:8084/
)
if "%doc_choice%"=="3" (
    echo.
    echo Configuration Files:
    echo   - START-GPU-LEGAL-AI-8084.bat - Main startup script
    echo   - docker-compose.yml - Docker configuration
    echo   - optimize-gpu-legal-ai.ps1 - GPU optimization
    echo   - monitor-gpu-ai.ps1 - Real-time monitoring
    echo.
    echo Environment Variables:
    echo   MAX_CONCURRENCY=3 (GPU request limit)
    echo   GPU_MEMORY_LIMIT_MB=6000 (VRAM allocation)
    echo   MODEL_CONTEXT=4096 (Token window)
    echo   TEMPERATURE=0.2 (Model creativity)
    echo.
    pause
)
if "%doc_choice%"=="4" goto menu

goto view_docs

:invalid_choice
echo.
echo Invalid choice. Please try again.
timeout /t 2 /nobreak >nul
goto menu

:exit_app
cls
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    SHUTTING DOWN                            ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Thank you for using GPU-Accelerated Legal AI!
echo.
timeout /t 2 /nobreak >nul
exit