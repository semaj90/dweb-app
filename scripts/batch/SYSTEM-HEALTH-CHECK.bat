@echo off
title AI Legal App - System Health Check

echo.
echo ===================================================
echo  AI Legal App - System Health Check
echo ===================================================
echo.
echo This script will check your environment to ensure all components are ready.
echo Please run this from the root of your project directory.
echo.

set "OLLAMA_CONTAINER=deeds-ollama-gpu"
set "SVELTE_CONTAINER=sveltekit-app"
set "COMPOSE_FILE=docker-compose-gpu.yml"
set "OLLAMA_VOLUME=deeds-web-app_ollama_data"

:CHECK_NVIDIA
echo [1/6] Checking Host System (NVIDIA GPU)...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo   [X] ERROR: NVIDIA drivers not found. Please install the latest drivers for your RTX 3060.
    goto :END
)
echo   [+] SUCCESS: NVIDIA GPU and drivers detected.
echo.

:CHECK_DOCKER
echo [2/6] Checking Docker Environment...
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo   [X] ERROR: Docker is not running. Please start Docker Desktop.
    goto :END
)
echo   [+] SUCCESS: Docker is running.
echo.

:CHECK_FILES
echo [3/6] Checking Project Files...
if not exist "%COMPOSE_FILE%" (
    echo   [X] ERROR: %COMPOSE_FILE% not found.
    goto :END
)
if not exist "sveltekit-frontend" (
    echo   [X] ERROR: 'sveltekit-frontend' directory not found.
    goto :END
)
if not exist "models" (
    echo   [X] ERROR: 'models' directory not found. Your GGUF model should be in here.
    goto :END
)
echo   [+] SUCCESS: Essential project files and directories are present.
echo.

:CHECK_UNHEALTHY
echo [4/6] Checking for Unhealthy Services (Corruption Check)...
docker ps --format "table {{.Names}}\t{{.Status}}" | findstr "unhealthy" >nul
if %errorlevel% equ 0 (
    echo   [!] WARNING: Unhealthy containers detected. This may indicate corrupted models.
    echo   [!] Performing a deep clean to fix the issue...
    echo   [!] Stopping all services...
    docker-compose -f %COMPOSE_FILE% down >nul 2>&1
    echo   [!] Removing potentially corrupted volumes...
    docker volume prune -f >nul 2>&1
    echo   [!] Deep clean complete. Services will be restarted.
) else (
    echo   [+] SUCCESS: No unhealthy services detected.
)
echo.

:CHECK_CONTAINERS
echo [5/6] Checking Running Services...
docker ps --format "{{.Names}}" | findstr "%OLLAMA_CONTAINER%" >nul
if %errorlevel% neq 0 (
    echo   [!] INFO: Ollama container (%OLLAMA_CONTAINER%) is not running.
    echo   [!] Attempting to start services...
    docker-compose -f %COMPOSE_FILE% up -d --build
    echo   [!] Waiting for services to initialize...
    timeout /t 20 >nul
    docker ps --format "{{.Names}}" | findstr "%OLLAMA_CONTAINER%" >nul
    if %errorlevel% neq 0 (
        echo   [X] ERROR: Failed to start the Ollama container.
        goto :END
    )
)
echo   [+] SUCCESS: Ollama container is running.
echo.

:CHECK_HEALTH
echo [6/6] Checking Service Health...
echo   [-] Testing Ollama API connectivity...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo   [X] WARNING: Ollama API not responding. Service may still be starting up.
) else (
    echo   [+] SUCCESS: Ollama API is responsive.
)
echo.

echo ===================================================
echo  ✅ System Check Complete: All systems nominal!
echo ===================================================
echo.
echo You can access your application at: http://localhost:5173/chat
echo.
goto :END

:END
pause
