@echo off
REM Legal AI System - Windows 10/11 Startup (Docker-free)
REM Starts Go service (8084), MCP servers, and SvelteKit dev

setlocal ENABLEDELAYEDEXPANSION

echo ========================================
echo Legal AI System - Windows Startup (No Docker)
echo ========================================
echo.

REM Resolve script dir
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"

REM 1) Start Go microservice on 8084 in a new window
echo [1/5] Starting Go microservice (port 8084)...
set "GO_DIR=%SCRIPT_DIR%go-microservice"
set "GO_SERVER_PORT=8084"
set "OLLAMA_HOST=http://localhost:11434"
set "MODEL_NAME=gemma3-legal:latest"

if exist "%GO_DIR%\main.go" (
    start "GO-LEGAL-AI" cmd /c "cd /d \"%GO_DIR%\" && set PORT=8084 && set OLLAMA_HOST=%OLLAMA_HOST% && set MODEL_NAME=%MODEL_NAME% && go run main.go"
) else (
    echo ⚠️  Go service not found at %GO_DIR%\main.go
)
echo.

REM 2) Start MCP servers (Context7, Memory, Postgres) if tasks exist
echo [2/5] Starting MCP servers (background tasks)...
REM These are launched via VS Code tasks; optional if not configured.
REM To start from VS Code: Run task "Start MCP Servers"
echo   - Use VS Code Task: Start MCP Servers
echo.

REM 3) Start SvelteKit dev (port 5173) in a new window
echo [3/5] Starting SvelteKit dev server (port 5173)...
set "SK_DIR=%SCRIPT_DIR%sveltekit-frontend"
if exist "%SK_DIR%\package.json" (
    start "SvelteKit Dev" cmd /c "cd /d \"%SK_DIR%\" && set GO_SERVICE_URL=http://localhost:8084 && npm run dev"
) else (
    echo ⚠️  SvelteKit frontend not found at %SK_DIR%
)
echo.

REM NOTE: User instruction per request
echo please downloal local llm and put in this directory
echo.

REM 4) Quick health checks (best-effort)
echo [4/5] Checking service health...
REM Try Go health
for /l %%i in (1,1,5) do (
    curl -s -f http://localhost:8084/api/health >nul 2>&1 && (
        echo   ✅ Go service healthy (http://localhost:8084)
        goto :done_health
    )
    timeout /t 2 /nobreak >nul
)
echo   ⚠️  Go health not responding yet: http://localhost:8084/api/health
:done_health

REM Try SvelteKit
for /l %%i in (1,1,5) do (
    curl -s -f http://localhost:5173/ >nul 2>&1 && (
        echo   ✅ SvelteKit dev reachable (http://localhost:5173)
        goto :done_sk
    )
    timeout /t 2 /nobreak >nul
)
echo   ⚠️  SvelteKit not responding yet: http://localhost:5173
:done_sk
echo.

REM 5) Open VS Code (optional)
echo [5/5] Opening VS Code (Legal AI profile)...
code . --profile "Legal AI" >nul 2>&1
echo.

echo ========================================
echo Legal AI System Startup Triggered (No Docker)
echo ========================================
echo.
echo Endpoints:
echo - Go Service Health:   http://localhost:8084/api/health
echo - Go Metrics:          http://localhost:8084/api/metrics
echo - SvelteKit App:       http://localhost:5173
echo - GPU Status (proxy):  http://localhost:5173/api/gpu-status
echo.
echo Tip: In VS Code, run Task: "Start MCP Servers" to launch Context7/Memory/Postgres MCP.
echo.
popd
pause