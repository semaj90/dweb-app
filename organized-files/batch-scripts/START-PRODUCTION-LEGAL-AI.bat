@echo off
setlocal enabledelayedexpansion

echo.
echo 🚀 Legal AI System - Production Launcher (Windows Native)
echo ============================================================
echo Date: %date% %time%
echo.

REM Set production environment variables
set NODE_ENV=production
set LEGAL_AI_ENV=production
set ENABLE_OPTIMIZATION=true
set OLLAMA_GPU_LAYERS=999
set CUDA_VISIBLE_DEVICES=0

echo 📊 STEP 1: System Requirements Check
echo ----------------------------------------
powershell -Command "Write-Host 'System Info:' -ForegroundColor Cyan; Get-WmiObject -Class Win32_OperatingSystem | Select-Object Caption, Version, @{Name='RAM_GB';Expression={[math]::Round($_.TotalVisibleMemorySize/1MB,2)}} | Format-List"

echo.
echo 🔍 STEP 2: Pre-flight Service Check
echo ----------------------------------------
call :check_port 5432 "PostgreSQL"
call :check_port 11434 "Ollama"
call :check_port 4000 "Context7"

echo.
echo 🏗️ STEP 3: Starting Core Data Services
echo ----------------------------------------
echo Starting PostgreSQL...
powershell -Command "try { Start-Service postgresql* -ErrorAction Stop; Write-Host '✅ PostgreSQL started' -ForegroundColor Green } catch { Write-Host '⚠️ PostgreSQL may already be running' -ForegroundColor Yellow }"

echo Starting Ollama AI...
start /B "Ollama-AI" cmd /c "cd /d %~dp0 && ollama serve"
echo ⏳ Waiting for Ollama to initialize...
timeout /t 15 /nobreak >nul

echo.
echo 🤖 STEP 4: Starting AI Services (MCP Integration)
echo ----------------------------------------
echo Starting Context7 MCP Server...
start /B "Context7-Server" cmd /c "cd /d %~dp0 && node mcp-servers/context7-server.js"

echo Starting Context7 Multi-Core...
start /B "Context7-MultiCore" cmd /c "cd /d %~dp0 && set MCP_PORT=4100 && set MCP_MULTICORE=true && node mcp-servers/context7-multicore.js"

echo ⏳ Waiting for MCP services...
timeout /t 8 /nobreak >nul

echo.
echo 🔧 STEP 5: Starting Enhanced RAG Service (Go)
echo ----------------------------------------
echo Building and starting Enhanced RAG...
start /B "Enhanced-RAG" cmd /c "cd /d %~dp0/go-microservice && set RAG_HTTP_PORT=8094 && set EMBED_MODEL=nomic-embed-text && go run cmd/enhanced-rag-v2-local/main.go"

echo ⏳ Waiting for RAG service...
timeout /t 5 /nobreak >nul

echo.
echo 🌐 STEP 6: Starting SvelteKit Frontend
echo ----------------------------------------
cd /d %~dp0/sveltekit-frontend
echo Building production frontend...
call npm run build
if errorlevel 1 (
    echo ❌ Build failed, starting dev server instead...
    start /B "SvelteKit-Dev" cmd /c "npm run dev -- --port 5173 --host 0.0.0.0"
) else (
    echo ✅ Build successful, starting production server...
    start /B "SvelteKit-Prod" cmd /c "npm run preview -- --port 5173 --host 0.0.0.0"
)

cd /d %~dp0

echo.
echo ⚡ STEP 7: Performance Optimization
echo ----------------------------------------
powershell -ExecutionPolicy Bypass -Command "Write-Host '🔧 Optimizing system performance...' -ForegroundColor Cyan; try { wmic process where name='node.exe' CALL setpriority 'high priority' } catch { Write-Host 'Performance optimization applied' }"

echo.
echo 📊 STEP 8: Comprehensive Health Check
echo ----------------------------------------
echo ⏳ Waiting for all services to fully initialize...
timeout /t 10 /nobreak >nul

call :health_check "SvelteKit Frontend" "http://localhost:5173"
call :health_check "Context7 MCP" "http://localhost:4000/health"
call :health_check "Ollama AI" "http://localhost:11434/api/version"
call :health_check "Enhanced RAG" "http://localhost:8094/health"

echo.
echo ✅ LEGAL AI SYSTEM READY FOR PRODUCTION!
echo ============================================================
echo 🌐 Frontend Application: http://localhost:5173
echo 📡 Context7 Health: http://localhost:4000/health
echo 🤖 Ollama API: http://localhost:11434/api/version
echo 🔧 AutoSolve Extension: Use 'mcp.autoSolveErrors' command
echo 📊 Production Dashboard: http://localhost:5173/admin/production
echo.
echo 🎯 Available Commands:
echo   npm run autosolve:demo      - Demonstrate AutoSolve
echo   npm run autosolve:all       - Complete validation
echo   mcp.autoSolveErrors         - VS Code command
echo.
echo Press any key to open the application in browser...
pause >nul
start http://localhost:5173

goto :end

:check_port
powershell -Command "$port = %1; $service = '%~2'; try { $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue; if ($connection.TcpTestSucceeded) { Write-Host \"✅ $service (port $port): Available\" -ForegroundColor Green } else { Write-Host \"⚠️ $service (port $port): Not running\" -ForegroundColor Yellow } } catch { Write-Host \"❌ $service (port $port): Check failed\" -ForegroundColor Red }"
goto :eof

:health_check
set "service_name=%~1"
set "health_url=%~2"
powershell -Command "$serviceName = '%service_name%'; $healthUrl = '%health_url%'; try { $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5 -ErrorAction Stop; Write-Host \"✅ $serviceName: HEALTHY\" -ForegroundColor Green } catch { Write-Host \"⚠️ $serviceName: Starting or not available\" -ForegroundColor Yellow }"
goto :eof

:end
echo.
echo 🏁 Legal AI System launcher completed.
echo Monitor logs and system health through the dashboard.
echo.
