@echo off
:: Unified System Architecture Startup Script
:: Orchestrates the complete AI-powered legal system with GPU acceleration

setlocal enabledelayedexpansion

:: Colors for better output
set "GREEN=[32m"
set "BLUE=[34m"
set "YELLOW=[33m"
set "RED=[31m"
set "RESET=[0m"

echo %BLUE%=========================================%RESET%
echo %BLUE%🚀 Legal AI Unified System Startup%RESET%
echo %BLUE%=========================================%RESET%
echo.

:: Configuration
set "PROJECT_ROOT=%~dp0.."
set "GO_SERVICE_PORT=50051"
set "NODE_GPU_PORT=50052"
set "SVELTEKIT_PORT=5173"
set "POSTGRES_PORT=5432"
set "REDIS_PORT=6379"
set "QDRANT_PORT=6333"
set "NEO4J_PORT=7687"
set "OLLAMA_PORT=11434"

:: Check prerequisites
echo %YELLOW%📋 Checking prerequisites...%RESET%
call :check_prerequisite "node" "Node.js"
call :check_prerequisite "go" "Go"
call :check_prerequisite "docker" "Docker"
call :check_prerequisite "npm" "NPM"

:: Start infrastructure services
echo.
echo %BLUE%🏗️ Starting infrastructure services...%RESET%

:: PostgreSQL with pgvector
echo %YELLOW%Starting PostgreSQL with pgvector...%RESET%
start /b "" "C:\Program Files\PostgreSQL\17\bin\pg_ctl" -D "C:\Program Files\PostgreSQL\17\data" start
timeout /t 3 /nobreak >nul

:: Redis
echo %YELLOW%Starting Redis...%RESET%
start /b "" "%PROJECT_ROOT%\redis-windows\redis-server.exe"
timeout /t 2 /nobreak >nul

:: Qdrant vector database
echo %YELLOW%Starting Qdrant...%RESET%
start /b "" "%PROJECT_ROOT%\qdrant-windows\qdrant.exe"
timeout /t 3 /nobreak >nul

:: Neo4j graph database
echo %YELLOW%Starting Neo4j...%RESET%
call "%PROJECT_ROOT%\scripts\start-neo4j.bat"
timeout /t 5 /nobreak >nul

:: Ollama for local LLM
echo %YELLOW%Starting Ollama...%RESET%
start /b "" ollama serve
timeout /t 3 /nobreak >nul

:: Wait for infrastructure to be ready
echo.
echo %YELLOW%⏳ Waiting for infrastructure services...%RESET%
call :wait_for_service "localhost" %POSTGRES_PORT% "PostgreSQL"
call :wait_for_service "localhost" %REDIS_PORT% "Redis"
call :wait_for_service "localhost" %QDRANT_PORT% "Qdrant"
call :wait_for_service "localhost" %NEO4J_PORT% "Neo4j"
call :wait_for_service "localhost" %OLLAMA_PORT% "Ollama"

:: Start AI microservices
echo.
echo %BLUE%🧠 Starting AI microservices...%RESET%

:: Go SIMD Service
echo %YELLOW%Starting Go SIMD Service...%RESET%
cd /d "%PROJECT_ROOT%\services\go-simd-service"
start /b "" cmd /c "go mod tidy && go run main.go"
cd /d "%PROJECT_ROOT%"
timeout /t 5 /nobreak >nul

:: Node.js GPU Service
echo %YELLOW%Starting Node.js GPU Service...%RESET%
cd /d "%PROJECT_ROOT%\services\node-gpu-service"
start /b "" cmd /c "npm install && npm run dev"
cd /d "%PROJECT_ROOT%"
timeout /t 10 /nobreak >nul

:: Wait for AI services
echo %YELLOW%⏳ Waiting for AI microservices...%RESET%
call :wait_for_service "localhost" %GO_SERVICE_PORT% "Go SIMD Service"
call :wait_for_service "localhost" %NODE_GPU_PORT% "Node.js GPU Service"

:: Initialize database schema and data
echo.
echo %BLUE%🗄️ Initializing database...%RESET%
echo %YELLOW%Setting up database schema...%RESET%
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -f "%PROJECT_ROOT%\database\schema.sql"

echo %YELLOW%Loading sample data...%RESET%
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -f "%PROJECT_ROOT%\database\sample-data.sql"

:: Start SvelteKit frontend
echo.
echo %BLUE%🎨 Starting SvelteKit frontend...%RESET%
echo %YELLOW%Installing dependencies...%RESET%
npm install

echo %YELLOW%Starting development server...%RESET%
start /b "" cmd /c "npm run dev -- --port %SVELTEKIT_PORT%"

:: Wait for SvelteKit to be ready
echo %YELLOW%⏳ Waiting for SvelteKit...%RESET%
call :wait_for_service "localhost" %SVELTEKIT_PORT% "SvelteKit"

:: System health checks
echo.
echo %BLUE%🏥 Performing system health checks...%RESET%
call :health_check "http://localhost:%SVELTEKIT_PORT%" "SvelteKit Frontend"
call :health_check "http://localhost:%GO_SERVICE_PORT%/health" "Go SIMD Service"
call :health_check "http://localhost:%NODE_GPU_PORT%/health" "Node.js GPU Service"

:: Display system status
echo.
echo %GREEN%=========================================%RESET%
echo %GREEN%🎉 System startup complete!%RESET%
echo %GREEN%=========================================%RESET%
echo.
echo %BLUE%📊 Service Endpoints:%RESET%
echo   Frontend:      http://localhost:%SVELTEKIT_PORT%
echo   Go SIMD:       localhost:%GO_SERVICE_PORT%
echo   Node.js GPU:   localhost:%NODE_GPU_PORT%
echo   PostgreSQL:    localhost:%POSTGRES_PORT%
echo   Redis:         localhost:%REDIS_PORT%
echo   Qdrant:        localhost:%QDRANT_PORT%
echo   Neo4j:         localhost:%NEO4J_PORT%
echo   Ollama:        localhost:%OLLAMA_PORT%
echo.
echo %BLUE%🔍 Management URLs:%RESET%
echo   GPU Processing: http://localhost:%SVELTEKIT_PORT%/gpu-orchestrator
echo   Database Admin: http://localhost:%SVELTEKIT_PORT%/db-admin
echo   System Monitor: http://localhost:%SVELTEKIT_PORT%/system-monitor
echo   API Explorer:   http://localhost:%SVELTEKIT_PORT%/api-docs
echo.
echo %YELLOW%Press Ctrl+C to shutdown all services%RESET%
echo.

:: Monitor services
call :monitor_services

goto :eof

:: Functions
:check_prerequisite
where %1 >nul 2>&1
if !errorlevel! neq 0 (
    echo %RED%❌ %2 not found. Please install %2 and try again.%RESET%
    pause
    exit /b 1
) else (
    echo %GREEN%✅ %2 found%RESET%
)
goto :eof

:wait_for_service
set "host=%1"
set "port=%2"
set "service_name=%3"
set "max_attempts=30"
set "attempt=0"

:wait_loop
set /a attempt+=1
netstat -an | find ":%port%" | find "LISTENING" >nul 2>&1
if !errorlevel! equ 0 (
    echo %GREEN%✅ %service_name% is ready%RESET%
    goto :eof
)

if !attempt! geq !max_attempts! (
    echo %RED%❌ %service_name% failed to start%RESET%
    goto :eof
)

timeout /t 2 /nobreak >nul
goto wait_loop

:health_check
set "url=%1"
set "service_name=%2"
curl -f -s "%url%" >nul 2>&1
if !errorlevel! equ 0 (
    echo %GREEN%✅ %service_name% health check passed%RESET%
) else (
    echo %YELLOW%⚠️ %service_name% health check failed (may not have HTTP endpoint)%RESET%
)
goto :eof

:monitor_services
echo %BLUE%📡 Monitoring system (press Ctrl+C to stop)...%RESET%
:monitor_loop
timeout /t 30 /nobreak >nul
echo %YELLOW%[%date% %time%] System running...%RESET%
goto monitor_loop