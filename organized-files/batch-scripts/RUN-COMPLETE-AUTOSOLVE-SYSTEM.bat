@echo off
:: RUN-COMPLETE-AUTOSOLVE-SYSTEM.bat
:: Complete integration of Enhanced RAG V2 with Autosolve, Orchestrator, and all services

echo ============================================================
echo   ENHANCED RAG V2 + AUTOSOLVE COMPLETE SYSTEM
echo   With TypeScript Error Fixing, AI Recommendations,
echo   Ollama Summaries, and Full Orchestration
echo ============================================================
echo.

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo This script requires Administrator privileges.
    echo Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

:: Set environment variables
set NODE_ENV=production
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_ai_rag?sslmode=disable
set REDIS_URL=localhost:6379
set RABBITMQ_URL=amqp://guest:guest@localhost:5672/
set OLLAMA_URL=http://localhost:11434
set ENHANCED_RAG_PORT=8097
set AGGREGATE_PORT=8123
set AUTO_FIX_ENABLED=true
set HTTP_STATUS_PORT=9099

echo.
echo [1/10] Starting Core Services...
echo ================================

:: Start PostgreSQL
echo Starting PostgreSQL...
net start postgresql-x64-14 2>nul || echo PostgreSQL already running

:: Start Redis
echo Starting Redis...
cd C:\Redis
start /min redis-server.exe
cd C:\Users\james\Desktop\deeds-web\deeds-web-app

:: Start RabbitMQ
echo Starting RabbitMQ...
net start RabbitMQ 2>nul || echo RabbitMQ already running

:: Start Ollama
echo Starting Ollama...
start /min ollama serve

timeout /t 5 /nobreak >nul

echo.
echo [2/10] Building Enhanced RAG V2...
echo ==================================
cd go-microservice
go build -o bin\enhanced-rag-v2-local.exe cmd\enhanced-rag-v2-local\main.go 2>nul
cd ..

echo.
echo [3/10] Starting Enhanced RAG V2 Service...
echo ==========================================
start /min "Enhanced RAG V2" go-microservice\bin\enhanced-rag-v2-local.exe

timeout /t 3 /nobreak >nul

echo.
echo [4/10] Starting Aggregate Server...
echo ====================================
start /min "Aggregate Server" node scripts\aggregate-server.cjs

echo.
echo [5/10] Starting Recommendation Service...
echo ==========================================
start /min "Recommendation Service" node scripts\recommendation-service.mjs

echo.
echo [6/10] Starting Error Processor Daemon...
echo ==========================================
start /min "Error Processor" cmd /c "npm run check:auto:daemon"

timeout /t 3 /nobreak >nul

echo.
echo [7/10] Checking Environment...
echo ===============================
call npm run orchestrator:check-env

echo.
echo [8/10] Building CUDA Worker (if needed)...
echo ===========================================
if exist cuda-worker (
    call npm run orchestrator:build-cuda
) else (
    echo CUDA worker directory not found, skipping...
)

echo.
echo [9/10] Running Initial TypeScript Check...
echo ===========================================
echo Running npm run check:full:recommend...
call npm run check:full:recommend

timeout /t 2 /nobreak >nul

echo.
echo [10/10] Starting Autosolve Loop...
echo ===================================
echo.
echo The system will now:
echo   1. Detect TypeScript errors
echo   2. Generate AI recommendations
echo   3. Apply automatic fixes
echo   4. Use Ollama for summaries
echo   5. Index and save to PostgreSQL
echo   6. Continue improving until convergence
echo.
echo Press Ctrl+C to stop the autosolve loop
echo.

:: Display service status
echo ============================================================
echo   SYSTEM STATUS
echo ============================================================
echo.
echo Services Running:
echo   [✓] PostgreSQL       - Port 5432
echo   [✓] Redis            - Port 6379
echo   [✓] RabbitMQ         - Port 5672
echo   [✓] Ollama           - Port 11434
echo   [✓] Enhanced RAG V2  - Port 8097
echo   [✓] Aggregate Server - Port 8123
echo   [✓] Error Processor  - Port 9099
echo.
echo Endpoints Available:
echo   API:        http://localhost:8097
echo   Aggregate:  http://localhost:8123/aggregate
echo   Health:     http://localhost:8097/health
echo   RabbitMQ:   http://localhost:15672
echo.
echo Commands:
echo   Trigger Autosolve: curl -X POST http://localhost:8123/autosolve/trigger
echo   Check Errors:      curl http://localhost:8123/errors
echo   Get Summary:       curl http://localhost:8123/aggregate
echo   Ollama Summary:    curl -X POST http://localhost:8123/ollama/summary
echo.
echo ============================================================
echo.

:: Start the autosolve loop
echo Starting continuous improvement loop...
call npm run autosolve:loop

echo.
echo ============================================================
echo   AUTOSOLVE COMPLETE
echo ============================================================
echo.
echo Check the results:
echo   - Logs: logs\autosolve-history.jsonl
echo   - Errors: logs\error-cache.json
echo   - Backups: backups\
echo   - Declarations: src\auto-decls.d.ts
echo.
pause
