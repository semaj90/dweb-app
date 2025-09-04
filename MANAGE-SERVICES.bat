@echo off
REM ================================================================================
REM LEGAL AI PLATFORM - SERVICE MANAGEMENT UTILITY
REM ================================================================================

if "%1"=="" goto :show_menu
if "%1"=="start" goto :start_services
if "%1"=="stop" goto :stop_services
if "%1"=="status" goto :check_status
if "%1"=="restart" goto :restart_services
if "%1"=="build" goto :build_services
goto :show_help

:show_menu
echo.
echo ================================================================================
echo LEGAL AI PLATFORM - SERVICE MANAGER
echo ================================================================================
echo.
echo Available commands:
echo   1. start    - Start all services
echo   2. stop     - Stop all services
echo   3. status   - Check service status
echo   4. restart  - Restart all services
echo   5. build    - Build Go services
echo.
echo Usage: %0 [command]
echo.
pause
goto :end

:start_services
echo Starting all Legal AI services...
call COMPLETE-WIRED-STARTUP.bat
goto :end

:stop_services
echo Stopping all Legal AI services...
echo Stopping SvelteKit...
taskkill /f /im node.exe 2>nul
echo Stopping Go services...
taskkill /f /im upload-service.exe 2>nul
taskkill /f /im grpc-server.exe 2>nul
taskkill /f /im main-service.exe 2>nul
taskkill /f /im summarizer-service.exe 2>nul
taskkill /f /im enhanced-rag.exe 2>nul
taskkill /f /im load-balancer.exe 2>nul
echo Stopping Ollama...
taskkill /f /im ollama.exe 2>nul
echo Stopping MinIO...
taskkill /f /im minio.exe 2>nul
echo Stopping Qdrant...
taskkill /f /im qdrant.exe 2>nul
echo Stopping Redis...
taskkill /f /im redis-server.exe 2>nul
echo All services stopped.
goto :end

:check_status
echo.
echo ================================================================================
echo LEGAL AI PLATFORM - SERVICE STATUS
echo ================================================================================
echo.

echo Database Services:
echo ------------------
net start | findstr "postgresql" >nul && echo ✓ PostgreSQL: Running || echo ✗ PostgreSQL: Stopped
tasklist | findstr "redis-server" >nul && echo ✓ Redis: Running || echo ✗ Redis: Stopped
tasklist | findstr "qdrant" >nul && echo ✓ Qdrant: Running || echo ✗ Qdrant: Stopped
sc query neo4j | findstr "RUNNING" >nul && echo ✓ Neo4j: Running || echo ✗ Neo4j: Stopped

echo.
echo AI Services:
echo ------------
tasklist | findstr "ollama" >nul && echo ✓ Ollama: Running || echo ✗ Ollama: Stopped
tasklist | findstr "minio" >nul && echo ✓ MinIO: Running || echo ✗ MinIO: Stopped

echo.
echo Go Microservices:
echo ----------------
tasklist | findstr "main-service" >nul && echo ✓ Main Service: Running || echo ✗ Main Service: Stopped
tasklist | findstr "grpc-server" >nul && echo ✓ gRPC Server: Running || echo ✗ gRPC Server: Stopped
tasklist | findstr "upload-service" >nul && echo ✓ Upload Service: Running || echo ✗ Upload Service: Stopped
tasklist | findstr "summarizer-service" >nul && echo ✓ Summarizer Service: Running || echo ✗ Summarizer Service: Stopped
tasklist | findstr "enhanced-rag" >nul && echo ✓ Enhanced RAG: Running || echo ✗ Enhanced RAG: Stopped
tasklist | findstr "load-balancer" >nul && echo ✓ Load Balancer: Running || echo ✗ Load Balancer: Stopped

echo.
echo Frontend:
echo ---------
tasklist | findstr "node.exe" >nul && echo ✓ SvelteKit: Running || echo ✗ SvelteKit: Stopped

echo.
echo API Health Checks:
echo ------------------
curl -s http://localhost:11434/api/tags >nul 2>&1 && echo ✓ Ollama API: Responding || echo ✗ Ollama API: Not responding
curl -s http://localhost:6333 >nul 2>&1 && echo ✓ Qdrant API: Responding || echo ✗ Qdrant API: Not responding
curl -s http://localhost:8080/health >nul 2>&1 && echo ✓ Main Service API: Responding || echo ✗ Main Service API: Not responding
curl -s http://localhost:8093/health >nul 2>&1 && echo ✓ Upload Service API: Responding || echo ✗ Upload Service API: Not responding
curl -s http://localhost:5173 >nul 2>&1 && echo ✓ SvelteKit: Responding || echo ✗ SvelteKit: Not responding

echo.
goto :end

:restart_services
echo Restarting all Legal AI services...
call %0 stop
timeout /t 3 /nobreak >nul
call %0 start
goto :end

:build_services
echo Building Go microservices...
cd go-microservice

echo Building Upload Service...
go build -o bin/upload-service.exe ./cmd/upload-service/main.go

echo Building gRPC Server...
go build -o bin/grpc-server.exe ./cmd/grpc-server/main.go

echo Building Main Service...
go build -o bin/main-service.exe ./main.go

echo Building Summarizer Service...
go build -o bin/summarizer-service.exe ./cmd/summarizer-service/main.go

echo Attempting to build Enhanced RAG Service...
go build -o bin/enhanced-rag.exe ./cmd/enhanced-rag/main.go 2>nul || echo Enhanced RAG build failed - using alternatives

echo Attempting to build Load Balancer...
go build -o bin/load-balancer.exe ./load-balancer.go 2>nul || echo Load Balancer build failed - optional service

cd ..
echo Build complete!
goto :end

:show_help
echo Usage: %0 [start|stop|status|restart|build]
goto :end

:end