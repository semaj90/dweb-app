@echo off
REM Complete Legal AI System Startup with Fixes
REM Automatically fixes configuration and starts all services

cls
echo ======================================
echo   Legal AI System - Complete Startup
echo ======================================
echo.

REM Fix configuration first
echo [1/7] Fixing configuration...
powershell -ExecutionPolicy Bypass -File scripts\fix-all-services.ps1 -AutoFix

REM Start PostgreSQL
echo [2/7] Starting PostgreSQL...
pg_ctl start -D "%PGDATA%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       [OK] PostgreSQL started
) else (
    echo       [WARN] PostgreSQL may already be running
)
timeout /t 2 /nobreak >nul

REM Start Redis
echo [3/7] Starting Redis...
if exist "redis\redis-server.exe" (
    start /B redis\redis-server.exe
    echo       [OK] Redis started
) else if exist "redis-windows\redis-server.exe" (
    start /B redis-windows\redis-server.exe
    echo       [OK] Redis started
) else (
    echo       [SKIP] Redis not found
)

REM Start MinIO
echo [4/7] Starting MinIO...
if exist "minio.exe" (
    if not exist "minio-data" mkdir minio-data
    start /B minio.exe server ./minio-data --console-address :9001
    echo       [OK] MinIO started (Console: http://localhost:9001)
) else (
    echo       [SKIP] MinIO not found
)

REM Start Neo4j
echo [5/7] Starting Neo4j...
if exist "neo4j-community-5.23.0\bin\neo4j.bat" (
    start /B neo4j-community-5.23.0\bin\neo4j.bat console >nul 2>&1
    echo       [OK] Neo4j starting (Browser: http://localhost:7474)
) else if exist "neo4j-community-5.21.2\bin\neo4j.bat" (
    start /B neo4j-community-5.21.2\bin\neo4j.bat console >nul 2>&1
    echo       [OK] Neo4j starting (Browser: http://localhost:7474)
) else (
    echo       [SKIP] Neo4j not found
)

REM Start Ollama
echo [6/7] Starting Ollama...
where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start /B ollama serve
    timeout /t 3 /nobreak >nul
    echo       [OK] Ollama started
    
    REM Load the model
    ollama list | findstr "gemma3-legal" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo       Loading gemma3-legal model...
        ollama pull gemma3-legal:latest >nul 2>&1
    )
) else (
    echo       [ERROR] Ollama not installed
    echo       Please install from: https://ollama.ai/download
)

REM Start Enhanced RAG if Go is available
echo [7/7] Starting Enhanced RAG...
where go >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    if exist "enhanced-rag-som-system.go" (
        start /B go run enhanced-rag-som-system.go
        echo       [OK] Enhanced RAG starting
    ) else if exist "enhanced-rag-som-system.exe" (
        start /B enhanced-rag-som-system.exe
        echo       [OK] Enhanced RAG starting
    ) else (
        echo       [SKIP] Enhanced RAG not found
    )
) else (
    echo       [SKIP] Go not installed
)

echo.
echo ======================================
echo   Starting Frontend Application...
echo ======================================
echo.

REM Wait for services to stabilize
timeout /t 5 /nobreak >nul

REM Start the frontend
cd sveltekit-frontend
echo Starting SvelteKit frontend...
echo.
echo Services Status:
echo   PostgreSQL: http://localhost:5432
echo   Ollama:     http://localhost:11434
echo   Neo4j:      http://localhost:7474
echo   Redis:      http://localhost:6379
echo   MinIO:      http://localhost:9001
echo   RAG:        http://localhost:8094
echo.
echo Frontend will be available at: http://localhost:5173
echo Admin Dashboard: Open admin-dashboard.html
echo.
echo Press Ctrl+C to stop all services
echo.

npm run dev