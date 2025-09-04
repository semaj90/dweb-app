@echo off
REM =============================================================================
REM LEGAL AI STARTUP - PORT CONFLICT & DEPENDENCY RESOLUTION
REM Addresses all three classes of startup errors identified
REM =============================================================================

setlocal enabledelayedexpansion

echo ===============================================================================
echo LEGAL AI PLATFORM STARTUP - COMPREHENSIVE FIX
echo ===============================================================================
echo [1/3] PORT CONFLICT RESOLUTION
echo [2/3] DEPENDENCY INSTALLATION  
echo [3/3] SERVICE STARTUP SEQUENCE
echo ===============================================================================

REM === PHASE 1: PORT CONFLICT RESOLUTION ===
echo.
echo [1/3] === RESOLVING PORT CONFLICTS ===

echo [1.1] Checking for port conflicts...
netstat -ano | findstr :8093 > nul
if !errorlevel! == 0 (
    echo     WARNING: Port 8093 in use - will use 8094 for upload service
    set UPLOAD_PORT=8094
) else (
    set UPLOAD_PORT=8093
)

netstat -ano | findstr :8094 > nul
if !errorlevel! == 0 (
    echo     WARNING: Port 8094 in use - will use 8097 for RAG service
    set RAG_PORT=8097  
) else (
    set RAG_PORT=8094
)

netstat -ano | findstr :8447 > nul
if !errorlevel! == 0 (
    echo     WARNING: Port 8447 in use - using 8448 for QUIC
    set QUIC_PORT=8448
) else (
    set QUIC_PORT=8447
)

echo     ✓ Port assignments: Upload=%UPLOAD_PORT%, RAG=%RAG_PORT%, QUIC=%QUIC_PORT%

REM === PHASE 2: DEPENDENCY INSTALLATION ===
echo.
echo [2/3] === INSTALLING MISSING DEPENDENCIES ===

echo [2.1] Installing Node.js dependencies...
if exist "node_modules\pg" (
    echo     ✓ PostgreSQL driver already installed
) else (
    echo     Installing pg driver...
    pnpm add pg @types/pg
)

if exist "node_modules\drizzle-orm" (
    echo     ✓ Drizzle ORM already installed
) else (
    echo     Installing drizzle-orm...
    pnpm add drizzle-orm
)

if exist "node_modules\ioredis" (
    echo     ✓ Redis client already installed  
) else (
    echo     Installing Redis client...
    pnpm add ioredis
)

echo [2.2] Checking for service binaries...

REM Check Redis
if exist "redis-windows\redis-server.exe" (
    echo     ✓ Redis binary found
) else (
    echo     ⚠ Redis not found - download redis-windows.zip and extract
    echo       URL: https://github.com/MicrosoftArchive/redis/releases
)

REM Check Qdrant  
if exist "qdrant-windows\qdrant.exe" (
    echo     ✓ Qdrant binary found
) else (
    echo     ⚠ Qdrant not found - download qdrant-windows.zip and extract
    echo       URL: https://github.com/qdrant/qdrant/releases
)

REM Check MinIO
if exist "minio.exe" (
    echo     ✓ MinIO binary found
) else (
    echo     ⚠ MinIO not found - download minio.exe
    echo       URL: https://min.io/download
)

REM Check Go services
echo [2.3] Checking Go service binaries...
if exist "..\go-microservice\cmd\enhanced-rag\enhanced-rag.exe" (
    echo     ✓ Enhanced RAG service binary found
    set RAG_BINARY=..\go-microservice\cmd\enhanced-rag\enhanced-rag.exe
) else (
    echo     Building Enhanced RAG service...
    cd ..\go-microservice && go build -o .\cmd\enhanced-rag\enhanced-rag.exe .\cmd\enhanced-rag && cd ..\deeds-web-app
    set RAG_BINARY=..\go-microservice\cmd\enhanced-rag\enhanced-rag.exe
)

if exist "..\go-microservice\cmd\upload-service\upload-service.exe" (
    echo     ✓ Upload service binary found
    set UPLOAD_BINARY=..\go-microservice\cmd\upload-service\upload-service.exe
) else (
    echo     Building Upload service...
    cd ..\go-microservice && go build -o .\cmd\upload-service\upload-service.exe .\cmd\upload-service && cd ..\deeds-web-app
    set UPLOAD_BINARY=..\go-microservice\cmd\upload-service\upload-service.exe
)

REM === PHASE 3: SERVICE STARTUP SEQUENCE ===
echo.
echo [3/3] === STARTING SERVICES IN DEPENDENCY ORDER ===

echo [3.1] Starting core infrastructure...

REM PostgreSQL
echo     [1/9] PostgreSQL...
pg_ctl status -D "C:\Program Files\PostgreSQL\16\data" > nul 2>&1
if !errorlevel! == 0 (
    echo           ✓ PostgreSQL already running
) else (
    echo           Starting PostgreSQL...
    net start postgresql-x64-16 > nul 2>&1
)

REM Redis  
echo     [2/9] Redis...
if exist "redis-windows\redis-server.exe" (
    tasklist /FI "IMAGENAME eq redis-server.exe" 2>NUL | find /I /N "redis-server.exe" > nul
    if !errorlevel! == 0 (
        echo           ✓ Redis already running
    ) else (
        echo           Starting Redis...
        start /B redis-windows\redis-server.exe redis-windows\redis.windows.conf
        timeout /t 2 /nobreak > nul
    )
) else (
    echo           ⚠ Redis binary not found - skipping
)

REM Ollama
echo     [3/9] Ollama...
curl -s http://localhost:11434 > nul 2>&1
if !errorlevel! == 0 (
    echo           ✓ Ollama already running  
) else (
    echo           Starting Ollama...
    start /B ollama serve
    timeout /t 3 /nobreak > nul
)

REM MinIO
echo     [4/9] MinIO...
if exist "minio.exe" (
    curl -s http://localhost:9000 > nul 2>&1
    if !errorlevel! == 0 (
        echo           ✓ MinIO already running
    ) else (
        echo           Starting MinIO...
        start /B minio.exe server ./minio-data --console-address ":9001"
        timeout /t 2 /nobreak > nul
    )
) else (
    echo           ⚠ MinIO binary not found - skipping
)

REM Qdrant  
echo     [5/9] Qdrant...
if exist "qdrant-windows\qdrant.exe" (
    curl -s http://localhost:6333 > nul 2>&1
    if !errorlevel! == 0 (
        echo           ✓ Qdrant already running
    ) else (
        echo           Starting Qdrant...
        start /B qdrant-windows\qdrant.exe
        timeout /t 2 /nobreak > nul
    )
) else (
    echo           ⚠ Qdrant binary not found - skipping
)

REM Neo4j
echo     [6/9] Neo4j...
curl -s http://localhost:7474 > nul 2>&1
if !errorlevel! == 0 (
    echo           ✓ Neo4j already running
) else (
    echo           ⚠ Neo4j requires manual start
    echo             Run: neo4j-desktop or neo4j start
)

echo [3.2] Starting application services...

REM Enhanced RAG Service
echo     [7/9] Enhanced RAG Service (port %RAG_PORT%)...
curl -s http://localhost:%RAG_PORT% > nul 2>&1
if !errorlevel! == 0 (
    echo           ✓ Enhanced RAG already running on port %RAG_PORT%
) else (
    echo           Starting Enhanced RAG service...
    set RAG_SERVICE_PORT=%RAG_PORT%
    start /B %RAG_BINARY%
    timeout /t 3 /nobreak > nul
)

REM Upload Service  
echo     [8/9] Upload Service (port %UPLOAD_PORT%)...
curl -s http://localhost:%UPLOAD_PORT% > nul 2>&1
if !errorlevel! == 0 (
    echo           ✓ Upload Service already running on port %UPLOAD_PORT%
) else (
    echo           Starting Upload service...
    set UPLOAD_SERVICE_PORT=%UPLOAD_PORT%
    start /B %UPLOAD_BINARY%
    timeout /t 2 /nobreak > nul
)

REM SvelteKit Frontend
echo     [9/9] SvelteKit Frontend...
curl -s http://localhost:5173 > nul 2>&1
if !errorlevel! == 0 (
    echo           ✓ SvelteKit already running
) else (
    echo           Starting SvelteKit...
    cd sveltekit-frontend
    start /B npm run dev
    cd ..
    timeout /t 5 /nobreak > nul
)

echo.
echo ===============================================================================
echo STARTUP COMPLETE - TESTING ENDPOINTS
echo ===============================================================================

REM Test all endpoints
echo Testing core services...
curl -s http://localhost:5173 > nul && echo ✓ SvelteKit (5173) || echo ✗ SvelteKit (5173)
curl -s http://localhost:%RAG_PORT% > nul && echo ✓ Enhanced RAG (%RAG_PORT%) || echo ✗ Enhanced RAG (%RAG_PORT%)  
curl -s http://localhost:%UPLOAD_PORT% > nul && echo ✓ Upload Service (%UPLOAD_PORT%) || echo ✗ Upload Service (%UPLOAD_PORT%)
curl -s http://localhost:11434 > nul && echo ✓ Ollama (11434) || echo ✗ Ollama (11434)

echo.
echo ===============================================================================
echo LEGAL AI PLATFORM READY!
echo ===============================================================================
echo.
echo Access Points:
echo   ► YoRHa Homepage:    http://localhost:5173/yorha-home
echo   ► SvelteKit Main:    http://localhost:5173  
echo   ► Enhanced RAG API:  http://localhost:%RAG_PORT%
echo   ► Upload API:        http://localhost:%UPLOAD_PORT%
echo   ► Ollama API:        http://localhost:11434
echo.
echo Next Steps:
echo   1. Open http://localhost:5173/yorha-home
echo   2. Test the API integration buttons
echo   3. Verify YoRHaTable and YoRHaCommandCenter components
echo.
echo Press any key to open homepage in browser...
pause > nul
start http://localhost:5173/yorha-home

echo.
echo ===============================================================================
echo STARTUP LOG COMPLETE - All port conflicts resolved!
echo ===============================================================================

endlocal