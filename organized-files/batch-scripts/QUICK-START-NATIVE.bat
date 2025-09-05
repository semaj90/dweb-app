@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo       Quick Native MinIO + SvelteKit + RAG Integration
echo ================================================================

REM Colors
for /f %%A in ('echo prompt $E ^| cmd') do set "ESC=%%A"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "RED=%ESC%[31m"
set "CYAN=%ESC%[36m"
set "RESET=%ESC%[0m"

echo %CYAN%Setting up environment...%RESET%

REM Set environment variables
set MINIO_ROOT_USER=minioadmin
set MINIO_ROOT_PASSWORD=minioadmin
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set PG_CONN_STRING=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set MINIO_ENDPOINT=localhost:9000
set MINIO_ACCESS_KEY=minioadmin
set MINIO_SECRET_KEY=minioadmin
set MINIO_BUCKET=legal-documents
set UPLOAD_SERVICE_PORT=8094
set RAG_HTTP_PORT=8093
set OLLAMA_BASE_URL=http://localhost:11434
set EMBED_MODEL=nomic-embed-text

REM Create directories
if not exist "C:\minio-data" mkdir "C:\minio-data"
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs

REM Test PostgreSQL
echo %YELLOW%Testing PostgreSQL...%RESET%
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT version();" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ PostgreSQL OK%RESET%
) else (
    echo %RED%❌ PostgreSQL not accessible%RESET%
    pause
    exit /b 1
)

REM Start Ollama
echo %YELLOW%Starting Ollama...%RESET%
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >NUL
if "%ERRORLEVEL%"=="1" (
    start /B "Ollama" cmd /c "ollama serve > logs\ollama.log 2>&1"
    timeout /t 3 >nul
)
echo %GREEN%✅ Ollama ready%RESET%

REM Start MinIO
echo %YELLOW%Starting MinIO...%RESET%
start /B "MinIO" cmd /c "minio server C:\minio-data --console-address :9001 > logs\minio.log 2>&1"
timeout /t 3 >nul
echo %GREEN%✅ MinIO started%RESET%

REM Start Go services
echo %YELLOW%Starting RAG service (port 8093)...%RESET%
cd go-microservice
start /B "RAG" cmd /c "go run cmd/rag-kratos/main.go > ..\logs\rag.log 2>&1"
cd ..
timeout /t 2 >nul

echo %YELLOW%Starting Upload service (port 8094)...%RESET%
cd go-microservice
start /B "Upload" cmd /c "go run cmd/upload-service/main.go > ..\logs\upload.log 2>&1"
cd ..
timeout /t 2 >nul

REM Start SvelteKit
echo %YELLOW%Starting SvelteKit (port 5173)...%RESET%
cd sveltekit-frontend
start /B "SvelteKit" cmd /c "npm run dev > ..\logs\svelte.log 2>&1"
cd ..

echo.
echo %GREEN%🎉 All services started!%RESET%
echo.
echo %CYAN%Service URLs:%RESET%
echo   • MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
echo   • RAG Service: http://localhost:8093 (/health, /embed, /rag)
echo   • Upload Service: http://localhost:8094 (/health, /upload)
echo   • SvelteKit App: http://localhost:5173
echo   • PostgreSQL: localhost:5432
echo.
echo %YELLOW%Wait 10 seconds then check: http://localhost:5173%RESET%
timeout /t 10 >nul
start http://localhost:5173
pause