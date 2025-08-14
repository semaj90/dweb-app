@echo off
REM ================================================================================
REM LEGAL AI PLATFORM - QUICK START
REM ================================================================================

echo.
echo ================================================================================
echo STARTING LEGAL AI PLATFORM - COMPLETE PIPELINE
echo ================================================================================
echo.

echo [1/8] Starting PostgreSQL...
net start postgresql-x64-17 2>nul

echo [2/8] Starting Redis...
start /min redis-server

echo [3/8] Starting Ollama...
start /min ollama serve

timeout /t 3 /nobreak >nul

echo [4/8] Starting MinIO...
if not exist minio-data mkdir minio-data
start /min minio.exe server ./minio-data --address :9000 --console-address :9001

echo [5/8] Starting Enhanced RAG...
cd go-services\cmd\enhanced-rag
start /min cmd /c "go run main.go"
cd ..\..\..

echo [6/8] Starting Upload Service...
cd go-microservice
start /min cmd /c "go run main.go"
cd ..

echo [7/8] Starting XState Manager...
cd go-services\cmd\xstate-manager
start /min cmd /c "go run main.go"
cd ..\..\..

echo [8/8] Starting Frontend...
cd sveltekit-frontend
start cmd /k "npm run dev -- --host 0.0.0.0"
cd ..

timeout /t 5 /nobreak >nul

echo.
echo ================================================================================
echo LEGAL AI PLATFORM STARTED SUCCESSFULLY!
echo ================================================================================
echo.
echo Access Points:
echo - Frontend:       http://localhost:5173
echo - RAG API:        http://localhost:8094/api/rag
echo - Upload API:     http://localhost:8093/upload
echo - MinIO Console:  http://localhost:9001
echo - Ollama API:     http://localhost:11434
echo.
echo Press any key to open the frontend in your browser...
pause >nul

start http://localhost:5173
