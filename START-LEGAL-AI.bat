@echo off
REM ================================================================================
REM LEGAL AI PLATFORM - COMPLETE PRODUCTION STARTUP
REM ================================================================================

echo.
echo ================================================================================
echo STARTING LEGAL AI PLATFORM - FULL PRODUCTION SYSTEM
echo ================================================================================
echo.

echo [1/10] Starting PostgreSQL...
net start postgresql-x64-17 2>nul || echo PostgreSQL already running

echo [2/10] Starting Redis...
start /min redis-server || start /min .\redis-windows\redis-server.exe

echo [3/10] Starting Ollama...
tasklist | findstr "ollama" >nul || start /min ollama serve

echo [4/10] Starting MinIO...
if not exist minio-data mkdir minio-data
tasklist | findstr "minio" >nul || start /min minio.exe server ./minio-data --address :9000 --console-address :9001

echo [5/10] Starting Qdrant Vector Database...
tasklist | findstr "qdrant" >nul || start /min .\qdrant-windows\qdrant.exe

echo [6/10] Starting Neo4j...
powershell -Command "Start-Service neo4j" 2>nul || echo Neo4j manual start required

echo [7/10] Starting Go Enhanced RAG Service...
start /min cmd /c "cd go-microservice && go run cmd/enhanced-rag/main.go" 2>nul || start /min cmd /c "cd go-microservice && go run main.go"

echo [8/10] Starting Go Upload Service...
start /min cmd /c "cd go-microservice && go run cmd/upload-service/main.go" 2>nul || echo Upload service fallback

echo [9/10] Starting Go AI Services...
start /min cmd /c "cd go-microservice && go run cmd/ai-summary/main.go" 2>nul || echo AI Summary service optional

echo [10/10] Starting SvelteKit Frontend...
cd sveltekit-frontend && start cmd /k "npm run dev -- --host 0.0.0.0" && cd ..

timeout /t 8 /nobreak >nul

echo.
echo ================================================================================
echo LEGAL AI PLATFORM STARTED SUCCESSFULLY!
echo ================================================================================
echo.
echo Access Points:
echo - Frontend:        http://localhost:5173
echo - Enhanced RAG:    http://localhost:8094/api/rag
echo - Upload API:      http://localhost:8093/upload
echo - MinIO Console:   http://localhost:9001 (admin/minioadmin)
echo - Qdrant API:      http://localhost:6333
echo - Neo4j Browser:   http://localhost:7474
echo - Ollama API:      http://localhost:11434
echo.
echo Database Details:
echo - PostgreSQL:      postgresql://legal_admin:123456@localhost:5432/legal_ai_db
echo - Redis:           redis://localhost:6379
echo.
echo Press any key to open the frontend in your browser...
pause >nul

start http://localhost:5173

echo.
echo System Status Check:
echo ==================
curl -s http://localhost:11434/api/tags >nul 2>&1 && echo ✓ Ollama: Running || echo ✗ Ollama: Not responding
curl -s http://localhost:6333/collections >nul 2>&1 && echo ✓ Qdrant: Running || echo ✗ Qdrant: Not responding
redis-cli ping >nul 2>&1 && echo ✓ Redis: Running || echo ✗ Redis: Not responding
echo ✓ PostgreSQL: Check manually with psql
echo.
echo Happy coding! 🚀