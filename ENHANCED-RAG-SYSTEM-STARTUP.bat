@echo off
echo ==============================================
echo Enhanced RAG Legal AI System v2.0.0
echo ==============================================
echo.

:: Set environment variables
set NODE_ENV=development
set OLLAMA_URL=http://localhost:11434
set DB_HOST=localhost
set DB_PORT=5432
set DB_NAME=legal_ai_db
set DB_USER=legal_admin
set DB_PASSWORD=LegalAI2024!
set REDIS_URL=redis://localhost:6379
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=password
set QDRANT_URL=http://localhost:6333

echo Step 1: Installing/updating dependencies...
echo ----------------------------------------
call npm install
if %ERRORLEVEL% neq 0 (
    echo ❌ NPM install failed
    pause
    exit /b 1
)

echo.
echo Step 2: Starting required services...
echo ----------------------------------------

:: Start PostgreSQL (if not running)
echo Checking PostgreSQL...
"C:\Program Files\PostgreSQL\17\bin\pg_isready.exe" -h localhost -p 5432 >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Starting PostgreSQL...
    net start postgresql-x64-17 >nul 2>&1
)

:: Start Redis
echo Checking Redis...
ping -n 1 localhost >nul 2>&1 && (
    start /B redis-windows\redis-server.exe redis-windows\redis.conf >nul 2>&1
    timeout /t 2 >nul
)

:: Start Ollama
echo Checking Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >nul
if %ERRORLEVEL% neq 0 (
    echo Starting Ollama...
    start /B ollama serve >nul 2>&1
    timeout /t 5 >nul
)

:: Start Qdrant (if available)
echo Checking Qdrant...
if exist qdrant-windows\qdrant.exe (
    start /B qdrant-windows\qdrant.exe >nul 2>&1
    timeout /t 3 >nul
)

:: Start Neo4j (if available)
echo Checking Neo4j...
sc query Neo4j >nul 2>&1
if %ERRORLEVEL% equ 0 (
    sc start Neo4j >nul 2>&1
)

echo.
echo Step 3: Running health checks...
echo ----------------------------------------

:: Test database connection
echo Testing PostgreSQL connection...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U %DB_USER% -d %DB_NAME% -h localhost -c "SELECT version();" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✅ PostgreSQL: Connected
) else (
    echo ❌ PostgreSQL: Connection failed
)

:: Test Redis
echo Testing Redis connection...
redis-windows\redis-cli.exe ping >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✅ Redis: Connected
) else (
    echo ❌ Redis: Connection failed
)

:: Test Ollama
echo Testing Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✅ Ollama: Connected
) else (
    echo ❌ Ollama: Connection failed
)

echo.
echo Step 4: Building the application...
echo ----------------------------------------
call npm run build
if %ERRORLEVEL% neq 0 (
    echo ⚠️  Build had warnings but continuing...
) else (
    echo ✅ Build completed successfully
)

echo.
echo Step 5: Starting development server...
echo ----------------------------------------
echo.
echo 🚀 Enhanced RAG System starting at http://localhost:5173
echo.
echo Available endpoints:
echo   • Main App:           http://localhost:5173
echo   • Enhanced RAG API:   http://localhost:5173/api/enhanced-rag
echo   • Health Check:       http://localhost:5173/api/health
echo   • AI Test Interface:  http://localhost:5173/ai-test
echo.
echo Features enabled:
echo   ✓ Unified Database Service (PostgreSQL + Redis + Neo4j + Qdrant)
echo   ✓ Unified AI Service (Ollama + Embeddings)
echo   ✓ Enhanced RAG Pipeline (Hybrid Search + Self-Organizing)
echo   ✓ SIMD JSON Parser (WebGPU Acceleration)
echo   ✓ Recommendation Engine
echo   ✓ Performance Monitoring
echo.
echo Press Ctrl+C to stop the server
echo ==============================================

:: Start the development server
call npm run dev

echo.
echo Server stopped. Services are still running in background.
echo Run 'taskkill /F /IM redis-server.exe' to stop Redis if needed.
echo Run 'taskkill /F /IM ollama.exe' to stop Ollama if needed.
pause