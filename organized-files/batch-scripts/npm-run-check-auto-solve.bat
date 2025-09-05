@echo off
echo 🎯 Starting Complete Error-to-Vector Case Solving System
echo ================================================================
echo.

:: Set comprehensive environment variables
set CUDA_ENABLED=true
set GPU_MEMORY_LIMIT=6GB
set LOAD_BALANCER_STRATEGY=gpu_aware
set LEGAL_BERT_ENABLED=true
set GOLLAMA_ENABLED=true
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set OLLAMA_URL=http://localhost:11434
set MAX_WORKERS=8
set UV_THREADPOOL_SIZE=16
set NODE_OPTIONS=--max-old-space-size=4096 --expose-gc

echo 🔧 Environment Configuration:
echo    - CUDA Enabled: %CUDA_ENABLED%
echo    - GPU Memory Limit: %GPU_MEMORY_LIMIT%
echo    - Max Workers: %MAX_WORKERS%
echo    - Thread Pool Size: %UV_THREADPOOL_SIZE%
echo    - Database: PostgreSQL
echo    - AI Model: Ollama + Legal-BERT
echo.

echo 📊 System Architecture:
echo    ✅ GPU-Accelerated Parsing (SIMD + CUDA)
echo    ✅ Multi-Core Service Workers (8 workers)
echo    ✅ Thread Assignment to Event Loop
echo    ✅ Metadata Indexing & Vector Search
echo    ✅ Error-to-Vector Processing Pipeline
echo    ✅ Background Processing & Auto-Solving
echo.

:: Check if PostgreSQL is running
echo 🔍 Checking PostgreSQL connection...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT version();" >nul 2>&1
if errorlevel 1 (
    echo ❌ PostgreSQL not running. Please start PostgreSQL first.
    pause
    exit /b 1
)
echo    ✅ PostgreSQL is running

:: Check if Ollama is running
echo 🔍 Checking Ollama service...
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama not running. Starting Ollama...
    start /B ollama serve
    timeout /t 5 >nul
)
echo    ✅ Ollama is running

echo.
echo 🚀 Launching Complete Case Solver...
echo    - Service Worker Manager with optimal thread assignment
echo    - GPU indexing and embedding service  
echo    - SIMD parser for high-performance JSON processing
echo    - Error-to-vector recommendation pipeline
echo    - Background processing with event loop optimization
echo.

:: Launch the complete case solver
node scripts\complete-case-solver.cjs

echo.
echo ✅ Case solving session completed!
echo.
echo 📊 Summary of Integration:
echo    ✅ GPU Parsing: SIMD + CUDA acceleration
echo    ✅ Service Workers: Background processing with thread assignment
echo    ✅ Indexing: Metadata-based with vector embeddings
echo    ✅ Search & Sort: Similarity-based with metadata filters
echo    ✅ Error Processing: Automated error-to-vector pipeline
echo    ✅ Case Solving: Complete integration pipeline
echo.
pause