@echo off
REM =================================================================
REM Go 1.25 Performance Benchmark Suite
REM Legal AI Platform - Real-World Performance Testing
REM =================================================================

echo 🏁 Go 1.25 Performance Benchmark Suite
echo    Testing Legal AI Platform Performance Improvements
echo.

REM Create benchmark results directory
if not exist "benchmark-results" mkdir benchmark-results
cd benchmark-results

echo 📊 Starting Comprehensive Performance Tests...
echo    Date: %date% %time%
echo    Go Version: Go 1.25.0 with greenteagc
echo.

REM Test 1: JSON Processing Performance (Legal Documents)
echo 🧪 Test 1: JSON Processing Performance
echo    Testing with legal document payloads (1KB to 10MB)

REM Start Enhanced RAG service for testing
start /B "RAG-Benchmark" ..\go-microservice\bin\enhanced-rag.exe --port=8094 --benchmark-mode
timeout /t 5 /nobreak >nul

echo    📤 Testing 1KB legal contract...
curl -X POST http://localhost:8094/api/benchmark/json ^
  -H "Content-Type: application/json" ^
  -H "X-Benchmark: true" ^
  -d "{\"document_type\":\"contract\",\"size\":\"1KB\",\"test\":\"json_v2_decode\"}" ^
  -w "Response time: %%{time_total}s\n" ^
  -o test1-1kb-results.json

echo    📤 Testing 100KB legal brief...
curl -X POST http://localhost:8094/api/benchmark/json ^
  -H "Content-Type: application/json" ^
  -H "X-Benchmark: true" ^
  -d "{\"document_type\":\"legal_brief\",\"size\":\"100KB\",\"test\":\"json_v2_decode\"}" ^
  -w "Response time: %%{time_total}s\n" ^
  -o test1-100kb-results.json

echo    📤 Testing 1MB case documents...
curl -X POST http://localhost:8094/api/benchmark/json ^
  -H "Content-Type: application/json" ^
  -H "X-Benchmark: true" ^
  -d "{\"document_type\":\"case_documents\",\"size\":\"1MB\",\"test\":\"json_v2_decode\"}" ^
  -w "Response time: %%{time_total}s\n" ^
  -o test1-1mb-results.json

echo.

REM Test 2: Vector Processing & Embeddings
echo 🧪 Test 2: Vector Processing & Embeddings
echo    Testing pgvector operations with Go 1.25 optimizations

curl -X POST http://localhost:8095/api/benchmark/vector ^
  -H "Content-Type: application/json" ^
  -d "{\"test\":\"vector_similarity\",\"dimensions\":768,\"batch_size\":1000}" ^
  -w "Vector processing time: %%{time_total}s\n" ^
  -o test2-vector-results.json 2>/dev/null &

echo.

REM Test 3: CUDA AI Processing
echo 🧪 Test 3: CUDA AI Processing (GPU Acceleration)
echo    Testing GPU workloads with Go 1.25 container-aware GOMAXPROCS

curl -X POST http://localhost:8096/api/benchmark/cuda ^
  -H "Content-Type: application/json" ^
  -d "{\"test\":\"gpu_inference\",\"model\":\"legal-bert\",\"batch_size\":32}" ^
  -w "CUDA processing time: %%{time_total}s\n" ^
  -o test3-cuda-results.json 2>/dev/null &

echo.

REM Test 4: Crypto Operations (Authentication/Security)
echo 🧪 Test 4: Crypto Operations Performance
echo    Testing 2-4x faster crypto with Go 1.25

curl -X POST http://localhost:8094/api/benchmark/crypto ^
  -H "Content-Type: application/json" ^
  -d "{\"test\":\"jwt_signing\",\"iterations\":10000}" ^
  -w "Crypto operations time: %%{time_total}s\n" ^
  -o test4-crypto-results.json

echo.

REM Test 5: Memory Usage & GC Performance
echo 🧪 Test 5: Memory Usage & GC Performance
echo    Testing 10-40%% GC overhead reduction with greenteagc

curl -X POST http://localhost:8094/api/benchmark/memory ^
  -H "Content-Type: application/json" ^
  -d "{\"test\":\"gc_pressure\",\"allocations\":1000000}" ^
  -w "Memory test time: %%{time_total}s\n" ^
  -o test5-memory-results.json

echo.

REM Wait for all background tests to complete
echo ⏳ Waiting for background tests to complete...
timeout /t 10 /nobreak >nul

REM Stop benchmark services
taskkill /F /IM enhanced-rag.exe >nul 2>&1
taskkill /F /IM vector-service.exe >nul 2>&1
taskkill /F /IM cuda-ai-service.exe >nul 2>&1

echo.
echo 📊 Benchmark Results Summary:
echo ================================

if exist "test1-1kb-results.json" (
    echo ✅ JSON 1KB Processing:
    type test1-1kb-results.json | findstr "response_time\|performance"
    echo.
)

if exist "test1-100kb-results.json" (
    echo ✅ JSON 100KB Processing:  
    type test1-100kb-results.json | findstr "response_time\|performance"
    echo.
)

if exist "test2-vector-results.json" (
    echo ✅ Vector Processing:
    type test2-vector-results.json | findstr "processing_time\|throughput"
    echo.
)

if exist "test3-cuda-results.json" (
    echo ✅ CUDA GPU Processing:
    type test3-cuda-results.json | findstr "gpu_time\|acceleration"
    echo.
)

if exist "test4-crypto-results.json" (
    echo ✅ Crypto Operations:
    type test4-crypto-results.json | findstr "crypto_time\|speedup"
    echo.
)

echo.
echo 🏆 Expected Go 1.25 Performance Improvements:
echo    📈 JSON Processing: 20-30%% faster decoding
echo    🔥 GC Overhead: 10-40%% reduction  
echo    🚀 Crypto Operations: 2-4x faster
echo    💾 Memory Usage: 15-25%% lower peak
echo    🔧 Binary Size: 32%% smaller
echo    ⚡ Container CPU: Auto-optimized GOMAXPROCS
echo.
echo 📁 Detailed results saved in: benchmark-results\
echo 🔄 Run again: BENCHMARK-GO125-PERFORMANCE.bat

cd ..
echo.
echo ✅ Benchmark Suite Complete!
pause