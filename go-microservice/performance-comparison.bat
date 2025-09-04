@echo off
REM =================================================================
REM Go 1.24 vs Go 1.25 Performance Comparison Script
REM Legal AI Platform Benchmarking
REM =================================================================

echo 🏁 Starting Performance Comparison: Go 1.24 vs Go 1.25
echo.

REM Create performance test directory
if not exist "performance-tests" mkdir performance-tests
cd performance-tests

echo 📊 Testing JSON Processing Performance...
echo Testing with 10MB legal document dataset...

REM Start Go 1.24 enhanced RAG (existing binary)
echo.
echo 🔵 Testing Go 1.24 Enhanced RAG Service...
start /B "Go1.24-RAG" ..\bin\enhanced-rag.exe --port=8094 --benchmark-mode
timeout /t 3 /nobreak >nul

REM Test JSON processing performance
curl -X POST http://localhost:8094/api/benchmark/json ^
  -H "Content-Type: application/json" ^
  -d "{\"test\":\"json_processing\",\"payload_size\":\"10MB\"}" ^
  -o go124-json-results.json

REM Stop Go 1.24 service
taskkill /F /IM enhanced-rag.exe >nul 2>&1

echo.
echo 🟢 Testing Go 1.25 Enhanced RAG Service (when available)...
if exist "..\bin\enhanced-rag-go125.exe" (
    start /B "Go1.25-RAG" ..\bin\enhanced-rag-go125.exe --port=8094 --benchmark-mode
    timeout /t 3 /nobreak >nul
    
    REM Test JSON processing performance
    curl -X POST http://localhost:8094/api/benchmark/json ^
      -H "Content-Type: application/json" ^
      -d "{\"test\":\"json_processing\",\"payload_size\":\"10MB\"}" ^
      -o go125-json-results.json
    
    taskkill /F /IM enhanced-rag-go125.exe >nul 2>&1
) else (
    echo ⚠️ Go 1.25 binary not found. Run build-optimized-go125.bat first.
)

echo.
echo 📈 Performance Comparison Results:
echo.
if exist "go124-json-results.json" (
    echo Go 1.24 Results:
    type go124-json-results.json
    echo.
)

if exist "go125-json-results.json" (
    echo Go 1.25 Results:
    type go125-json-results.json
    echo.
    echo 🚀 Expected improvements with Go 1.25:
    echo    - JSON processing: 20-30%% faster decoding
    echo    - GC overhead: 10-40%% reduction
    echo    - Memory usage: 15-25%% lower peak usage
    echo    - Crypto operations: 2-4x faster
)

cd ..
echo.
echo ✅ Performance comparison complete!
echo 📊 Results saved in performance-tests/ directory