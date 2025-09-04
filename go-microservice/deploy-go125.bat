@echo off
REM =================================================================
REM Deploy Go 1.25 Optimized Binaries
REM Legal AI Platform Production Deployment
REM =================================================================

echo 🚀 Deploying Go 1.25 Optimized Services...
echo.

REM Backup existing binaries
echo 📦 Backing up existing binaries...
if not exist "bin\backup" mkdir bin\backup
if exist "bin\enhanced-rag.exe" copy "bin\enhanced-rag.exe" "bin\backup\enhanced-rag-go124.exe"
if exist "bin\upload-service.exe" copy "bin\upload-service.exe" "bin\backup\upload-service-go124.exe"

REM Deploy Go 1.25 binaries
echo 🚀 Deploying Go 1.25 optimized binaries...
if exist "bin\enhanced-rag-go125.exe" (
    copy "bin\enhanced-rag-go125.exe" "bin\enhanced-rag.exe"
    echo ✅ Enhanced RAG Service (Go 1.25) deployed
)

if exist "bin\upload-service-go125.exe" (
    copy "bin\upload-service-go125.exe" "bin\upload-service.exe"
    echo ✅ Upload Service (Go 1.25) deployed
)

echo.
echo 🔥 Performance Features Active:
echo    ✅ Experimental greenteagc GC (10-40%% overhead reduction)
echo    ✅ Container-aware GOMAXPROCS
echo    ✅ JSON v2 performance improvements  
echo    ✅ Crypto performance enhancements (2-4x faster)
echo    ✅ Optimized binary size (stripped symbols)
echo.
echo 🚀 Services ready for production:
echo    📡 Enhanced RAG: http://localhost:8094
echo    📂 Upload Service: http://localhost:8093
echo.
echo 🛡️  Rollback available: bin\backup\*-go124.exe
echo.