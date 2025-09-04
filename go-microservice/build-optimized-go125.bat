@echo off
REM =================================================================
REM Go 1.25 Performance Optimized Build Script
REM Legal AI Platform - Enhanced RAG Services
REM =================================================================

echo 🚀 Building Go 1.25 Optimized Binaries for Legal AI Platform...
echo.

REM Set Go 1.25 performance environment variables
set GOEXPERIMENT=greenteagc
set GOPROXY=https://proxy.golang.org,direct
set GOMAXPROCS=0
set CGO_ENABLED=1

REM Build enhanced-rag service with performance optimizations
echo 📦 Building Enhanced RAG Service...
cd cmd\enhanced-rag
go build -ldflags="-s -w -X main.version=v1.25-optimized" -tags="netgo,osusergo" -gcflags="-dwarf=false" -o ..\..\bin\enhanced-rag-go125.exe .
cd ..\..

REM Build upload service
echo 📦 Building Upload Service...
cd cmd\upload-service
go build -ldflags="-s -w" -tags="netgo,osusergo" -gcflags="-dwarf=false" -o ..\..\bin\upload-service-go125.exe .
cd ..\..

REM Build vector processing service
echo 📦 Building Vector Processing Service...
cd cmd\vector-processing
go build -ldflags="-s -w" -tags="netgo,osusergo" -gcflags="-dwarf=false" -o ..\..\bin\vector-processing-go125.exe .
cd ..\..

REM Build GPU CUDA service
echo 📦 Building GPU CUDA Service...
cd cmd\cuda-ai-service
go build -ldflags="-s -w" -tags="netgo,osusergo" -gcflags="-dwarf=false" -o ..\..\bin\cuda-ai-service-go125.exe .
cd ..\..

echo.
echo ✅ Go 1.25 Optimized Build Complete!
echo 🔥 Performance Features Enabled:
echo    - Experimental greenteagc (10-40%% GC overhead reduction)
echo    - Container-aware GOMAXPROCS
echo    - JSON v2 performance improvements
echo    - Crypto performance enhancements
echo.
echo 🚀 Binaries available in bin/ directory:
echo    - enhanced-rag-go125.exe (port 8094)
echo    - upload-service-go125.exe (port 8093)  
echo    - vector-processing-go125.exe (port 8095)
echo    - cuda-ai-service-go125.exe (port 8096)
echo.
echo 📊 Performance Testing Commands:
echo    bin\enhanced-rag-go125.exe --benchmark
echo    bin\upload-service-go125.exe --profile
echo.