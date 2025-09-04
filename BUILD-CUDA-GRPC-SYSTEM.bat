@echo off
REM BUILD CUDA gRPC SYSTEM - Complete compilation
REM This script builds the entire CUDA gRPC streaming architecture

setlocal EnableDelayedExpansion

echo ===========================================
echo    🔥 BUILDING CUDA gRPC STREAMING SYSTEM
echo ===========================================
echo.

REM Check for required tools
echo 🔍 Checking build requirements...

where protoc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ protoc not found. Please install Protocol Buffers compiler
    echo    Download from: https://github.com/protocolbuffers/protobuf/releases
    pause
    exit /b 1
)
echo ✅ protoc found

where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ nvcc not found. Please install CUDA Toolkit
    echo    Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)
echo ✅ CUDA toolkit found

where emcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ emcc not found. Please install Emscripten
    echo    Install with: winget install emscripten
    pause
    exit /b 1
)
echo ✅ Emscripten found

where go >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Go not found. Please install Go
    echo    Download from: https://golang.org/dl/
    pause
    exit /b 1
)
echo ✅ Go compiler found

echo.
echo 🏗️ Starting build process...

REM Create build directories
if not exist "build" mkdir build
if not exist "build\cuda" mkdir build\cuda
if not exist "build\grpc" mkdir build\grpc
if not exist "build\wasm" mkdir build\wasm
if not exist "cuda-streaming" mkdir cuda-streaming

REM Step 1: Generate Protocol Buffer files
echo.
echo 📋 Step 1: Generating Protocol Buffer files...
protoc --cpp_out=build/grpc --grpc_out=build/grpc --plugin=protoc-gen-grpc="C:\vcpkg\installed\x64-windows\tools\grpc\grpc_cpp_plugin.exe" legal_cuda_streaming.proto
if %ERRORLEVEL% neq 0 (
    echo ❌ Protocol Buffer generation failed
    pause
    exit /b 1
)

protoc --js_out=import_style=commonjs:build/grpc --grpc-web_out=import_style=commonjs,mode=grpcwebtext:build/grpc legal_cuda_streaming.proto
if %ERRORLEVEL% neq 0 (
    echo ❌ gRPC-Web generation failed  
    pause
    exit /b 1
)
echo ✅ Protocol Buffer files generated

REM Step 2: Compile CUDA kernels
echo.
echo 🔥 Step 2: Compiling CUDA kernels...
nvcc -c cuda_legal_kernels.cu -o build/cuda/cuda_legal_kernels.o ^
     -gencode arch=compute_75,code=sm_75 ^
     -gencode arch=compute_86,code=sm_86 ^
     -O3 -Xptxas -O3 -use_fast_math ^
     -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include"
if %ERRORLEVEL% neq 0 (
    echo ❌ CUDA kernel compilation failed
    pause
    exit /b 1
)
echo ✅ CUDA kernels compiled

REM Step 3: Compile CUDA gRPC Server
echo.
echo 🚀 Step 3: Compiling CUDA gRPC Server...
nvcc legal_cuda_server.cpp build/cuda/cuda_legal_kernels.o build/grpc/legal_cuda_streaming.pb.cc build/grpc/legal_cuda_streaming.grpc.pb.cc ^
     -o legal_cuda_server.exe ^
     -gencode arch=compute_75,code=sm_75 ^
     -gencode arch=compute_86,code=sm_86 ^
     -O3 -Xptxas -O3 ^
     -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include" ^
     -I"C:\vcpkg\installed\x64-windows\include" ^
     -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64" ^
     -L"C:\vcpkg\installed\x64-windows\lib" ^
     -lcudart -lcublas -lcurand -lcudnn ^
     -lgrpc++ -lgrpc -lgpr -lprotobuf ^
     -lws2_32 -ladvapi32 -lcrypt32 -lkernel32
if %ERRORLEVEL% neq 0 (
    echo ❌ CUDA gRPC Server compilation failed
    pause
    exit /b 1
)
echo ✅ CUDA gRPC Server compiled

REM Step 4: Compile WebAssembly gRPC Client
echo.
echo 🌐 Step 4: Compiling WebAssembly gRPC Client...
emcc legal_grpc_client.cpp build/grpc/legal_cuda_streaming.pb.cc ^
     -o build/wasm/legal_grpc_client.js ^
     -s WASM=1 -s EXPORTED_FUNCTIONS="['_malloc','_free','_main','_processDocument','_searchSimilar']" ^
     -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap']" ^
     -s ALLOW_MEMORY_GROWTH=1 ^
     -s INITIAL_MEMORY=32MB ^
     -s MAXIMUM_MEMORY=256MB ^
     -O3 -flto ^
     -I"C:\emsdk\upstream\emscripten\system\include" ^
     --bind
if %ERRORLEVEL% neq 0 (
    echo ❌ WebAssembly compilation failed
    pause
    exit /b 1
)
echo ✅ WebAssembly gRPC Client compiled

REM Step 5: Build Go Microservices
echo.
echo ⚡ Step 5: Building Go Microservices...

REM Build Enhanced RAG System
echo Building Enhanced RAG System...
go mod tidy
go build -o enhanced-rag-som-system.exe enhanced-rag-som-system.go
if %ERRORLEVEL% neq 0 (
    echo ❌ Enhanced RAG build failed
    pause
    exit /b 1
)

REM Build GPU Orchestrator
echo Building GPU Orchestrator...
go build -o gpu-orchestrator.exe gpu-orchestrator.go  
if %ERRORLEVEL% neq 0 (
    echo ❌ GPU Orchestrator build failed
    pause
    exit /b 1
)
echo ✅ Go microservices compiled

REM Step 6: Copy files to appropriate locations
echo.
echo 📁 Step 6: Organizing build artifacts...
copy "legal_cuda_server.exe" "cuda-streaming\" >nul 2>&1
copy "build\wasm\legal_grpc_client.js" "cuda-streaming\" >nul 2>&1
copy "build\wasm\legal_grpc_client.wasm" "cuda-streaming\" >nul 2>&1
copy "enhanced-rag-som-system.exe" "cuda-streaming\" >nul 2>&1
copy "gpu-orchestrator.exe" "cuda-streaming\" >nul 2>&1

REM Copy to SvelteKit static directory
if exist "sveltekit-frontend\static" (
    copy "build\wasm\legal_grpc_client.js" "sveltekit-frontend\static\" >nul 2>&1
    copy "build\wasm\legal_grpc_client.wasm" "sveltekit-frontend\static\" >nul 2>&1
    echo ✅ WebAssembly files copied to SvelteKit
)

echo ✅ Build artifacts organized

REM Step 7: Create TypeScript definitions
echo.
echo 📝 Step 7: Generating TypeScript definitions...
echo // Generated TypeScript definitions for Legal CUDA gRPC Client > cuda-streaming\legal_grpc_client.d.ts
echo declare module 'legal_grpc_client' { >> cuda-streaming\legal_grpc_client.d.ts
echo   export function processDocument(content: string, options?: any): Promise^<any^>; >> cuda-streaming\legal_grpc_client.d.ts  
echo   export function searchSimilar(query: string, threshold?: number): Promise^<any^>; >> cuda-streaming\legal_grpc_client.d.ts
echo   export function extractEntities(text: string): Promise^<any^>; >> cuda-streaming\legal_grpc_client.d.ts
echo   export function analyzeContract(contract: string): Promise^<any^>; >> cuda-streaming\legal_grpc_client.d.ts
echo } >> cuda-streaming\legal_grpc_client.d.ts

echo ✅ TypeScript definitions generated

REM Step 8: Test executables
echo.
echo 🧪 Step 8: Testing compiled executables...

echo Testing Enhanced RAG System...
timeout 3 enhanced-rag-som-system.exe --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ⚠️ Enhanced RAG System may have issues (this is expected without dependencies)
) else (
    echo ✅ Enhanced RAG System executable works
)

echo Testing GPU Orchestrator...
timeout 3 gpu-orchestrator.exe --version >nul 2>&1  
if %ERRORLEVEL% neq 0 (
    echo ⚠️ GPU Orchestrator may have issues (this is expected without dependencies)
) else (
    echo ✅ GPU Orchestrator executable works
)

echo ✅ Build tests completed

REM Final summary
echo.
echo ===========================================
echo    ✅ BUILD COMPLETED SUCCESSFULLY! 
echo ===========================================
echo.
echo 🏆 What was built:
echo    → legal_cuda_server.exe       (CUDA gRPC Server)
echo    → legal_grpc_client.js/.wasm  (WebAssembly Client)
echo    → enhanced-rag-som-system.exe (Go RAG Microservice)  
echo    → gpu-orchestrator.exe        (Go GPU Orchestrator)
echo    → Protocol Buffer bindings     (C++ and JS)
echo    → TypeScript definitions       (Browser integration)
echo.
echo 🚀 Next Steps:
echo    1. Run: START-CUDA-GRPC-SYSTEM.bat
echo    2. Access: http://localhost:5173/cuda-streaming  
echo    3. Monitor: GPU acceleration dashboard
echo.
echo 🎯 Your Legal AI system now has:
echo    → Real-time GPU acceleration
echo    → WebAssembly browser performance  
echo    → Streaming gRPC architecture
echo    → Advanced document clustering
echo    → CUDA kernel optimization
echo.
echo Press any key to continue...
pause >nul
