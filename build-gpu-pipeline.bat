@echo off
echo 🔨 GPU Acceleration Pipeline Build Script
echo ==========================================

:: Check prerequisites
echo.
echo 🔍 Checking build prerequisites...

:: Check for NVCC
nvcc --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ NVCC not found - CUDA Toolkit required for GPU acceleration
    echo 📥 Download from: https://developer.nvidia.com/cuda-downloads
    goto :error
) else (
    echo ✅ NVCC found - CUDA compilation ready
)

:: Check for Emscripten
emcc --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Emscripten not found - WebAssembly compilation will be skipped
    echo 📥 Install from: https://emscripten.org/docs/getting_started/downloads.html
    set SKIP_WASM=1
) else (
    echo ✅ Emscripten found - WebAssembly compilation ready
)

:: Check for Go
go version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Go compiler not found
    echo 📥 Download from: https://golang.org/dl/
    goto :error
) else (
    echo ✅ Go compiler found
)

echo.
echo 🏗️ Building GPU Pipeline Components...

:: Step 1: Compile CUDA Worker
echo.
echo [1/4] ⚡ Compiling CUDA Worker...
if not exist "cuda-worker" mkdir cuda-worker

:: Check if nlohmann/json header exists
if not exist "cuda-worker\json.hpp" (
    echo 📥 Downloading nlohmann/json header...
    curl -s -L https://github.com/nlohmann/json/releases/latest/download/json.hpp -o cuda-worker\json.hpp
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to download JSON header
        echo 💡 Manual download: Save https://github.com/nlohmann/json/releases/latest/download/json.hpp to cuda-worker\json.hpp
        goto :error
    )
)

:: Compile CUDA worker
echo 🔨 nvcc -O3 -std=c++14 cuda-worker\cuda-worker.cu -o cuda-worker\cuda-worker.exe...
nvcc -O3 -std=c++14 cuda-worker\cuda-worker.cu -o cuda-worker\cuda-worker.exe -I cuda-worker
if %ERRORLEVEL% NEQ 0 (
    echo ❌ CUDA compilation failed
    goto :error
) else (
    echo ✅ CUDA worker compiled successfully
)

:: Step 2: Compile WebAssembly Module (if Emscripten available)
echo.
echo [2/4] 🌐 Compiling WebAssembly GPU Module...
if defined SKIP_WASM (
    echo ⏭️ Skipping WebAssembly compilation (Emscripten not available)
) else (
    if not exist "static\wasm" mkdir static\wasm
    echo 🔨 emcc wasm\gpu-compute.cpp -O3 -s WASM=1 -o static\wasm\gpu-compute.js...
    emcc wasm\gpu-compute.cpp -O3 -s WASM=1 -s USE_WEBGPU=1 -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap']" -o static\wasm\gpu-compute.js
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ WebAssembly compilation failed
        goto :error
    ) else (
        echo ✅ WebAssembly module compiled successfully
    )
)

:: Step 3: Build Go GPU Orchestrator
echo.
echo [3/4] 🔧 Building Go GPU Orchestrator...
cd go-microservice
echo 🔨 go build gpu-orchestrator-service.go...
go mod tidy >nul 2>&1
go build -o gpu-orchestrator-service.exe gpu-orchestrator-service.go
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Go compilation failed
    cd ..
    goto :error
) else (
    echo ✅ GPU Orchestrator compiled successfully
)
cd ..

:: Step 4: Install Node.js dependencies
echo.
echo [4/4] 📦 Installing Node.js dependencies...
if exist "package.json" (
    echo 🔨 npm install...
    npm install >nul
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ npm install failed
        goto :error
    ) else (
        echo ✅ Node.js dependencies installed
    )
) else (
    echo ⚠️ No package.json found, skipping npm install
)

echo.
echo 🎯 GPU Pipeline Build Summary:
echo ===============================
echo ✅ CUDA Worker: cuda-worker\cuda-worker.exe
if not defined SKIP_WASM echo ✅ WebAssembly: static\wasm\gpu-compute.js
echo ✅ Go Orchestrator: go-microservice\gpu-orchestrator-service.exe
echo ✅ Node.js dependencies: Installed
echo.
echo 🚀 Build completed successfully!
echo.
echo 📋 Next Steps:
echo   1. Test the pipeline: test-gpu-pipeline.bat
echo   2. Start services: START-LEGAL-AI.bat (or npm run dev:full)
echo   3. Access GPU chat: http://localhost:5173/gpu-chat
echo.

goto :success

:error
echo.
echo ❌ Build failed! Please check the errors above.
echo.
echo 💡 Common solutions:
echo   • Install CUDA Toolkit for NVCC
echo   • Install Emscripten for WebAssembly (optional)
echo   • Install Go compiler
echo   • Check internet connection for downloads
echo.
exit /b 1

:success
echo ✅ GPU Pipeline build completed successfully!
exit /b 0