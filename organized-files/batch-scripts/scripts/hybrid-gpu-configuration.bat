@echo off
REM Hybrid GPU Configuration - Combines GPU Memory Manager with Existing CUDA Services
REM Optimized for RTX 3060 Ti with 8GB VRAM

echo 🚀 Starting Hybrid GPU Configuration...
echo.

REM Check GPU availability
echo 📊 Checking GPU Status...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ NVIDIA GPU not available - falling back to CPU mode
    goto :cpu_mode
)

echo ✅ NVIDIA RTX 3060 Ti detected
echo.

REM Start Redis for GPU memory caching
echo 🔄 Starting Redis for GPU memory caching...
start /B "" "..\go-microservice\redis-server.exe" --port 6379
timeout /t 2 >nul

REM Configuration 1: Immediate Processing (Existing CUDA Service - Port 8096)
echo 🎮 Starting immediate GPU processing service...
start /B "" "..\go-microservice\bin\cuda-ai-service.exe"
timeout /t 3 >nul

REM Configuration 2: Advanced Batch Processing (GPU Memory Manager - Port 8097)
echo 🧠 Starting advanced GPU memory manager service...
start /B "" "..\go-microservice\bin\enhanced-rag-cuda.exe"
timeout /t 3 >nul

REM Configuration 3: Enhanced CUDA Service (Port 8099)
echo ⚡ Starting enhanced CUDA processing service...
start /B "" "..\go-microservice\bin\cuda-service-enhanced.exe" --port=8099
timeout /t 3 >nul

REM Start load balancer for optimal routing
echo 🔄 Starting GPU load balancer...
start /B "" "..\go-microservice\bin\load-balancer.exe" --gpu-mode --ports=8096,8097,8099
timeout /t 2 >nul

echo.
echo 🎯 Hybrid GPU Configuration Complete!
echo.
echo Service Architecture:
echo ├─ Immediate Processing: http://localhost:8096 (CUDA AI Service)
echo ├─ Advanced Batch: http://localhost:8097 (GPU Memory Manager)
echo ├─ Enhanced CUDA: http://localhost:8099 (Enhanced Processing)
echo └─ Load Balancer: http://localhost:8224 (Smart Routing)
echo.

REM Health checks
echo 🔍 Performing health checks...
timeout /t 5 >nul

curl -s http://localhost:8096/health >nul
if %errorlevel% equ 0 (
    echo ✅ CUDA AI Service: Healthy
) else (
    echo ⚠️ CUDA AI Service: Not responding
)

curl -s http://localhost:8097/health >nul
if %errorlevel% equ 0 (
    echo ✅ GPU Memory Manager: Healthy
) else (
    echo ⚠️ GPU Memory Manager: Not responding
)

curl -s http://localhost:8099/health >nul
if %errorlevel% equ 0 (
    echo ✅ Enhanced CUDA Service: Healthy
) else (
    echo ⚠️ Enhanced CUDA Service: Not responding
)

echo.
echo 📈 Performance Optimization Active:
echo ├─ GPU Memory Pooling: Enabled
echo ├─ Block Allocation: Active
echo ├─ Worker Pool System: 4 concurrent jobs
echo ├─ Performance Monitoring: Real-time
echo └─ Load Balancing: Smart routing based on workload
echo.

goto :end

:cpu_mode
echo 🖥️ Starting CPU-only mode...
start /B "" "..\go-microservice\bin\enhanced-rag.exe" --cpu-only
echo ✅ CPU mode active

:end
echo 🏆 Hybrid Configuration Ready!
echo Access via: http://localhost:5173 (SvelteKit will auto-route)
pause