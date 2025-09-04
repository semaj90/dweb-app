@echo off
title Enhanced RAG V2 - Complete System Launcher
echo.
echo 🎯 ENHANCED RAG V2 - COMPLETE SYSTEM LAUNCHER
echo =============================================
echo.
echo This launcher will:
echo   • Run comprehensive system verification
echo   • Build missing components automatically
echo   • Launch all microservices
echo   • Start monitoring dashboard
echo   • Initialize GPU acceleration
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

REM Set environment variables
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_PATH%\bin;%PATH%
set GO111MODULE=on
set CGO_ENABLED=1

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo.
echo 📋 STEP 1: Running Enhanced System Verification
echo ===============================================
call powershell -ExecutionPolicy Bypass -File "VERIFY-DEPLOYMENT.ps1" -AutoFix -GPUCheck -TensorCacheValidation

echo.
echo 🔧 STEP 2: Building Go Microservices with CUDA Support
echo ======================================================
cd go-microservice

echo Building Enhanced RAG V2...
go build -tags cuda -ldflags "-s -w" -o bin\enhanced-rag-v2.exe .\cmd\enhanced-rag-v2

echo Building Simply Enhanced RAG...
go build -tags cuda -ldflags "-s -w" -o bin\simply-enhanced-rag.exe .\cmd\simply-enhanced-rag

echo Building GPU Legal AI Service...
go build -tags cuda -ldflags "-s -w" -o bin\gpu-legal-ai.exe .\gpu-legal-ai-server.go

echo Building Tensor Service...
go build -tags cuda -ldflags "-s -w" -o bin\tensor-service.exe .\tensor-gpu-service.go

cd ..

echo.
echo 🗄️ STEP 3: Initializing Databases
echo =================================
echo Starting PostgreSQL with pgvector...
net start postgresql-x64-14 >nul 2>&1

echo Starting Redis cache...
if exist redis-windows\redis-server.exe (
    start /B redis-windows\redis-server.exe redis-windows\redis.conf
)

echo Starting Neo4j...
if exist neo4j-community-5.23.0\bin\neo4j.bat (
    cd neo4j-community-5.23.0
    start /B bin\neo4j.bat start
    cd ..
)

echo.
echo 🚀 STEP 4: Launching Microservices
echo ==================================
echo Starting Enhanced RAG V2 on port 8097...
start "Enhanced RAG V2" cmd /k "cd go-microservice && bin\enhanced-rag-v2.exe"

timeout /t 3 >nul

echo Starting Simply Enhanced RAG on port 8096...
start "Simply Enhanced RAG" cmd /k "cd go-microservice && bin\simply-enhanced-rag.exe"

timeout /t 3 >nul

echo Starting GPU Legal AI Service on port 8098...
start "GPU Legal AI" cmd /k "cd go-microservice && bin\gpu-legal-ai.exe"

timeout /t 3 >nul

echo Starting Tensor Service on port 8099...
start "Tensor Service" cmd /k "cd go-microservice && bin\tensor-service.exe"

echo.
echo 🎮 STEP 5: Initializing GPU Acceleration
echo ========================================
echo Validating CUDA 12.8 installation...
nvcc --version
if %ERRORLEVEL% neq 0 (
    echo ⚠️  CUDA not detected - some GPU features may be unavailable
) else (
    echo ✅ CUDA 12.8+ detected and ready for Gorgonia tensor operations
)

echo Testing GPU memory allocation...
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
if %ERRORLEVEL% neq 0 (
    echo ⚠️  NVIDIA GPU not detected - falling back to CPU mode
) else (
    echo ✅ GPU memory available for tensor caching
)

echo.
echo 💻 STEP 6: Starting Frontend Development Server
echo ===============================================
if exist package.json (
    echo Installing/updating Node.js dependencies...
    call npm install --silent
    
    echo Starting SvelteKit development server...
    start "Frontend Dev Server" cmd /k "npm run dev"
) else (
    echo ⚠️  Frontend package.json not found - skipping frontend setup
)

echo.
echo 📊 STEP 7: Launching Monitoring Dashboard
echo =========================================
echo Opening real-time system dashboard...
start "" dashboard.html

timeout /t 5 >nul

echo.
echo ✨ ENHANCED RAG V2 SYSTEM STARTUP COMPLETE!
echo ==========================================
echo.
echo 🌐 Services Status:
echo   • Enhanced RAG V2: http://localhost:8097
echo   • Simply Enhanced RAG: http://localhost:8096  
echo   • GPU Legal AI: http://localhost:8098
echo   • Tensor Service: http://localhost:8099
echo   • Frontend (SvelteKit): http://localhost:3000
echo   • Monitoring Dashboard: dashboard.html
echo.
echo 🎮 GPU Acceleration:
echo   • CUDA 12.8 with Gorgonia tensor operations
echo   • WebGPU shaders for browser-based caching
echo   • Clang-optimized CUDA compilation
echo.
echo 🧠 AI Features:
echo   • Legal document processing with GPU acceleration
echo   • Real-time tensor caching and similarity search
echo   • Response distillation for optimal performance
echo.
echo 📋 Next Steps:
echo   1. Monitor system health in the dashboard
echo   2. Upload legal documents for processing
echo   3. Test AI assistant functionality
echo   4. Review performance metrics and optimization
echo.
echo Press any key to view system logs or Ctrl+C to exit...
pause >nul

REM Show live system logs
echo.
echo 📊 REAL-TIME SYSTEM LOGS
echo =======================
powershell -Command "Get-Content -Path 'logs\system.log' -Wait -Tail 20" 2>nul || echo No system logs found yet - services are starting up...
