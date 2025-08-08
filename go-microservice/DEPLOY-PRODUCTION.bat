@echo off
:: Production GPU+SIMD Legal Processor Deployment

:: Environment setup
set BUILD_DIR=build
set SERVICE_NAME=legal-processor-gpu
set CONFIG_FILE=.env.production

:: Validate environment
nvcc --version >nul 2>&1 || (echo CUDA required & exit /b 1)
clang --version >nul 2>&1 || (echo Clang required & exit /b 1)

:: Build production binary
call BUILD-GPU-SIMD-PRODUCTION.bat
if %ERRORLEVEL% NEQ 0 exit /b 1

:: Create deployment structure
mkdir %BUILD_DIR% 2>nul
copy legal-processor-gpu.exe %BUILD_DIR%\
copy %CONFIG_FILE% %BUILD_DIR%\
copy generated_reports %BUILD_DIR%\ /E /Y 2>nul

:: Register as Windows service (optional)
echo Creating Windows service...
sc create %SERVICE_NAME% binPath= "%cd%\%BUILD_DIR%\legal-processor-gpu.exe" start= auto
sc description %SERVICE_NAME% "GPU-accelerated legal document processor"

:: Start service
echo Starting service...
sc start %SERVICE_NAME%

:: Verify deployment
timeout /t 3 /nobreak >nul
curl -s http://localhost:8080/health | findstr "gpu_enabled" >nul && (
    echo ✅ Deployment successful
) || (
    echo ❌ Health check failed
    exit /b 1
)

echo 🎯 Service running at http://localhost:8080
echo 📊 Monitor: nvidia-smi
echo 🔧 Config: %BUILD_DIR%\%CONFIG_FILE%