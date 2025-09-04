@echo off
REM Advanced CUDA Service Startup Script
REM Sets up environment and starts the CUDA service

echo Starting Advanced CUDA Service with proper environment...

REM Set CUDA worker path
set CUDA_WORKER_PATH=C:\Users\james\Desktop\deeds-web\deeds-web-app\cuda-worker\cuda-worker.exe

REM Set GPU memory limit (6GB for RTX 3060 Ti)
set CUDA_GPU_MEMORY_LIMIT_GB=6

REM Set service port
set ADVANCED_CUDA_PORT=8095

REM Enable GPU acceleration
set CUDA_VISIBLE_DEVICES=0

echo CUDA_WORKER_PATH=%CUDA_WORKER_PATH%
echo CUDA_GPU_MEMORY_LIMIT_GB=%CUDA_GPU_MEMORY_LIMIT_GB%
echo ADVANCED_CUDA_PORT=%ADVANCED_CUDA_PORT%
echo.

REM Check if CUDA worker exists
if not exist "%CUDA_WORKER_PATH%" (
    echo ERROR: CUDA worker not found at %CUDA_WORKER_PATH%
    echo Please build the CUDA worker first.
    pause
    exit /b 1
)

echo Testing CUDA worker...
echo {"jobId":"startup_test","type":"health_check","data":[1,2,3]} | "%CUDA_WORKER_PATH%"
echo.

echo Starting Advanced CUDA Service...
echo Press Ctrl+C to stop the service
echo.

REM Start the service
advanced-cuda-service.exe

pause