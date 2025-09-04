@echo off
REM Test script for Advanced CUDA Service
echo Testing Advanced CUDA Service...

REM Set environment
set CUDA_WORKER_PATH=C:\Users\james\Desktop\deeds-web\deeds-web-app\cuda-worker\cuda-worker.exe
set CUDA_GPU_MEMORY_LIMIT_GB=6
set ADVANCED_CUDA_PORT=8095

echo.
echo Starting service in background...
start "" advanced-cuda-service.exe

REM Wait for service to start
timeout /t 3 /nobreak >nul

echo Testing health endpoint...
curl -s http://localhost:8095/health | echo.

echo.
echo Testing attention endpoint...
curl -X POST http://localhost:8095/api/v1/attention -H "Content-Type: application/json" -d "{\"jobId\":\"test_001\",\"type\":\"attention\",\"text\":\"test input\",\"embeddings\":[1,2,3,4,5],\"useCache\":true,\"userId\":\"test_user\"}" | echo.

echo.
echo Testing cache stats...
curl -s http://localhost:8095/api/v1/cache/stats | echo.

echo.
echo Test complete. Press any key to stop the service...
pause

REM Stop the service (this is a simple approach)
taskkill /f /im advanced-cuda-service.exe 2>nul