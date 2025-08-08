@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

echo Building with GPU support...
set CGO_ENABLED=1
set CGO_CFLAGS=-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"
set CGO_LDFLAGS=-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" -lcudart -lcublas

go build -o ai-microservice.exe .

if %errorlevel% neq 0 (
    echo Build failed
    pause
    exit /b 1
)

echo Starting AI Microservice with GPU...
start /B ai-microservice.exe

timeout /t 3 /nobreak >nul

echo Testing connections...
curl -s http://localhost:7474 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Neo4j: http://localhost:7474
) else (
    echo [FAIL] Neo4j not responding
)

curl -s http://localhost:8081/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] AI Service: http://localhost:8081
    
    echo.
    echo Testing GPU endpoint...
    curl http://localhost:8081/gpu/metrics
) else (
    echo [FAIL] AI Service not responding
)

echo.
echo GPU System Ready
pause
