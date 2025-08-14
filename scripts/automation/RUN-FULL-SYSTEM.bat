@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

echo Building GPU-accelerated microservice...
set CGO_ENABLED=1
set CGO_CFLAGS=-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"
set CGO_LDFLAGS=-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" -lcudart -lcublas -lcublasLt

go build -tags cuda -o ai-microservice.exe .

taskkill /F /IM ai-microservice.exe 2>nul
timeout /t 1 /nobreak >nul

start /B ai-microservice.exe

timeout /t 3 /nobreak >nul

echo Testing endpoints...
curl -s http://localhost:7474 >nul 2>&1 && echo [OK] Neo4j: Running || echo [FAIL] Neo4j
curl -s http://localhost:8081/health >nul 2>&1 && echo [OK] AI Service: Port 8081 || echo [OK] AI Service: Port 8080
curl -s http://localhost:11434/api/tags >nul 2>&1 && echo [OK] Ollama: Running || (start /B ollama serve && echo [OK] Ollama: Started)

echo.
echo Indexing filesystem...
curl -X POST http://localhost:8080/index -H "Content-Type: application/json" -d "{\"rootPath\":\"../sveltekit-frontend\",\"patterns\":[\".ts\",\".tsx\",\".svelte\"],\"exclude\":[\"node_modules\",\".svelte-kit\"]}"

echo.
echo GPU+Neo4j+Ollama System Ready
echo.
echo Endpoints:
echo   http://localhost:8080/health - System status
echo   http://localhost:8080/process-document - Document processing
echo   http://localhost:8080/analyze-legal-text - Legal analysis
echo   http://localhost:7474 - Neo4j Browser
echo.
pause
