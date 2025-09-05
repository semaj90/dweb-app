@echo off
echo ================================
echo GPU-Accelerated Legal AI Service
echo RTX 3060 Ti Optimized
echo ================================
echo.

REM Set GPU optimization environment variables
set CUDA_VISIBLE_DEVICES=0
set TF_FORCE_GPU_ALLOW_GROWTH=true
set OLLAMA_NUM_GPU=1
set OLLAMA_GPU_LAYERS=32

REM Configure service parameters
set REDIS_ADDR=localhost:6379
set POSTGRES_URL=postgres://postgres:postgres@localhost:5432/legal_ai_db
set PORT=8084
set OLLAMA_URL=http://localhost:11434
set GEMMA_MODEL_NAME=gemma3-legal:latest
set MAX_CONCURRENCY=3
set CACHE_EXPIRATION_MINUTES=30
set ENABLE_GPU=true
set MODEL_CONTEXT=4096
set TEMPERATURE=0.2
set STREAMING_ENABLED=true
set BATCH_SIZE=5
set GPU_MEMORY_LIMIT_MB=6000

echo [*] Starting Redis cache...
start /B redis-server

echo [*] Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Starting Ollama service...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
)

echo [*] Loading Gemma3-Legal model into GPU memory...
ollama run gemma3-legal:latest "Ready" --verbose

echo [*] Starting GPU-Accelerated Legal AI Service...
echo.
echo [+] Service URL: http://localhost:8084
echo [+] Health Check: http://localhost:8084/api/health
echo [+] Metrics: http://localhost:8084/api/metrics
echo [+] AI Summarization: http://localhost:8084/api/ai/summarize
echo.

REM Start the Go service with GPU optimizations
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"
go run main.go

pause