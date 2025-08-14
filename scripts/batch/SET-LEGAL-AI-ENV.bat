@echo off
rem =============================================================================
rem SET-LEGAL-AI-ENV.bat
rem Configure environment variables for local AI models
rem =============================================================================

echo Setting Legal AI environment variables...

rem Model Configuration
set LEGAL_MODEL=gemma3-legal
set EMBEDDING_MODEL=nomic-embed-text
set VISION_MODEL=llava:7b

rem Service URLs
set OLLAMA_URL=http://localhost:11434
set GO_SERVER_URL=http://localhost:8081
set POSTGRES_URL=postgres://postgres:password@localhost/legal_ai?sslmode=disable
set REDIS_URL=localhost:6379

rem GPU Configuration
set CUDA_AVAILABLE=true
set GPU_MEMORY_LIMIT=12GB

rem Ollama Configuration
set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_KEEP_ALIVE=5m
set OLLAMA_NUM_PARALLEL=4

rem Processing Options
set ENABLE_GPU_ACCELERATION=true
set ENABLE_LEGAL_MODEL=true
set MODEL_TEMPERATURE=0.3
set MODEL_CONTEXT_SIZE=4096

echo.
echo Environment configured for local models:
echo   - Legal Model: %LEGAL_MODEL%
echo   - Embedding Model: %EMBEDDING_MODEL%
echo   - Ollama URL: %OLLAMA_URL%
echo   - GPU Acceleration: %ENABLE_GPU_ACCELERATION%
echo.
