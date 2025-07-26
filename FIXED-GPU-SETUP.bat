@echo off
title Fixed CUDA & GPU Setup
echo Fixing CUDA detection...

echo Detecting CUDA version...
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*" (
    echo ✅ CUDA 12 detected - Perfect for Ollama
) else (
    echo ⚠️ CUDA version check failed
)

echo Stopping old containers...
docker stop deeds-ollama deeds-ollama-gpu 2>nul

echo Starting GPU Ollama...
docker run -d --name deeds-ollama-gpu --gpus all -p 11434:11434 -v ollama_data:/root/.ollama --restart unless-stopped ollama/ollama:latest

echo Waiting for startup...
timeout /t 15 >nul

echo Creating model from local file...
docker cp Modelfile-LowMem deeds-ollama-gpu:/tmp/
docker cp gemma3Q4_K_M/mo16.gguf deeds-ollama-gpu:/tmp/
docker exec deeds-ollama-gpu ollama create gemma3-legal -f /tmp/Modelfile-LowMem

echo Testing model...
docker exec deeds-ollama-gpu ollama run gemma3-legal "Legal AI test"

pause
