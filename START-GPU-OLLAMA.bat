@echo off
title Start GPU Ollama - Fixed
echo Starting GPU-accelerated Ollama...

echo Stopping current Ollama...
docker stop deeds-ollama

echo Starting with GPU support...
docker run -d --name deeds-ollama-gpu --gpus all -p 11434:11434 -v ollama_data:/root/.ollama --restart unless-stopped ollama/ollama:latest

echo Waiting for startup...
timeout /t 15 >nul

echo Loading model...
docker exec deeds-ollama-gpu ollama pull gemma3-legal 2>nul || (
    docker cp Modelfile-LowMem deeds-ollama-gpu:/tmp/
    docker cp gemma3Q4_K_M/mo16.gguf deeds-ollama-gpu:/tmp/
    docker exec deeds-ollama-gpu ollama create gemma3-legal -f /tmp/Modelfile-LowMem
)

echo Testing...
docker exec deeds-ollama-gpu ollama run gemma3-legal "Legal AI ready"

pause
