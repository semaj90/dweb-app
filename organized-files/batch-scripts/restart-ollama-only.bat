@echo off
title Restart Ollama GPU Service
echo ======================================
echo    Restarting Ollama GPU Service
echo ======================================

echo [1/3] Stopping existing Ollama containers...
docker stop deeds-ollama-gpu 2>nul
docker rm deeds-ollama-gpu 2>nul

echo [2/3] Starting Ollama with GPU support...
docker-compose -f docker-compose-gpu.yml up -d ollama

echo [3/3] Waiting for Ollama to start...
timeout /t 10 /nobreak >nul

echo ======================================
echo [4/4] Testing Ollama connection...
docker exec deeds-ollama-gpu curl -f http://localhost:11434/api/version

echo.
echo [5/5] Checking models...
docker exec deeds-ollama-gpu ollama list

echo ======================================
echo     Ollama GPU setup complete!
echo ======================================
pause
