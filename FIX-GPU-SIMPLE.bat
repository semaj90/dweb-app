@echo off
title Quick GPU Fix for Ollama
echo ======================================
echo      RTX 3060 GPU Fix for Ollama
echo ======================================
echo.

echo [1/4] Stopping containers...
docker-compose down
timeout /t 3 /nobreak > nul

echo [2/4] Starting with GPU compose file...
docker-compose -f docker-compose-gpu.yml up -d
timeout /t 10 /nobreak > nul

echo [3/4] Applying memory optimization...
docker cp Modelfile-LowMem deeds-ollama-gpu:/tmp/
docker exec deeds-ollama-gpu ollama rm gemma3-legal
docker exec deeds-ollama-gpu ollama create gemma3-legal -f /tmp/Modelfile-LowMem

echo [4/4] Testing model...
docker exec deeds-ollama-gpu ollama list
docker exec deeds-ollama-gpu ollama run gemma3-legal "What is a legal brief?"

echo ======================================
echo      GPU setup complete!
echo ======================================
echo.
echo To use the model in your app:
echo 1. Make sure the model name in +server.js matches 'gemma3-legal'
echo 2. Run npm run db:migrate if you have schema changes
echo 3. Start your application with npm run dev
echo.
pause
