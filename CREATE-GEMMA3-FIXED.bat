@echo off
title Create Gemma3 Legal Model - Fixed
echo [1/3] Copying files to container...
docker cp gemma3Q4_K_M/mo16.gguf deeds-ollama:/tmp/
docker cp Modelfile-Fixed deeds-ollama:/tmp/

echo [2/3] Creating gemma3-legal model...
docker exec deeds-ollama ollama create gemma3-legal -f /tmp/Modelfile-Fixed

echo [3/3] Verifying model...
docker exec deeds-ollama ollama list

echo ✅ Gemma3 legal model created
pause
