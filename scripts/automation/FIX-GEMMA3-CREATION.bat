@echo off
title Fix Gemma3 Model Creation
echo Copying modelfile to container...
docker cp Gemma3-Legal-Enhanced-Modelfile-v2 deeds-ollama:/tmp/
docker cp gemma3Q4_K_M/mo16.gguf deeds-ollama:/tmp/

echo Creating model with correct paths...
docker exec deeds-ollama ollama create gemma3-legal -f /tmp/Gemma3-Legal-Enhanced-Modelfile-v2

echo Testing model...
docker exec deeds-ollama ollama list
pause
