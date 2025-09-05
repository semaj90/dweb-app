@echo off
title Fix Ollama Memory Issue
echo Recreating model with lower memory...
docker cp Modelfile-LowMem deeds-ollama:/tmp/
docker exec deeds-ollama ollama rm gemma3-legal
docker exec deeds-ollama ollama create gemma3-legal -f /tmp/Modelfile-LowMem
echo Testing...
docker exec deeds-ollama ollama run gemma3-legal "Test"
pause
