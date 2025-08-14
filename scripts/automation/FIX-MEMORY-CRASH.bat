@echo off
title Fix Memory Issue
echo Recreating with ultra-light config...
docker exec deeds-ollama-gpu ollama rm gemma3-legal
docker cp Modelfile-Ultra-Light deeds-ollama-gpu:/tmp/
docker exec deeds-ollama-gpu ollama create gemma3-legal -f /tmp/Modelfile-Ultra-Light
echo Testing...
docker exec deeds-ollama-gpu ollama run gemma3-legal "Test"
pause
