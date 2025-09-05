@echo off
title Quick Fix - Create Gemma3 Model
echo Creating gemma3-legal model...
docker exec deeds-ollama ollama create gemma3-legal -f Gemma3-Legal-Enhanced-Modelfile-v2
echo ✅ Gemma3 legal model created
pause
