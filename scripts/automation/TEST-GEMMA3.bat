@echo off
title Test Gemma3 Legal AI
echo Testing AI model...
docker exec deeds-ollama ollama run gemma3-legal "Analyze the key elements needed for a strong evidence timeline in a criminal prosecution case."
pause
