@echo off
title Download Smaller Model
echo Model too large - downloading Gemma 2B...
docker exec deeds-ollama-gpu ollama pull gemma2:2b

echo Creating legal version...
> Modelfile-2B (
echo FROM gemma2:2b
echo SYSTEM "Legal AI assistant for prosecutors."
echo PARAMETER temperature 0.3
echo PARAMETER num_ctx 2048
)

docker cp Modelfile-2B deeds-ollama-gpu:/tmp/
docker exec deeds-ollama-gpu ollama create gemma-legal-2b -f /tmp/Modelfile-2B

echo Testing 2B model...
docker exec deeds-ollama-gpu ollama run gemma-legal-2b "Legal test"
pause
