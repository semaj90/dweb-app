@echo off
title Fix Ollama & Load Gemma3
echo Downloading and loading Gemma3 model...

echo [1/3] Pulling Gemma3 model...
docker exec deeds-ollama-gpu ollama pull FROM ./gemma3Q4_K_M/mo16.gguf

echo [2/3] Creating legal variant...
docker exec deeds-ollama-gpu ollama create gemma3-legal -f -<<EOF
FROM ./gemma3Q4_K_M/mo16.gguf
SYSTEM You are a legal AI assistant for prosecutors. Provide accurate legal analysis and case recommendations.
EOF

echo [3/3] Testing model...
docker exec deeds-ollama-gpu ollama list

echo ✅ Gemma3 legal model ready
pause
