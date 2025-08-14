@echo off
title Fixed GPU Setup for Legal AI
echo ======================================
echo      RTX 3060 Legal AI Setup
echo ======================================
echo.

echo [1/5] Starting Ollama container with GPU...
docker run -d --gpus=all -v ollama_data:/root/.ollama -p 11434:11434 --name deeds-ollama-gpu ollama/ollama
timeout /t 5 /nobreak > nul

echo [2/5] Waiting for Ollama to initialize...
timeout /t 10 /nobreak > nul

echo [3/5] Pulling Gemma3 model (this may take a while)...
docker exec deeds-ollama-gpu ollama pull gemma3:latest

echo [4/5] Creating legal-optimized Modelfile...
docker exec deeds-ollama-gpu sh -c "cat > /tmp/Modelfile-Legal << 'EOF'
FROM gemma3:latest

TEMPLATE \"\"\"<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
\"\"\"

SYSTEM \"\"\"You are a specialized legal AI assistant for prosecutors. Focus on:
- Evidence analysis and case timeline construction
- Legal precedent research and citation
- Witness statement evaluation
- Strategic prosecution planning
- Document summarization and review

Provide concise, actionable legal insights while maintaining professional standards and confidentiality.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop \"<start_of_turn>\"
PARAMETER stop \"<end_of_turn>\"
PARAMETER num_ctx 4096
PARAMETER num_predict 512
EOF"

echo [5/5] Creating legal AI model...
docker exec deeds-ollama-gpu ollama create gemma3-legal -f /tmp/Modelfile-Legal

echo.
echo ======================================
echo      Testing Legal AI Model
echo ======================================
docker exec deeds-ollama-gpu ollama list
docker exec deeds-ollama-gpu ollama run gemma3-legal "What are the key elements of probable cause?"

echo.
echo ======================================
echo      Setup Complete!
echo ======================================
echo.
echo Model: gemma3-legal
echo Endpoint: http://localhost:11434
echo.
echo To test in your app:
echo 1. Update your API to use model name 'gemma3-legal'
echo 2. Start SvelteKit: npm run dev
echo 3. Visit: http://localhost:5174/ai-test
echo.
pause
