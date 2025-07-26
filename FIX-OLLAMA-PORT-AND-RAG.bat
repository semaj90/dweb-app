@echo off
title 🔧 Fix Ollama Port Conflict and Start RAG System
color 0A

echo.
echo ================================================================
echo  🔧 COMPREHENSIVE OLLAMA PORT FIX AND RAG SYSTEM STARTUP
echo ================================================================
echo.

echo [1/8] 🔍 Checking current port 11434 usage...
netstat -ano | findstr :11434
echo.

echo [2/8] ⏹️ Stopping native Ollama processes...
tasklist | findstr ollama.exe
echo Killing all Ollama processes...
taskkill /F /IM ollama.exe /T 2>nul
timeout /t 2 >nul

echo Stopping Ollama Windows service if running...
sc stop ollama 2>nul
timeout /t 3 >nul

echo [3/8] 🐳 Stopping Docker containers...
docker stop ollama-gpu ollama-container postgres-db redis-cache rabbitmq-queue qdrant-vector 2>nul
docker rm ollama-gpu ollama-container 2>nul
timeout /t 5 >nul

echo [4/8] 🧹 Cleaning up Docker volumes and networks...
docker volume prune -f
docker network prune -f

echo [5/8] 🚀 Starting fresh Docker Ollama on port 11434...
docker run -d ^
  --name ollama-gpu ^
  --restart unless-stopped ^
  --gpus all ^
  -p 11434:11434 ^
  -v ollama-data:/root/.ollama ^
  -e OLLAMA_HOST=0.0.0.0 ^
  -e OLLAMA_ORIGINS=* ^
  -e NVIDIA_VISIBLE_DEVICES=all ^
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility ^
  ollama/ollama:latest

echo Waiting for Ollama to start...
timeout /t 10 >nul

echo [6/8] 📥 Pulling required models...
docker exec ollama-gpu ollama pull gemma2:9b
echo.
docker exec ollama-gpu ollama pull nomic-embed-text
echo.

echo [7/8] 🏗️ Creating custom legal model...
echo Creating Modelfile...
docker exec ollama-gpu sh -c "cat > /tmp/Modelfile << 'EOF'
FROM gemma2:9b

TEMPLATE \"\"\"<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are a specialized legal AI assistant with expertise in legal document analysis, case law research, and legal writing. You provide accurate, well-researched legal information while always reminding users to consult with qualified legal professionals for specific legal advice.

Key capabilities:
- Legal document analysis and review
- Case law research and citation
- Legal writing assistance
- Contract analysis
- Regulatory compliance guidance
- Legal research methodology

Always provide sources and citations when possible, and maintain professional legal communication standards.\"\"\"
EOF"

echo Creating gemma3-legal-enhanced model...
docker exec ollama-gpu ollama create gemma3-legal-enhanced -f /tmp/Modelfile
echo.

echo [8/8] 🔍 Final system check...
echo Checking Docker containers:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

echo Checking Ollama API:
timeout /t 3 >nul
curl -s http://localhost:11434/api/tags | findstr "gemma"
echo.

echo Testing model availability:
docker exec ollama-gpu ollama list
echo.

echo ================================================================
echo  ✅ OLLAMA PORT FIX AND RAG SYSTEM SETUP COMPLETE
echo ================================================================
echo.
echo 🚀 Your system should now be ready with:
echo    • Docker Ollama running on port 11434
echo    • gemma2:9b and nomic-embed-text models
echo    • Custom gemma3-legal-enhanced model
echo.
echo 🔗 Test the API: http://localhost:11434/api/tags
echo 🔗 Start your SvelteKit app: npm run dev
echo.
echo Press any key to exit...
pause >nul
