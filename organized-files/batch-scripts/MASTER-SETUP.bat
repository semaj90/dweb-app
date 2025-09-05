@echo off
setlocal enabledelayedexpansion
title Legal AI - Master Setup
color 0A

echo ========================================
echo LEGAL AI - MASTER SETUP (ERROR-FREE)
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "FRONTEND_PATH=%PROJECT_ROOT%sveltekit-frontend"

echo [1/8] Environment Check...
docker version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Docker Desktop not running
    echo Please start Docker Desktop first
    pause
    exit /b 1
)
echo ✅ Docker available

echo [2/8] Stopping existing containers...
docker stop deeds-postgres deeds-redis deeds-qdrant deeds-rabbitmq deeds-ollama-gpu 2>nul

echo [3/8] Starting unified stack...
docker-compose -f docker-compose-unified-gpu.yml up -d

echo [4/8] Waiting for services (30s)...
timeout /t 30 >nul

echo [5/8] Health checks...
docker exec deeds-postgres pg_isready -U legal_admin && echo ✅ PostgreSQL || echo ❌ PostgreSQL
docker exec deeds-redis redis-cli ping && echo ✅ Redis || echo ❌ Redis
curl -s http://localhost:6333/health >nul && echo ✅ Qdrant || echo ❌ Qdrant
curl -s http://localhost:11434/api/version >nul && echo ✅ Ollama || echo ❌ Ollama

echo [6/8] Loading AI model...
docker cp Modelfile-LowMem deeds-ollama-gpu:/tmp/
docker cp gemma3Q4_K_M/mo16.gguf deeds-ollama-gpu:/tmp/
docker exec deeds-ollama-gpu ollama create gemma3-legal -f /tmp/Modelfile-LowMem

echo [7/8] Installing AI dependencies...
cd "%FRONTEND_PATH%"
npm install langchain @langchain/ollama @langchain/postgres

echo [8/8] Creating AI services...
mkdir src\lib\services 2>nul
> src\lib\services\ai-service.ts (
echo import { Ollama } from '@langchain/ollama';
echo.
echo export class AIService {
echo   private llm = new Ollama({
echo     baseUrl: 'http://localhost:11434',
echo     model: 'gemma3-legal'
echo   }^);
echo.
echo   async analyze(text: string^) {
echo     return await this.llm.invoke(`Analyze this legal text: ${text}`^);
echo   }
echo }
)

cd "%PROJECT_ROOT%"
echo.
echo ✅ Master setup complete!
echo 🚀 Start development: npm run dev
pause
