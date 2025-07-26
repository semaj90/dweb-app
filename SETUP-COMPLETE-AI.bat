@echo off
setlocal enabledelayedexpansion
title Complete AI Development Environment
color 0A

echo ========================================
echo COMPLETE AI DEVELOPMENT ENVIRONMENT
echo ========================================
echo.

echo This script will:
echo 1. Set up all Docker containers
echo 2. Load the 3GB Gemma3 legal model
echo 3. Install LangChain dependencies
echo 4. Create AI chat and analysis components
echo 5. Set up API endpoints
echo 6. Test everything
echo.
pause

echo [STEP 1] Running master setup...
call MASTER-SETUP.bat

echo [STEP 2] Implementing AI features...
call IMPLEMENT-AI-FEATURES.bat

echo [STEP 3] Testing integration...
call TEST-AI-INTEGRATION.bat

echo.
echo ========================================
echo AI ENVIRONMENT READY!
echo ========================================
echo.
echo 🎉 Your Legal AI system is fully operational:
echo.
echo 📊 Services running:
echo   - PostgreSQL with pgvector (port 5432)
echo   - Redis cache (port 6379)
echo   - Qdrant vector DB (port 6333)
echo   - RabbitMQ messaging (port 5672)
echo   - Ollama GPU with Gemma3 (port 11434)
echo.
echo 🤖 AI Features:
echo   - Legal AI Chat component
echo   - Evidence Analysis component
echo   - API endpoints for AI services
echo.
echo 🚀 Next steps:
echo   1. cd sveltekit-frontend
echo   2. npm run dev
echo   3. Open http://localhost:5173
echo   4. Test AI chat and evidence analysis
echo.
echo ✨ Happy coding with your Legal AI assistant!
pause
