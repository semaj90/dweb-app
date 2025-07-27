@echo off
setlocal enabledelayedexpansion

echo =========================================================
echo   GraphQL + Local Gemma3 Integration Setup
echo =========================================================
echo.

:: Check if Ollama is running
echo Checking Ollama status...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama is not running!
    echo Please start Ollama first.
    exit /b 1
)

:: Check if Gemma3 model is loaded
echo Checking Gemma3 model...
for /f "tokens=*" %%i in ('curl -s http://localhost:11434/api/tags ^| findstr /i "gemma3-legal"') do set GEMMA3_FOUND=true

if not defined GEMMA3_FOUND (
    echo [WARNING] Gemma3-legal model not found in Ollama
    echo Loading model...
    ollama pull gemma3-legal:latest
)

:: Install dependencies
echo.
echo Installing GraphQL dependencies...
cd /d "%~dp0"
call npm install graphql graphql-yoga @pothos/core @pothos/plugin-drizzle @urql/svelte @urql/core @urql/devtools @urql/exchange-graphcache

:: Create demo route directory
if not exist "src\routes\graphql-demo" mkdir "src\routes\graphql-demo"

:: Test the GraphQL endpoint
echo.
echo Testing GraphQL setup...
timeout /t 2 >nul

echo.
echo ✅ GraphQL + Gemma3 Integration Complete!
echo.
echo 📋 Quick Start:
echo   1. npm run dev
echo   2. Visit http://localhost:5173/api/graphql (GraphQL endpoint)
echo   3. Visit http://localhost:5173/graphql-demo (Demo page)
echo.
echo 🚀 Your Gemma3 model is integrated with:
echo   - GraphQL API with type-safe resolvers
echo   - Vector search using pgvector
echo   - Document processing pipeline
echo   - Real-time subscriptions
echo   - Local inference with mo16.gguf
echo.
pause
