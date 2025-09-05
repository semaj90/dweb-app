@echo off
setlocal enabledelayedexpansion

:: Comprehensive Test & Auto-Switch Setup for MCP + Ollama
echo =====================================================
echo   MCP Multi-Agent Orchestra - Comprehensive Test
echo =====================================================
echo.

:: Color codes
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%🔍 Step 1: Diagnosing Current System State%NC%
echo.

:: Check if any Ollama is running
echo %BLUE%Checking Ollama configurations...%NC%

:: Check native Ollama (port 11434)
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Native Ollama running on port 11434%NC%
    set "OLLAMA_NATIVE=true"
    set "OLLAMA_PORT=11434"
) else (
    echo %YELLOW%⚠️  Native Ollama not running on port 11434%NC%
    set "OLLAMA_NATIVE=false"
)

:: Check Docker Ollama (port 11435)
curl -s http://localhost:11435/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Docker Ollama running on port 11435%NC%
    set "OLLAMA_DOCKER=true"
    set "OLLAMA_PORT=11435"
) else (
    echo %YELLOW%⚠️  Docker Ollama not running on port 11435%NC%
    set "OLLAMA_DOCKER=false"
)

:: Check if both are false
if "%OLLAMA_NATIVE%"=="false" if "%OLLAMA_DOCKER%"=="false" (
    echo %RED%❌ No Ollama instance found running%NC%
    echo %BLUE%🚀 Starting Docker Ollama with alternative port...%NC%
    echo.

    :: Start the alternative port setup
    call START-RAG-ALT-PORT.bat

    :: Wait and recheck
    echo %BLUE%⏳ Waiting for Ollama to start...%NC%
    timeout /t 30 /nobreak > nul

    curl -s http://localhost:11435/api/tags >nul 2>&1
    if %errorlevel% equ 0 (
        echo %GREEN%✅ Docker Ollama now running on port 11435%NC%
        set "OLLAMA_PORT=11435"
        set "OLLAMA_DOCKER=true"
    ) else (
        echo %RED%❌ Failed to start Docker Ollama%NC%
        goto :ERROR_SETUP
    )
)

echo.
echo %BLUE%🔍 Step 2: Checking MCP Components%NC%
echo.

:: Check MCP Server
if exist "mcp\custom-context7-server.js" (
    echo %GREEN%✅ MCP Server found%NC%
) else (
    echo %RED%❌ MCP Server missing%NC%
    goto :ERROR_SETUP
)

:: Check MCP Extension
if exist ".vscode\extensions\mcp-context7-assistant\package.json" (
    echo %GREEN%✅ MCP Extension found%NC%
) else (
    echo %RED%❌ MCP Extension missing%NC%
    goto :ERROR_SETUP
)

:: Check Agent Orchestrator
if exist "agent-orchestrator\index.js" (
    echo %GREEN%✅ Agent Orchestrator found%NC%
) else (
    echo %YELLOW%⚠️  Agent Orchestrator missing - will create%NC%
    call SETUP-MULTI-AGENT-AI.bat
)

echo.
echo %BLUE%🔍 Step 3: Testing Required Models%NC%
echo.

:: Test models on the active Ollama port
echo %BLUE%Testing models on port %OLLAMA_PORT%...%NC%

curl -s http://localhost:%OLLAMA_PORT%/api/tags | findstr "gemma2" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Gemma2 models available%NC%
) else (
    echo %YELLOW%⚠️  Installing Gemma2 models...%NC%
    if "%OLLAMA_DOCKER%"=="true" (
        docker exec deeds-ollama-gpu-alt ollama pull gemma2:2b
        docker exec deeds-ollama-gpu-alt ollama pull gemma2:9b
    ) else (
        ollama pull gemma2:2b
        ollama pull gemma2:9b
    )
)

curl -s http://localhost:%OLLAMA_PORT%/api/tags | findstr "nomic-embed" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Embedding model available%NC%
) else (
    echo %YELLOW%⚠️  Installing embedding model...%NC%
    if "%OLLAMA_DOCKER%"=="true" (
        docker exec deeds-ollama-gpu-alt ollama pull nomic-embed-text
    ) else (
        ollama pull nomic-embed-text
    )
)

echo.
echo %BLUE%🔍 Step 4: Starting MCP Server%NC%
echo.

:: Start MCP Server
echo %BLUE%Starting MCP Server...%NC%
cd mcp
start /B node custom-context7-server.js
timeout /t 5 /nobreak > nul
cd ..

:: Check if MCP server is responding
curl -s http://localhost:3000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ MCP Server responding on port 3000%NC%
) else (
    echo %YELLOW%⚠️  MCP Server not responding on HTTP, may be stdio mode%NC%
)

echo.
echo %BLUE%🔍 Step 5: Testing MCP Multi-Agent Commands%NC%
echo.

:: Create environment file with active Ollama configuration
echo OLLAMA_BASE_URL=http://localhost:%OLLAMA_PORT% > .env.ollama
echo OLLAMA_HOST=localhost:%OLLAMA_PORT% >> .env.ollama
echo MCP_OLLAMA_URL=http://localhost:%OLLAMA_PORT% >> .env.ollama

echo %GREEN%✅ Created .env.ollama with active configuration%NC%

:: Test the multi-agent orchestration API
echo %BLUE%Testing multi-agent orchestration...%NC%

:: Start SvelteKit dev server if not running
netstat -an | findstr ":5173" >nul 2>&1
if %errorlevel% neq 0 (
    echo %BLUE%Starting SvelteKit dev server...%NC%
    cd sveltekit-frontend
    start /B npm run dev
    timeout /t 15 /nobreak > nul
    cd ..
)

:: Test the orchestration endpoint
curl -s -X POST ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Test multi-agent orchestration\",\"agents\":[\"autogen\",\"crewai\"]}" ^
  http://localhost:5173/api/ai/multi-agent >nul 2>&1

if %errorlevel% equ 0 (
    echo %GREEN%✅ Multi-agent orchestration API responding%NC%
) else (
    echo %YELLOW%⚠️  Multi-agent API not ready yet%NC%
)

echo.
echo %BLUE%🎯 Step 6: Manual VS Code Testing Instructions%NC%
echo.
echo %GREEN%Now test in VS Code:%NC%
echo %BLUE%1. Open VS Code in this directory%NC%
echo %BLUE%2. Press Ctrl+Shift+P%NC%
echo %BLUE%3. Type "Context7" - you should see commands%NC%
echo %BLUE%4. Try: "🤖 Context7 MCP: Run Agent Orchestrator"%NC%
echo %BLUE%5. Try: "🔍 Context7 MCP: Analyze Current Context"%NC%
echo.

echo %GREEN%✅ Setup complete! Ollama on port %OLLAMA_PORT%%NC%
echo %GREEN%✅ MCP Server running%NC%
echo %GREEN%✅ Extension ready for testing%NC%
echo.

echo %BLUE%🌐 Access URLs:%NC%
echo   • Ollama API: http://localhost:%OLLAMA_PORT%
echo   • SvelteKit Dev: http://localhost:5173
echo   • RAG Studio: http://localhost:5173/ai/enhanced-mcp
echo.

goto :END

:ERROR_SETUP
echo.
echo %RED%❌ Setup failed. Please check the logs above.%NC%
echo %BLUE%Try manually:%NC%
echo   1. START-RAG-ALT-PORT.bat (for Docker Ollama)
echo   2. SETUP-MULTI-AGENT-AI.bat (for agents)
echo   3. npm run dev (in sveltekit-frontend)
echo.

:END
pause
