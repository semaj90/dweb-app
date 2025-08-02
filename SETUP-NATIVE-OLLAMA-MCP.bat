@echo off
setlocal enabledelayedexpansion

:: Native Windows Ollama + MCP Test Setup
echo =====================================================
echo   Native Windows Ollama + MCP Setup & Test
echo =====================================================
echo.

:: Color codes
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%🔍 Step 1: Checking for Native Windows Ollama%NC%
echo.

:: Check if Ollama is installed on Windows
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Native Windows Ollama found%NC%
    set "OLLAMA_AVAILABLE=true"
) else (
    echo %YELLOW%⚠️  Native Windows Ollama not found%NC%
    echo %BLUE%📥 Downloading and installing Ollama for Windows...%NC%

    :: Download Ollama for Windows
    echo Downloading Ollama installer...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://ollama.ai/download/OllamaSetup.exe' -OutFile 'OllamaSetup.exe'}"

    if exist "OllamaSetup.exe" (
        echo %GREEN%✅ Downloaded Ollama installer%NC%
        echo %BLUE%🚀 Please run OllamaSetup.exe and restart this script%NC%
        start OllamaSetup.exe
        pause
        exit /b 1
    ) else (
        echo %RED%❌ Failed to download Ollama%NC%
        echo %BLUE%💡 Please manually install from: https://ollama.ai%NC%
        pause
        exit /b 1
    )
)

:: Check if Ollama service is running
echo %BLUE%Checking Ollama service...%NC%
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Ollama service running on port 11434%NC%
) else (
    echo %YELLOW%⚠️  Starting Ollama service...%NC%
    start /B ollama serve
    timeout /t 10 /nobreak > nul

    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% equ 0 (
        echo %GREEN%✅ Ollama service now running%NC%
    ) else (
        echo %RED%❌ Failed to start Ollama service%NC%
        echo %BLUE%💡 Try manually: ollama serve%NC%
        pause
        exit /b 1
    )
)

echo.
echo %BLUE%🔍 Step 2: Installing Required Models%NC%
echo.

:: Install required models
echo %BLUE%Installing Gemma2:2b model...%NC%
ollama pull gemma2:2b
if %errorlevel% equ 0 (
    echo %GREEN%✅ Gemma2:2b installed%NC%
) else (
    echo %YELLOW%⚠️  Gemma2:2b installation issue%NC%
)

echo %BLUE%Installing embedding model...%NC%
ollama pull nomic-embed-text
if %errorlevel% equ 0 (
    echo %GREEN%✅ Embedding model installed%NC%
) else (
    echo %YELLOW%⚠️  Embedding model installation issue%NC%
)

:: Create legal-specialized model
echo %BLUE%Creating legal-specialized model...%NC%
echo FROM gemma2:2b > Modelfile.legal
echo. >> Modelfile.legal
echo TEMPLATE """^<^|im_start^|^>system >> Modelfile.legal
echo {{ .System }}^<^|im_end^|^> >> Modelfile.legal
echo ^<^|im_start^|^>user >> Modelfile.legal
echo {{ .Prompt }}^<^|im_end^|^> >> Modelfile.legal
echo ^<^|im_start^|^>assistant >> Modelfile.legal
echo """ >> Modelfile.legal
echo. >> Modelfile.legal
echo SYSTEM """You are a specialized legal AI assistant for prosecutors and legal professionals. You excel at evidence analysis, case timeline construction, legal precedent research, witness statement evaluation, strategic prosecution planning, and document review. Provide concise, actionable legal insights with proper citations and maintain professional standards.""" >> Modelfile.legal
echo. >> Modelfile.legal
echo PARAMETER temperature 0.2 >> Modelfile.legal
echo PARAMETER top_k 30 >> Modelfile.legal
echo PARAMETER top_p 0.8 >> Modelfile.legal
echo PARAMETER repeat_penalty 1.15 >> Modelfile.legal
echo PARAMETER num_ctx 8192 >> Modelfile.legal
echo PARAMETER num_predict 1024 >> Modelfile.legal

ollama create gemma2-legal -f Modelfile.legal
if %errorlevel% equ 0 (
    echo %GREEN%✅ Legal model created%NC%
) else (
    echo %YELLOW%⚠️  Legal model creation issue%NC%
)

del Modelfile.legal >nul 2>&1

echo.
echo %BLUE%🔍 Step 3: Testing Models%NC%
echo.

:: Test basic model
echo %BLUE%Testing Gemma2:2b...%NC%
echo What is artificial intelligence? | ollama run gemma2:2b --verbose=false | head -n 3

:: Test legal model
echo %BLUE%Testing legal model...%NC%
echo What are the elements of probable cause? | ollama run gemma2-legal --verbose=false | head -n 3

echo.
echo %BLUE%🔍 Step 4: Configuring MCP Environment%NC%
echo.

:: Create environment configuration
echo OLLAMA_BASE_URL=http://localhost:11434 > .env.ollama
echo OLLAMA_HOST=localhost:11434 >> .env.ollama
echo MCP_OLLAMA_URL=http://localhost:11434 >> .env.ollama
echo OLLAMA_PORT=11434 >> .env.ollama
echo OLLAMA_TYPE=native >> .env.ollama

echo %GREEN%✅ Created .env.ollama configuration%NC%

echo.
echo %BLUE%🔍 Step 5: Starting MCP Server%NC%
echo.

:: Start MCP Server
echo %BLUE%Starting MCP Server...%NC%
cd mcp
start /B "MCP Server" node custom-context7-server.js
timeout /t 5 /nobreak > nul
cd ..

:: Check if MCP server is responding (stdio mode doesn't respond to HTTP)
tasklist | findstr "node.exe" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ MCP Server process running%NC%
) else (
    echo %YELLOW%⚠️  MCP Server may not be running%NC%
)

echo.
echo %BLUE%🔍 Step 6: Starting SvelteKit Dev Server%NC%
echo.

:: Check if SvelteKit is already running
netstat -an | findstr ":5173" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ SvelteKit already running on port 5173%NC%
) else (
    echo %BLUE%Starting SvelteKit dev server...%NC%
    cd sveltekit-frontend
    start /B "SvelteKit Dev" npm run dev
    timeout /t 15 /nobreak > nul
    cd ..

    netstat -an | findstr ":5173" >nul 2>&1
    if %errorlevel% equ 0 (
        echo %GREEN%✅ SvelteKit now running on port 5173%NC%
    ) else (
        echo %YELLOW%⚠️  SvelteKit may still be starting%NC%
    )
)

echo.
echo %BLUE%🔍 Step 7: Testing MCP Multi-Agent Commands%NC%
echo.

:: Test the multi-agent orchestration API
echo %BLUE%Testing MCP multi-agent API...%NC%
timeout /t 5 /nobreak > nul

curl -s -X POST ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Analyze this legal case\",\"agents\":[\"claude\",\"copilot\"]}" ^
  http://localhost:5173/api/ai/multi-agent 2>nul | findstr "result" >nul 2>&1

if %errorlevel% equ 0 (
    echo %GREEN%✅ Multi-agent API responding%NC%
) else (
    echo %YELLOW%⚠️  Multi-agent API not ready (this is normal if SvelteKit is still starting)%NC%
)

echo.
echo %GREEN%✅ SETUP COMPLETE!%NC%
echo.
echo %BLUE%🎯 System Status:%NC%
echo   • Native Ollama: Running on port 11434
echo   • Models: gemma2:2b, nomic-embed-text, gemma2-legal
echo   • MCP Server: Running in background
echo   • SvelteKit: Starting/Running on port 5173
echo.
echo %BLUE%🧪 Next Steps - Test in VS Code:%NC%
echo   1. Open VS Code in this directory
echo   2. Press Ctrl+Shift+P
echo   3. Type "Context7" - you should see commands
echo   4. Try "🤖 Context7 MCP: Run Agent Orchestrator"
echo   5. Try "🔍 Context7 MCP: Analyze Current Context"
echo.
echo %BLUE%🌐 Web Access:%NC%
echo   • Main App: http://localhost:5173
echo   • RAG Studio: http://localhost:5173/ai/enhanced-mcp
echo   • Ollama API: http://localhost:11434
echo.
echo %GREEN%🎉 MCP Multi-Agent Orchestra is now ready!%NC%
echo %BLUE%The orchestration failure should be fixed with native Ollama.%NC%
echo.

pause
