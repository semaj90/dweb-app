@echo off
setlocal enabledelayedexpansion

:: Auto-Start Script for MCP + Ollama System
echo ================================================
echo   Auto-Starting MCP + Ollama Optimal Setup
echo ================================================

:: Load environment configuration
if exist ".env.ollama" (
    for /f "tokens=1,2 delims==" %%a in (.env.ollama) do (
        set "%%a=%%b"
    )
    echo ✅ Loaded Ollama configuration: Port %OLLAMA_PORT%
) else (
    echo ⚠️ No Ollama configuration found, using defaults
    set "OLLAMA_PORT=11435"
)

:: Start MCP Server
echo 🔧 Starting MCP Server...
cd mcp
start /B node custom-context7-server.js
cd ..
timeout /t 3 /nobreak > nul

:: Start SvelteKit Dev Server
echo 🚀 Starting SvelteKit development server...
cd sveltekit-frontend
start /B npm run dev
cd ..

echo ✅ System starting up...
echo 🌐 Web interface will be available at: http://localhost:5173
echo 🤖 Ollama API available at: http://localhost:%OLLAMA_PORT%
echo 🔧 MCP Server running in background

echo.
echo 📋 Next steps:
echo 1. Wait 30 seconds for services to start
echo 2. Open VS Code in this directory
echo 3. Press Ctrl+Shift+P and type "Context7"
echo 4. Test MCP commands

pause
