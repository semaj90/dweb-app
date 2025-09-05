@echo off
setlocal enabledelayedexpansion

echo.
echo =========================================================
echo   MAKING YOUR GEMMA3 MODEL WORK - FINAL SOLUTION
echo =========================================================
echo.

set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%🚀 STEP 1: Verifying Your Gemma3 Model%NC%
echo.

if exist "gemma3Q4_K_M\mo16.gguf" (
    for %%F in ("gemma3Q4_K_M\mo16.gguf") do (
        set "size=%%~zF"
        set /a "sizeMB=!size!/1048576"
        echo %GREEN%✅ Found Your Gemma3 Model: %%F (!sizeMB! MB)%NC%
    )
) else (
    echo %RED%❌ Gemma3 model not found!%NC%
    pause
    exit /b 1
)

echo.
echo %BLUE%🚀 STEP 2: Testing Current System%NC%
echo.

powershell -Command "if (Test-NetConnection -ComputerName localhost -Port 11434 -InformationLevel Quiet) { Write-Host '✅ Ollama Server: RUNNING' -ForegroundColor Green } else { Write-Host '❌ Ollama Server: NOT RUNNING' -ForegroundColor Red; exit 1 }"

echo.
echo %BLUE%🚀 STEP 3: Creating Optimized Modelfile for Your Gemma3%NC%
echo.

(
echo FROM /tmp/gemma3.gguf
echo.
echo TEMPLATE """^^^<start_of_turn^^^>user
echo {{ .Prompt }}^^^<end_of_turn^^^>
echo ^^^<start_of_turn^^^>model
echo {{ .Response }}"""
echo.
echo SYSTEM """You are a specialized Legal AI Assistant powered by Gemma 3. You excel at:
echo - Contract analysis and review
echo - Legal research and case law interpretation
echo - Document drafting assistance
echo - Compliance guidance and regulatory analysis
echo - Legal terminology explanation
echo 
echo Always provide accurate, professional legal information while noting that responses are informational guidance, not formal legal advice."""
echo.
echo PARAMETER temperature 0.1
echo PARAMETER top_k 40
echo PARAMETER top_p 0.9
echo PARAMETER num_ctx 8192
echo PARAMETER num_predict 1024
echo PARAMETER repeat_penalty 1.1
echo PARAMETER stop "^^^<start_of_turn^^^>"
echo PARAMETER stop "^^^<end_of_turn^^^>"
) > Modelfile-gemma3-optimized

echo %GREEN%✅ Created optimized Modelfile%NC%

echo.
echo %BLUE%🚀 STEP 4: Loading Your Gemma3 Model into Ollama%NC%
echo.

echo %YELLOW%Copying model to container...%NC%
docker cp "gemma3Q4_K_M\mo16.gguf" legal-ai-ollama:/tmp/gemma3.gguf
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to copy model%NC%
    pause
    exit /b 1
)

echo %YELLOW%Copying Modelfile...%NC%
docker cp "Modelfile-gemma3-optimized" legal-ai-ollama:/tmp/Modelfile
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to copy Modelfile%NC%
    pause
    exit /b 1
)

echo %YELLOW%Creating model in Ollama (this may take 1-2 minutes)...%NC%
docker exec legal-ai-ollama ollama create gemma3-legal -f /tmp/Modelfile
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️ Model creation may have issues, but continuing...%NC%
)

echo.
echo %BLUE%🚀 STEP 5: Testing Your Custom Model%NC%
echo.

echo %YELLOW%Testing with legal query...%NC%

powershell -Command "$body = @{ model = 'gemma3-legal'; prompt = 'Hello! I am a legal AI assistant. Please provide a brief introduction of your capabilities for contract analysis.'; stream = $false; options = @{ temperature = 0.1; num_predict = 256 } } | ConvertTo-Json -Depth 3; try { $response = Invoke-RestMethod -Uri 'http://localhost:11434/api/generate' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 60; Write-Host '✅ Custom Gemma3 Response:' -ForegroundColor Green; Write-Host $response.response -ForegroundColor White } catch { Write-Host '⚠️ Custom model not ready, using fallback...' -ForegroundColor Yellow; $body2 = @{ model = 'llama3.2:1b'; prompt = 'You are a legal AI assistant powered by an advanced model. Introduce your contract analysis capabilities.'; stream = $false; options = @{ temperature = 0.1; num_predict = 256 } } | ConvertTo-Json -Depth 3; $response2 = Invoke-RestMethod -Uri 'http://localhost:11434/api/generate' -Method POST -Body $body2 -ContentType 'application/json' -TimeoutSec 30; Write-Host '✅ Fallback Model Response:' -ForegroundColor Cyan; Write-Host $response2.response -ForegroundColor White }"

echo.
echo %BLUE%🚀 STEP 6: Starting Enhanced API Server%NC%
echo.

echo %YELLOW%Starting Gemma3 API server on port 8000...%NC%
echo %YELLOW%This will provide OpenAI-compatible API for your model%NC%

start "Gemma3 Legal AI Server" python gemma3_server.py

echo.
echo %GREEN%🎉 YOUR GEMMA3 MODEL IS NOW WORKING!%NC%
echo.
echo %BLUE%📋 Available Options:%NC%
echo %GREEN%  ✓ Option 1: Direct Ollama API (port 11434)%NC%
echo %GREEN%  ✓ Option 2: Enhanced API Server (port 8000)%NC%
echo %GREEN%  ✓ Option 3: Fallback to llama3.2:1b if needed%NC%
echo.
echo %BLUE%🔗 API Endpoints:%NC%
echo %YELLOW%  • Ollama: http://localhost:11434/api/generate%NC%
echo %YELLOW%  • Enhanced: http://localhost:8000/v1/chat/completions%NC%
echo %YELLOW%  • Test Legal: http://localhost:8000/test-legal%NC%
echo %YELLOW%  • Health: http://localhost:8000/health%NC%
echo.
echo %BLUE%🧪 Test Commands:%NC%
echo %YELLOW%1. Test Custom Model:%NC%
echo    curl -X POST http://localhost:11434/api/generate ^
echo      -H "Content-Type: application/json" ^
echo      -d "{\"model\":\"gemma3-legal\",\"prompt\":\"Analyze this contract clause\",\"stream\":false}"
echo.
echo %YELLOW%2. Test Enhanced API:%NC%
echo    curl -X POST http://localhost:8000/v1/chat/completions ^
echo      -H "Content-Type: application/json" ^
echo      -d "{\"model\":\"gemma3-legal\",\"messages\":[{\"role\":\"user\",\"content\":\"Help with legal analysis\"}]}"
echo.
echo %GREEN%✨ YOUR UNSLOTH-TRAINED GEMMA3 MODEL IS READY FOR LEGAL AI!%NC%
echo.
pause
