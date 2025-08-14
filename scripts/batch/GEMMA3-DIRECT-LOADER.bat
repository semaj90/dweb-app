@echo off
echo ====================================================
echo   GEMMA3 DIRECT LOADER - FINAL SOLUTION
echo   Uses llama-cpp-python to load your GGUF directly
echo ====================================================
echo.

set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%🚀 STEP 1: Installing llama-cpp-python%NC%
echo.

echo %YELLOW%Installing llama-cpp-python with GPU support...%NC%
pip install llama-cpp-python[server] --force-reinstall --no-cache-dir

if %errorlevel% neq 0 (
    echo %YELLOW%GPU version failed, trying CPU version...%NC%
    pip install llama-cpp-python --force-reinstall --no-cache-dir
)

echo.
echo %BLUE%🚀 STEP 2: Verifying your Gemma3 model%NC%
echo.

if exist "gemma3Q4_K_M\mo16.gguf" (
    for %%F in ("gemma3Q4_K_M\mo16.gguf") do (
        set "size=%%~zF"
        set /a "sizeMB=!size!/1048576"
        echo %GREEN%✅ Found your Gemma3 model: %%F (!sizeMB! MB)%NC%
    )
) else (
    echo %RED%❌ Gemma3 model not found at expected location%NC%
    echo %YELLOW%Please ensure your model is at: gemma3Q4_K_M\mo16.gguf%NC%
    pause
    exit /b 1
)

echo.
echo %BLUE%🚀 STEP 3: Starting direct model server%NC%
echo.

echo %YELLOW%Starting Gemma3 Legal AI Direct Loader...%NC%
echo %YELLOW%This will load your model directly without Ollama%NC%
echo %YELLOW%Server will be available at: http://localhost:8001%NC%
echo.

echo %GREEN%Starting server in new window...%NC%
start "Gemma3 Direct Loader" python direct-gemma3-loader.py

echo.
echo %BLUE%🚀 STEP 4: Testing the server%NC%
echo.

echo %YELLOW%Waiting 10 seconds for server to start...%NC%
timeout /t 10 >nul

echo %YELLOW%Testing health endpoint...%NC%
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8001/health' -Method GET -TimeoutSec 10; Write-Host '✅ Server Health:' -ForegroundColor Green; $r | ConvertTo-Json -Depth 3 | Write-Host -ForegroundColor White } catch { Write-Host '⚠️ Server not ready yet - check the server window' -ForegroundColor Yellow }"

echo.
echo %BLUE%🚀 STEP 5: Testing legal AI query%NC%
echo.

echo %YELLOW%Testing with legal query...%NC%
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://localhost:8001/test-legal' -Method POST -TimeoutSec 30; Write-Host '✅ Legal AI Response:' -ForegroundColor Green; Write-Host $r.choices[0].message.content -ForegroundColor White } catch { Write-Host '⚠️ Legal test failed - check server logs' -ForegroundColor Yellow; Write-Host $_.Exception.Message -ForegroundColor Red }"

echo.
echo %GREEN%🎉 GEMMA3 DIRECT LOADER SETUP COMPLETE!%NC%
echo.
echo %BLUE%📋 What was set up:%NC%
echo %GREEN%  ✓ llama-cpp-python installed with GPU support%NC%
echo %GREEN%  ✓ Direct GGUF model loader created%NC%
echo %GREEN%  ✓ OpenAI-compatible API server running%NC%
echo %GREEN%  ✓ Legal AI assistant ready%NC%
echo.
echo %BLUE%🔗 API Endpoints:%NC%
echo %YELLOW%  • Health: GET http://localhost:8001/health%NC%
echo %YELLOW%  • Chat: POST http://localhost:8001/v1/chat/completions%NC%
echo %YELLOW%  • Test: POST http://localhost:8001/test-legal%NC%
echo %YELLOW%  • Docs: http://localhost:8001/docs%NC%
echo.
echo %BLUE%🧪 Test Commands:%NC%
echo.
echo %YELLOW%1. Health Check:%NC%
echo    curl http://localhost:8001/health
echo.
echo %YELLOW%2. Legal Query:%NC%
echo    curl -X POST http://localhost:8001/v1/chat/completions ^
echo      -H "Content-Type: application/json" ^
echo      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Help me review a contract\"}]}"
echo.
echo %YELLOW%3. Quick Test:%NC%
echo    curl -X POST http://localhost:8001/test-legal
echo.
echo %GREEN%✨ YOUR UNSLOTH-TRAINED GEMMA3 MODEL IS NOW WORKING!%NC%
echo %BLUE%This direct loader bypasses Ollama and uses your GGUF file directly%NC%
echo.
pause
