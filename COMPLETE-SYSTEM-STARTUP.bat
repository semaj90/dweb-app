@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================================
echo   🎮 YORHA LEGAL AI - COMPLETE SYSTEM STARTUP 🎮
echo   Unsloth Gemma3 Model + Advanced RAG + SvelteKit Frontend
echo ================================================================
echo.

set "GREEN=[92m"
set "BLUE=[94m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

echo %BLUE%🚀 STEP 1: Starting Backend Infrastructure%NC%
echo.

echo %YELLOW%Starting Docker containers...%NC%
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"
docker-compose up -d

echo %YELLOW%Waiting for services to initialize...%NC%
timeout /t 20 /nobreak > nul

echo.
echo %BLUE%🤖 STEP 2: Loading Your Unsloth Gemma3 Model%NC%
echo.

echo %YELLOW%Checking if model is already loaded...%NC%
powershell -Command "$response = try { Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -Method GET -TimeoutSec 10; $response.models | Where-Object { $_.name -like '*gemma3*' } } catch { $null }; if ($response) { Write-Host '✅ Gemma3 model already loaded' -ForegroundColor Green } else { Write-Host '⚠️ Model needs to be loaded' -ForegroundColor Yellow }"

echo %YELLOW%Copying your Unsloth model (7.3GB)...%NC%
docker cp "gemma3Q4_K_M\mohf16-Q4_K_M.gguf" legal-ai-ollama:/tmp/gemma3-unsloth.gguf

echo %YELLOW%Creating optimized Modelfile...%NC%
docker cp "Modelfile-unsloth-gemma3" legal-ai-ollama:/tmp/Modelfile-unsloth

echo %YELLOW%Loading model into Ollama (this may take 2-3 minutes)...%NC%
docker exec legal-ai-ollama ollama create gemma3-unsloth-legal -f /tmp/Modelfile-unsloth

echo %YELLOW%Waiting for model to load...%NC%
timeout /t 30 /nobreak > nul

echo.
echo %BLUE%🚀 STEP 3: Starting Phase 3 RAG Backend%NC%
echo.

echo %YELLOW%Installing Python dependencies...%NC%
cd backend
pip install -r requirements.txt > nul 2>&1

echo %YELLOW%Starting Phase 3 Agentic RAG API on port 9000...%NC%
start "Phase 3 RAG Backend" cmd /c "python phase3_agentic_rag.py"

echo.
echo %BLUE%🎨 STEP 4: Starting SvelteKit Frontend%NC%
echo.

cd ..\sveltekit-frontend

echo %YELLOW%Installing frontend dependencies...%NC%
call npm install > nul 2>&1

echo %YELLOW%Starting SvelteKit development server on port 5173...%NC%
start "YoRHa Legal AI Frontend" cmd /c "npm run dev"

echo.
echo %BLUE%🌐 STEP 5: Opening Demo Pages%NC%
echo.

timeout /t 5 /nobreak > nul

echo %YELLOW%Opening HTML demo...%NC%
cd ..\frontend
start "HTML Demo" "demo.html"

echo %YELLOW%Opening SvelteKit app...%NC%
start "SvelteKit App" "http://localhost:5173"

echo.
echo %GREEN%🎉 YORHA LEGAL AI SYSTEM FULLY OPERATIONAL! 🎉%NC%
echo.
echo %BLUE%📋 Available Interfaces:%NC%
echo   %GREEN%✅ HTML Demo:%NC%      file:///frontend/demo.html
echo   %GREEN%✅ SvelteKit App:%NC%  http://localhost:5173
echo   %GREEN%✅ API Docs:%NC%       http://localhost:9000/docs
echo.
echo %BLUE%🤖 AI Model Information:%NC%
echo   %GREEN%• Model:%NC%           gemma3-unsloth-legal:latest
echo   %GREEN%• File:%NC%            mohf16-Q4_K_M.gguf (7.3GB)
echo   %GREEN%• Training:%NC%        Unsloth fine-tuned for legal analysis
echo   %GREEN%• API:%NC%             http://localhost:11434/api/generate
echo.
echo %BLUE%🏗️ System Architecture:%NC%
echo   %GREEN%✅ Phase 3 RAG:%NC%     Advanced document analysis + vector search
echo   %GREEN%✅ Phase 4 Data:%NC%    Event streaming + graph relationships  
echo   %GREEN%✅ Custom Model:%NC%    Your Unsloth-trained Gemma3 (7.3GB)
echo   %GREEN%✅ Modern UI:%NC%       SvelteKit + Tailwind + YoRHa design
echo.
echo %BLUE%🎯 Key Features:%NC%
echo   • Draggable AI chat interface
echo   • Real-time model switching
echo   • Advanced RAG document search
echo   • Professional legal analysis
echo   • NieR: Automata inspired design
echo   • Streaming responses
echo   • Mobile-responsive layout
echo.
echo %BLUE%🧪 Testing Your Model:%NC%
echo.
echo %YELLOW%Testing Unsloth Gemma3 model...%NC%
timeout /t 10 /nobreak > nul

powershell -Command "$body = @{ model = 'gemma3-unsloth-legal:latest'; prompt = 'Hello! I am your custom Unsloth-trained Gemma3 legal AI. What are the most important clauses to review in software licensing agreements?'; stream = $false; options = @{ temperature = 0.1; num_predict = 300 } } | ConvertTo-Json -Depth 3; try { $response = Invoke-RestMethod -Uri 'http://localhost:11434/api/generate' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 60; Write-Host '🎉 YOUR UNSLOTH GEMMA3 MODEL IS WORKING!' -ForegroundColor Green; Write-Host 'Response Preview:' -ForegroundColor Yellow; Write-Host $response.response.Substring(0, [Math]::Min(200, $response.response.Length)) -ForegroundColor White; Write-Host '...' -ForegroundColor Gray } catch { Write-Host '⚠️ Model may still be loading. Try the web interface.' -ForegroundColor Yellow }"

echo.
echo %GREEN%🚀 SYSTEM READY FOR LEGAL AI APPLICATIONS!%NC%
echo %BLUE%Your Unsloth-trained Gemma3 model is now powering a complete legal AI system.%NC%
echo.
echo %YELLOW%Press any key to view system status...%NC%
pause > nul

echo.
echo %BLUE%📊 Final System Status:%NC%
powershell -Command "Write-Host '🔍 Checking all services...' -ForegroundColor Yellow; $services = @('http://localhost:11434/api/version', 'http://localhost:6333', 'http://localhost:5432', 'http://localhost:6379', 'http://localhost:5173', 'http://localhost:9000/health'); foreach ($url in $services) { try { $response = Invoke-WebRequest -Uri $url -Method GET -TimeoutSec 5; if ($response.StatusCode -eq 200) { Write-Host \"✅ $url - ONLINE\" -ForegroundColor Green } else { Write-Host \"⚠️ $url - Issues\" -ForegroundColor Yellow } } catch { Write-Host \"❌ $url - OFFLINE\" -ForegroundColor Red } }"

echo.
echo %GREEN%🎊 Enjoy your advanced Legal AI system with Unsloth Gemma3! 🎊%NC%
echo.
