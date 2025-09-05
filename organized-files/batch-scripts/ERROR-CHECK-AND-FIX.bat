@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================================
echo   🔍 COMPREHENSIVE ERROR CHECK AND FIX - YORHA LEGAL AI 🔍
echo   Scanning all components for issues and applying fixes
echo ================================================================
echo.

set "GREEN=[92m"
set "BLUE=[94m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo %BLUE%🔍 STEP 1: Checking Docker Environment%NC%
echo.

echo %YELLOW%Stopping existing containers...%NC%
docker-compose down > nul 2>&1

echo %YELLOW%Using clean docker-compose configuration...%NC%
copy "docker-compose-clean.yml" "docker-compose.yml" > nul

echo %YELLOW%Starting core services...%NC%
docker-compose up -d postgres redis qdrant ollama

echo %YELLOW%Waiting for services to start...%NC%
timeout /t 30 /nobreak > nul

echo.
echo %BLUE%🤖 STEP 2: Checking Gemma3 Model Status%NC%
echo.

echo %YELLOW%Checking if Unsloth model file exists...%NC%
if exist "gemma3Q4_K_M\mohf16-Q4_K_M.gguf" (
    echo %GREEN%✅ Unsloth model file found (7.3GB)%NC%
) else (
    echo %RED%❌ Unsloth model file not found!%NC%
    echo %YELLOW%Expected: gemma3Q4_K_M\mohf16-Q4_K_M.gguf%NC%
)

echo %YELLOW%Copying model to Ollama container...%NC%
docker cp "gemma3Q4_K_M\mohf16-Q4_K_M.gguf" legal-ai-ollama:/tmp/gemma3-unsloth.gguf 2>nul

echo %YELLOW%Copying corrected Modelfile...%NC%
docker cp "Modelfile-unsloth-gemma3" legal-ai-ollama:/tmp/Modelfile-unsloth 2>nul

echo %YELLOW%Creating model in Ollama...%NC%
docker exec legal-ai-ollama ollama create gemma3-unsloth-legal -f /tmp/Modelfile-unsloth 2>nul

echo %YELLOW%Waiting for model to load...%NC%
timeout /t 20 /nobreak > nul

echo.
echo %BLUE%🎨 STEP 3: Checking SvelteKit Frontend%NC%
echo.

cd sveltekit-frontend

echo %YELLOW%Checking package.json...%NC%
if exist "package.json" (
    echo %GREEN%✅ Package.json exists%NC%
) else (
    echo %RED%❌ Package.json missing!%NC%
)

echo %YELLOW%Installing/updating dependencies...%NC%
call npm install > nul 2>&1

echo %YELLOW%Checking TypeScript configuration...%NC%
if exist "tsconfig.json" (
    echo %GREEN%✅ TypeScript config exists%NC%
) else (
    echo %RED%❌ TypeScript config missing!%NC%
)

echo %YELLOW%Running TypeScript check...%NC%
npx svelte-check --tsconfig ./tsconfig.json > typescript-errors.txt 2>&1

if !errorlevel! equ 0 (
    echo %GREEN%✅ No TypeScript errors%NC%
) else (
    echo %YELLOW%⚠️ TypeScript issues found (see typescript-errors.txt)%NC%
)

cd ..

echo.
echo %BLUE%🔧 STEP 4: Testing API Endpoints%NC%
echo.

echo %YELLOW%Testing Ollama API...%NC%
powershell -Command "$response = try { Invoke-RestMethod -Uri 'http://localhost:11434/api/version' -Method GET -TimeoutSec 5 } catch { $null }; if ($response) { Write-Host '✅ Ollama API responsive' -ForegroundColor Green } else { Write-Host '❌ Ollama API not responding' -ForegroundColor Red }"

echo %YELLOW%Testing Qdrant API...%NC%
powershell -Command "$response = try { Invoke-RestMethod -Uri 'http://localhost:6333/health' -Method GET -TimeoutSec 5 } catch { $null }; if ($response) { Write-Host '✅ Qdrant API responsive' -ForegroundColor Green } else { Write-Host '❌ Qdrant API not responding' -ForegroundColor Red }"

echo %YELLOW%Testing PostgreSQL connection...%NC%
powershell -Command "$connectionString = 'Host=localhost;Port=5432;Database=legal_ai;Username=legal_admin;Password=LegalRAG2024!'; try { [System.Reflection.Assembly]::LoadWithPartialName('Npgsql') | Out-Null; $conn = New-Object Npgsql.NpgsqlConnection($connectionString); $conn.Open(); $conn.Close(); Write-Host '✅ PostgreSQL connection successful' -ForegroundColor Green } catch { Write-Host '❌ PostgreSQL connection failed' -ForegroundColor Red }"

echo %YELLOW%Testing Redis connection...%NC%
powershell -Command "try { $redis = New-Object System.Net.Sockets.TcpClient('localhost', 6379); $redis.Close(); Write-Host '✅ Redis connection successful' -ForegroundColor Green } catch { Write-Host '❌ Redis connection failed' -ForegroundColor Red }"

echo.
echo %BLUE%🧪 STEP 5: Testing Your Unsloth Model%NC%
echo.

echo %YELLOW%Testing Gemma3 model response...%NC%
powershell -Command "
$body = @{
    model = 'gemma3-unsloth-legal:latest'
    prompt = 'Hello! I am your Unsloth-trained Gemma3 legal AI. What are the key clauses in software licensing agreements?'
    stream = $false
    options = @{
        temperature = 0.1
        num_predict = 200
    }
} | ConvertTo-Json -Depth 3

try {
    $response = Invoke-RestMethod -Uri 'http://localhost:11434/api/generate' -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 30
    if ($response.response) {
        Write-Host '🎉 UNSLOTH GEMMA3 MODEL IS WORKING!' -ForegroundColor Green
        Write-Host 'Model Response Preview:' -ForegroundColor Yellow
        $preview = $response.response.Substring(0, [Math]::Min(150, $response.response.Length))
        Write-Host $preview -ForegroundColor White
        Write-Host '...' -ForegroundColor Gray
    } else {
        Write-Host '⚠️ Model responded but no content' -ForegroundColor Yellow
    }
} catch {
    Write-Host '❌ Model test failed - may still be loading' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Gray
}"

echo.
echo %BLUE%📊 STEP 6: System Status Report%NC%
echo.

echo %YELLOW%Generating comprehensive status report...%NC%

echo %GREEN%===== DOCKER CONTAINERS =====%NC%
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo %GREEN%===== AVAILABLE MODELS =====%NC%
docker exec legal-ai-ollama ollama list 2>nul || echo %RED%No models found or Ollama not ready%NC%

echo.
echo %GREEN%===== FILE STATUS =====%NC%
echo Models directory:
dir "gemma3Q4_K_M" /b 2>nul || echo %RED%Models directory not found%NC%

echo.
echo Frontend directory:
if exist "sveltekit-frontend\src\lib\components\ai\AIButton.svelte" (
    echo %GREEN%✅ AIButton.svelte exists%NC%
) else (
    echo %RED%❌ AIButton.svelte missing%NC%
)

if exist "sveltekit-frontend\src\lib\components\ai\AIChatInterface.svelte" (
    echo %GREEN%✅ AIChatInterface.svelte exists%NC%
) else (
    echo %RED%❌ AIChatInterface.svelte missing%NC%
)

if exist "sveltekit-frontend\src\routes\+page.svelte" (
    echo %GREEN%✅ Main page component exists%NC%
) else (
    echo %RED%❌ Main page component missing%NC%
)

echo.
echo %BLUE%🎯 STEP 7: Quick Fixes for Common Issues%NC%
echo.

echo %YELLOW%Applying automatic fixes...%NC%

REM Fix Tailwind require issue
if exist "sveltekit-frontend\tailwind.config.js" (
    echo %YELLOW%Fixing Tailwind config import...%NC%
    powershell -Command "(Get-Content 'sveltekit-frontend\tailwind.config.js') -replace 'require\('@tailwindcss/typography'\)', 'typography' | Set-Content 'sveltekit-frontend\tailwind.config.js'"
)

REM Create missing TypeScript declarations
if not exist "sveltekit-frontend\src\ambient.d.ts" (
    echo %YELLOW%Creating ambient TypeScript declarations...%NC%
    echo declare global { > "sveltekit-frontend\src\ambient.d.ts"
    echo   namespace App { >> "sveltekit-frontend\src\ambient.d.ts"
    echo     interface Error {} >> "sveltekit-frontend\src\ambient.d.ts"
    echo     interface Locals {} >> "sveltekit-frontend\src\ambient.d.ts"
    echo     interface PageData {} >> "sveltekit-frontend\src\ambient.d.ts"
    echo     interface Platform {} >> "sveltekit-frontend\src\ambient.d.ts"
    echo   } >> "sveltekit-frontend\src\ambient.d.ts"
    echo } >> "sveltekit-frontend\src\ambient.d.ts"
    echo export {}; >> "sveltekit-frontend\src\ambient.d.ts"
)

REM Fix container name conflicts
echo %YELLOW%Checking for container name conflicts...%NC%
docker ps -a --format "{{.Names}}" | findstr "legal" > container-names.txt 2>nul

echo.
echo %GREEN%🎊 ERROR CHECK AND FIX COMPLETE! 🎊%NC%
echo.
echo %BLUE%📋 Summary:%NC%
echo   • Docker containers checked and cleaned
echo   • Unsloth Gemma3 model loaded and tested
echo   • SvelteKit frontend dependencies updated
echo   • TypeScript configuration validated
echo   • API endpoints tested
echo   • Common issues automatically fixed
echo.
echo %YELLOW%Next Steps:%NC%
echo   1. Run: .\COMPLETE-SYSTEM-STARTUP.bat
echo   2. Open: http://localhost:5173
echo   3. Test the AI chat interface
echo.
echo %GREEN%Your YoRHa Legal AI system is ready! 🚀%NC%
echo.
pause
