// Complete Production Validation System
// File: manual-validation.ps1

# Force GPU acceleration
$env:OLLAMA_HOST = "0.0.0.0:11434"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_GPU_COMPUTE_CAPABILITY = "7.5"

Write-Host "🚀 PRODUCTION VALIDATION - MANUAL EXECUTION" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Yellow

# Create logs directory
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" }

# Test 1: Ollama Service
Write-Host "`n📡 Testing Ollama GPU service..." -ForegroundColor Cyan
try {
    $ollamaResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5
    Write-Host "✅ Ollama: OPERATIONAL" -ForegroundColor Green
    $ollamaStatus = "OPERATIONAL"
} catch {
    Write-Host "❌ Ollama: OFFLINE" -ForegroundColor Red
    $ollamaStatus = "OFFLINE"
}

# Test 2: Model Import
Write-Host "`n🤖 Testing Gemma3 Legal model..." -ForegroundColor Cyan
try {
    $modelBody = @{
        model = "gemma3-legal"
        prompt = "Legal AI test"
        stream = $false
        options = @{
            num_gpu = 1
            temperature = 0.1
        }
    } | ConvertTo-Json

    $modelResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method POST -Body $modelBody -ContentType "application/json" -TimeoutSec 10
    Write-Host "✅ Gemma3 Legal: OPERATIONAL" -ForegroundColor Green
    $modelStatus = "OPERATIONAL"
} catch {
    Write-Host "❌ Gemma3 Legal: OFFLINE" -ForegroundColor Red
    $modelStatus = "OFFLINE"
}

# Test 3: SvelteKit Dev Server
Write-Host "`n🌐 Testing SvelteKit dev server..." -ForegroundColor Cyan
try {
    $svelteResponse = Invoke-WebRequest -Uri "http://localhost:5173" -Method GET -TimeoutSec 5
    Write-Host "✅ SvelteKit: RUNNING" -ForegroundColor Green
    $svelteStatus = "RUNNING"
} catch {
    Write-Host "❌ SvelteKit: OFFLINE" -ForegroundColor Red
    $svelteStatus = "OFFLINE"
}

# Test 4: Evidence Synthesis API
Write-Host "`n🔬 Testing Synthesis API..." -ForegroundColor Cyan
try {
    $synthesisBody = @{
        evidenceIds = @("test-1", "test-2")
        synthesisType = "correlation"
        caseId = "validation-test"
        title = "Manual Validation Test"
    } | ConvertTo-Json

    $synthesisResponse = Invoke-RestMethod -Uri "http://localhost:5173/api/evidence/synthesize" -Method POST -Body $synthesisBody -ContentType "application/json" -TimeoutSec 10
    Write-Host "✅ Synthesis API: OPERATIONAL" -ForegroundColor Green
    $synthesisStatus = "OPERATIONAL"
} catch {
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Host "✅ Synthesis API: ACCESSIBLE (Auth Required)" -ForegroundColor Yellow
        $synthesisStatus = "ACCESSIBLE"
    } else {
        Write-Host "❌ Synthesis API: OFFLINE" -ForegroundColor Red
        $synthesisStatus = "OFFLINE"
    }
}

# Generate validation report
$report = @{
    timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    services = @{
        ollama = $ollamaStatus
        gemma3_legal = $modelStatus
        sveltekit = $svelteStatus
        synthesis_api = $synthesisStatus
    }
    recommendations = @(
        "Start Ollama: ollama serve",
        "Import model: ollama create gemma3-legal -f local-models/Modelfile.gemma3-legal",
        "Start SvelteKit: npm run dev",
        "Test endpoints manually via browser"
    )
    manual_tests = @{
        ai_assistant = "http://localhost:5173/ai-assistant"
        ai_summary = "http://localhost:5173/ai-summary"
        synthesis_api = "POST http://localhost:5173/api/evidence/synthesize"
    }
} | ConvertTo-Json -Depth 3

$report | Out-File -FilePath "logs/manual-validation-report.json"

Write-Host "`n📊 VALIDATION COMPLETE" -ForegroundColor Green
Write-Host "Ollama: $ollamaStatus" -ForegroundColor $(if($ollamaStatus -eq "OPERATIONAL"){"Green"}else{"Red"})
Write-Host "Gemma3: $modelStatus" -ForegroundColor $(if($modelStatus -eq "OPERATIONAL"){"Green"}else{"Red"})
Write-Host "SvelteKit: $svelteStatus" -ForegroundColor $(if($svelteStatus -eq "RUNNING"){"Green"}else{"Red"})
Write-Host "Synthesis: $synthesisStatus" -ForegroundColor $(if($synthesisStatus -like "*OPERATIONAL*" -or $synthesisStatus -eq "ACCESSIBLE"){"Green"}else{"Red"})

Write-Host "`n🎯 NEXT ACTIONS:" -ForegroundColor Yellow
if ($ollamaStatus -eq "OFFLINE") { Write-Host "1. Start Ollama: ollama serve" -ForegroundColor White }
if ($modelStatus -eq "OFFLINE") { Write-Host "2. Import model: ollama create gemma3-legal -f local-models/Modelfile.gemma3-legal" -ForegroundColor White }
if ($svelteStatus -eq "OFFLINE") { Write-Host "3. Start dev server: npm run dev" -ForegroundColor White }
Write-Host "4. Test UI: http://localhost:5173/ai-assistant" -ForegroundColor White

Write-Host "`nReport saved: logs/manual-validation-report.json" -ForegroundColor Cyan
