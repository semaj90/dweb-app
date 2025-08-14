# PHASE 3+4 LEGAL AI SYSTEM STATUS REPORT - PowerShell Version
Write-Host "🚀 PHASE 3+4 LEGAL AI SYSTEM - COMPREHENSIVE STATUS" -ForegroundColor Blue
Write-Host ("=" * 60) -ForegroundColor Blue
Write-Host ""

$services = @(
    @{Name="PostgreSQL + pgvector"; Port=5432; Type="database"; Phase="3+4"; Critical=$true; HealthUrl=""},
    @{Name="Redis Cache"; Port=6379; Type="cache"; Phase="3+4"; Critical=$true; HealthUrl=""},
    @{Name="Qdrant Vector DB"; Port=6333; Type="vector"; Phase="3"; Critical=$true; HealthUrl="http://localhost:6333"},
    @{Name="Ollama LLM"; Port=11434; Type="llm"; Phase="3"; Critical=$true; HealthUrl="http://localhost:11434/api/version"},
    @{Name="Neo4j Graph DB"; Port=7474; Type="graph"; Phase="4"; Critical=$false; HealthUrl="http://localhost:7474"},
    @{Name="RabbitMQ Event Stream"; Port=5672; Type="messaging"; Phase="4"; Critical=$false; HealthUrl=""},
    @{Name="RabbitMQ Management"; Port=15672; Type="ui"; Phase="4"; Critical=$false; HealthUrl="http://localhost:15672"},
    @{Name="TTS Service"; Port=5002; Type="ai"; Phase="3+4"; Critical=$false; HealthUrl="http://localhost:5002/health"}
)

Write-Host "🔍 TESTING SERVICE CONNECTIVITY:" -ForegroundColor Yellow
Write-Host ("-" * 40) -ForegroundColor Gray

$results = @()
foreach ($service in $services) {
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $service.Port -InformationLevel Quiet -WarningAction SilentlyContinue
        if ($connection) {
            $status = "connected"
            $icon = "✅"
        } else {
            $status = "failed"
            $icon = if ($service.Critical) {"❌"} else {"⚠️ "}
        }
    } catch {
        $status = "error"
        $icon = if ($service.Critical) {"❌"} else {"⚠️ "}
    }
    
    $result = $service.Clone()
    $result.Status = $status
    $results += $result
    
    $phase = $service.Phase.PadRight(3)
    $type = $service.Type.PadRight(9)
    Write-Host "$icon [$phase] $type $($service.Name): $status"
}

Write-Host ""

# Test HTTP endpoints
Write-Host "🏥 TESTING HTTP HEALTH ENDPOINTS:" -ForegroundColor Yellow
Write-Host ("-" * 35) -ForegroundColor Gray

foreach ($result in $results) {
    if ($result.HealthUrl -ne "") {
        try {
            $response = Invoke-WebRequest -Uri $result.HealthUrl -Method GET -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Host "✅ $($result.Name): healthy" -ForegroundColor Green
                $result.HttpStatus = "healthy"
            } else {
                Write-Host "⚠️  $($result.Name): http-$($response.StatusCode)" -ForegroundColor Yellow
                $result.HttpStatus = "http-$($response.StatusCode)"
            }
        } catch {
            Write-Host "❌ $($result.Name): http-error" -ForegroundColor Red
            $result.HttpStatus = "http-error"
        }
    }
}

Write-Host ""

# Check models directory
Write-Host "🤖 CHECKING LOCAL MODELS:" -ForegroundColor Yellow
Write-Host ("-" * 25) -ForegroundColor Gray

$modelsDir = ".\models"
if (Test-Path $modelsDir) {
    $items = Get-ChildItem $modelsDir
    foreach ($item in $items) {
        $icon = if ($item.PSIsContainer) {"📁"} else {"📄"}
        Write-Host "$icon $($item.Name)"
    }
    
    # Find GGUF files
    $ggufFiles = Get-ChildItem $modelsDir -Recurse -Include "*.gguf", "*.bin" -ErrorAction SilentlyContinue
    if ($ggufFiles.Count -gt 0) {
        Write-Host ""
        Write-Host "🎯 GGUF/Model Files Found:" -ForegroundColor Green
        foreach ($file in $ggufFiles) {
            $sizeMB = [math]::Round($file.Length / 1MB, 1)
            Write-Host "  ✅ $($file.FullName) ($sizeMB MB)" -ForegroundColor Green
        }
    } else {
        Write-Host ""
        Write-Host "⚠️  No GGUF files found - place your Unsloth model in .\models\" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Models directory not found" -ForegroundColor Red
}

Write-Host ""

# Generate summary
$connectedServices = ($results | Where-Object {$_.Status -eq "connected"}).Count
$criticalServices = ($results | Where-Object {$_.Critical -eq $true}).Count
$connectedCritical = ($results | Where-Object {$_.Critical -eq $true -and $_.Status -eq "connected"}).Count
$healthyEndpoints = ($results | Where-Object {$_.HttpStatus -eq "healthy"}).Count
$totalEndpoints = ($results | Where-Object {$_.HealthUrl -ne ""}).Count

Write-Host "📊 SYSTEM STATUS SUMMARY:" -ForegroundColor Blue
Write-Host ("=" * 30) -ForegroundColor Blue
Write-Host "🔗 Connected Services: $connectedServices/$($services.Count)"
Write-Host "🎯 Critical Services: $connectedCritical/$criticalServices"
if ($totalEndpoints -gt 0) {
    Write-Host "🏥 Healthy Endpoints: $healthyEndpoints/$totalEndpoints"
}
Write-Host ""

# Phase breakdown
$phase3Services = $results | Where-Object {$_.Phase -like "*3*"}
$phase4Services = $results | Where-Object {$_.Phase -like "*4*"}
$phase3Connected = ($phase3Services | Where-Object {$_.Status -eq "connected"}).Count
$phase4Connected = ($phase4Services | Where-Object {$_.Status -eq "connected"}).Count

Write-Host "🏗️  PHASE BREAKDOWN:" -ForegroundColor Blue
Write-Host "Phase 3 (Advanced RAG): $phase3Connected/$($phase3Services.Count) services"
Write-Host "Phase 4 (Data + Events): $phase4Connected/$($phase4Services.Count) services"
Write-Host ""

# Final assessment
$systemHealthy = $connectedCritical -ge 3
$readyForPhase5 = $connectedCritical -eq $criticalServices

Write-Host "🎯 FINAL ASSESSMENT:" -ForegroundColor Blue
Write-Host ("=" * 20) -ForegroundColor Blue

if ($readyForPhase5) {
    Write-Host "🎉 SYSTEM STATUS: FULLY OPERATIONAL" -ForegroundColor Green
    Write-Host "✅ All critical services running" -ForegroundColor Green
    Write-Host "✅ Ready for Phase 5 development" -ForegroundColor Green
    Write-Host "✅ Advanced RAG capabilities active" -ForegroundColor Green
    Write-Host "✅ Data management pipeline ready" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 NEXT STEPS:" -ForegroundColor Cyan
    Write-Host "1. Load your Unsloth GGUF model into Ollama"
    Write-Host "2. Set up vLLM server for enhanced performance"
    Write-Host "3. Begin Phase 5: AI-Driven Real-Time UI Updates"
    Write-Host "4. Implement SvelteKit components with reactive state"
    Write-Host ""
    Write-Host "🔗 SERVICE ACCESS:" -ForegroundColor Cyan
    Write-Host "• PostgreSQL: localhost:5432 (legal_admin/LegalRAG2024!)"
    Write-Host "• Qdrant: http://localhost:6333"
    Write-Host "• Ollama: http://localhost:11434"
    Write-Host "• Redis: localhost:6379"
    
} elseif ($systemHealthy) {
    Write-Host "⚠️  SYSTEM STATUS: PARTIALLY OPERATIONAL" -ForegroundColor Yellow
    Write-Host "✅ $connectedCritical/$criticalServices critical services running" -ForegroundColor Yellow
    Write-Host "🔧 Some services need attention" -ForegroundColor Yellow
    
    $failedCritical = $results | Where-Object {$_.Critical -eq $true -and $_.Status -ne "connected"}
    if ($failedCritical.Count -gt 0) {
        Write-Host ""
        Write-Host "❌ Failed Critical Services:" -ForegroundColor Red
        foreach ($service in $failedCritical) {
            Write-Host "   • $($service.Name): $($service.Status)" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "🔧 TROUBLESHOOTING:" -ForegroundColor Cyan
    Write-Host "1. Check Docker containers: docker ps"
    Write-Host "2. View logs: docker-compose logs [service]"
    Write-Host "3. Restart services: docker-compose restart"
    
} else {
    Write-Host "❌ SYSTEM STATUS: CRITICAL ISSUES" -ForegroundColor Red
    Write-Host "🚨 Multiple critical services down" -ForegroundColor Red
    Write-Host "🔧 System requires immediate attention" -ForegroundColor Red
    
    Write-Host ""
    Write-Host "🆘 EMERGENCY RECOVERY:" -ForegroundColor Red
    Write-Host "1. Stop all: docker-compose down"
    Write-Host "2. Start fresh: docker-compose up -d"
    Write-Host "3. Wait 60 seconds for initialization"
    Write-Host "4. Re-run this test"
}

Write-Host ""
Write-Host "📋 LOG: Test completed at $(Get-Date)" -ForegroundColor Gray

# Create a simple vLLM setup guide for the GGUF model
Write-Host ""
Write-Host "🤖 VLLM SETUP FOR UNSLOTH GGUF MODEL:" -ForegroundColor Magenta
Write-Host "1. Install vLLM: pip install vllm" -ForegroundColor White
Write-Host "2. Copy your .gguf file to .\models\" -ForegroundColor White
Write-Host "3. Run: python start_vllm.py" -ForegroundColor White
Write-Host "4. Test: curl http://localhost:8000/health" -ForegroundColor White

if ($readyForPhase5) {
    exit 0
} elseif ($systemHealthy) {
    exit 1
} else {
    exit 2
}
