Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "   AI SYNTHESIS SYSTEM STATUS CHECK" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$services = @(
    @{Name="PostgreSQL"; Port=5432; Check={Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet}},
    @{Name="Redis"; Port=6379; Check={Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet}},
    @{Name="Ollama"; Port=11434; Check={Test-NetConnection -ComputerName localhost -Port 11434 -InformationLevel Quiet}},
    @{Name="Neo4j"; Port=7687; Check={Test-NetConnection -ComputerName localhost -Port 7687 -InformationLevel Quiet}},
    @{Name="Qdrant"; Port=6333; Check={Test-NetConnection -ComputerName localhost -Port 6333 -InformationLevel Quiet}},
    @{Name="SvelteKit"; Port=5173; Check={Test-NetConnection -ComputerName localhost -Port 5173 -InformationLevel Quiet}},
    @{Name="Enhanced RAG"; Port=8094; Check={Test-NetConnection -ComputerName localhost -Port 8094 -InformationLevel Quiet}},
    @{Name="GPU Orchestrator"; Port=8095; Check={Test-NetConnection -ComputerName localhost -Port 8095 -InformationLevel Quiet}},
    @{Name="Context7 MCP"; Port=4000; Check={Test-NetConnection -ComputerName localhost -Port 4000 -InformationLevel Quiet}},
    @{Name="AI Synthesis MCP"; Port=8200; Check={Test-NetConnection -ComputerName localhost -Port 8200 -InformationLevel Quiet}}
)

$running = 0
$total = $services.Count

Write-Host "Checking Services..." -ForegroundColor Yellow
Write-Host "--------------------" -ForegroundColor Gray

foreach ($service in $services) {
    Write-Host -NoNewline "$($service.Name) (Port $($service.Port)): "
    
    $isRunning = & $service.Check
    
    if ($isRunning) {
        Write-Host "✅ RUNNING" -ForegroundColor Green
        $running++
    } else {
        Write-Host "❌ OFFLINE" -ForegroundColor Red
    }
}

Write-Host "`n--------------------" -ForegroundColor Gray
Write-Host "Services Running: $running/$total" -ForegroundColor $(if ($running -eq $total) {"Green"} elseif ($running -gt $total/2) {"Yellow"} else {"Red"})

# Quick start commands for offline services
if ($running -lt $total) {
    Write-Host "`n📝 Quick Start Commands:" -ForegroundColor Cyan
    
    if (-not (& {Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet})) {
        Write-Host "  Redis: redis-server --port 6379" -ForegroundColor Gray
    }
    if (-not (& {Test-NetConnection -ComputerName localhost -Port 11434 -InformationLevel Quiet})) {
        Write-Host "  Ollama: ollama serve" -ForegroundColor Gray
    }
    if (-not (& {Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet})) {
        Write-Host "  PostgreSQL: Start-Service postgresql-x64-15" -ForegroundColor Gray
    }
    if (-not (& {Test-NetConnection -ComputerName localhost -Port 7687 -InformationLevel Quiet})) {
        Write-Host "  Neo4j: C:\neo4j-community-5.23.0\bin\neo4j.bat console" -ForegroundColor Gray
    }
    if (-not (& {Test-NetConnection -ComputerName localhost -Port 5173 -InformationLevel Quiet})) {
        Write-Host "  SvelteKit: npm run dev" -ForegroundColor Gray
    }
    
    Write-Host "`n  Or run everything: .\START-AI-SYNTHESIS-FULL-STACK.bat" -ForegroundColor Yellow
}

Write-Host "`n✅ Status check complete!" -ForegroundColor Green
