# Quick Native Windows Status Dashboard
# Shows a visual overview of your Legal AI Platform status

Clear-Host

$banner = @"
╔═══════════════════════════════════════════════════════════════════════════╗
║                    YORHA LEGAL AI - SYSTEM STATUS                          ║
║                         Native Windows Edition                             ║
╚═══════════════════════════════════════════════════════════════════════════╝
"@

Write-Host $banner -ForegroundColor Cyan

# Quick port check function
function Get-PortStatus {
    param($Port, $ServiceName)
    
    $test = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
    
    if ($test.TcpTestSucceeded) {
        return @{
            Status = "ONLINE"
            Color = "Green"
            Symbol = "✓"
        }
    } else {
        return @{
            Status = "OFFLINE"
            Color = "Red"
            Symbol = "✗"
        }
    }
}

# Create status board
$services = @(
    @{Name="PostgreSQL Database"; Port=5432; Required=$true},
    @{Name="Redis Cache"; Port=6379; Required=$true},
    @{Name="Neo4j Graph DB"; Port=7474; Required=$false},
    @{Name="MinIO Storage"; Port=9000; Required=$false},
    @{Name="Ollama AI"; Port=11434; Required=$true},
    @{Name="Dev Server"; Port=5173; Required=$true},
    @{Name="GPU Orchestrator"; Port=8084; Required=$false},
    @{Name="RAG Service"; Port=8085; Required=$false}
)

Write-Host "`n┌─────────────────────────┬──────────┬────────┐" -ForegroundColor White
Write-Host "│ SERVICE                 │ PORT     │ STATUS │" -ForegroundColor White
Write-Host "├─────────────────────────┼──────────┼────────┤" -ForegroundColor White

$onlineCount = 0
$requiredMissing = @()

foreach ($service in $services) {
    $status = Get-PortStatus -Port $service.Port -ServiceName $service.Name
    
    $nameDisplay = $service.Name.PadRight(23)
    $portDisplay = $service.Port.ToString().PadRight(8)
    
    if ($status.Status -eq "ONLINE") {
        $onlineCount++
        $statusDisplay = "$($status.Symbol) ONLINE"
        Write-Host "│ $nameDisplay │ $portDisplay │ " -NoNewline
        Write-Host $statusDisplay -ForegroundColor Green -NoNewline
        Write-Host " │"
    } else {
        $statusDisplay = "$($status.Symbol) OFFLINE"
        if ($service.Required) {
            $requiredMissing += $service.Name
        }
        Write-Host "│ $nameDisplay │ $portDisplay │ " -NoNewline
        Write-Host $statusDisplay -ForegroundColor Red -NoNewline
        Write-Host "│"
    }
}

Write-Host "└─────────────────────────┴──────────┴────────┘" -ForegroundColor White

# Database check
Write-Host "`n📊 DATABASE STATUS:" -ForegroundColor Yellow
try {
    $env:PGPASSWORD = "123456"
    $dbCheck = psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT 'Connected' as status;" 2>$null
    if ($dbCheck -match "Connected") {
        # Get counts
        $cases = psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM cases;" 2>$null
        $docs = psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM legal_documents;" 2>$null
        $evidence = psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM evidence;" 2>$null
        
        Write-Host "   ✓ Database Connected" -ForegroundColor Green
        Write-Host "   • Cases: $($cases.Trim())" -ForegroundColor Gray
        Write-Host "   • Documents: $($docs.Trim())" -ForegroundColor Gray
        Write-Host "   • Evidence: $($evidence.Trim())" -ForegroundColor Gray
    } else {
        Write-Host "   ✗ Database connection failed" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ PostgreSQL not accessible" -ForegroundColor Red
}

# AI Models check
Write-Host "`n🤖 AI MODELS:" -ForegroundColor Yellow
try {
    $models = ollama list 2>$null | Select-Object -Skip 1
    $modelList = @()
    foreach ($model in $models) {
        if ($model -and $model.Trim()) {
            $modelName = ($model -split '\s+')[0]
            if ($modelName -and $modelName -ne "NAME") {
                $modelList += $modelName
            }
        }
    }
    
    if ($modelList.Count -gt 0) {
        Write-Host "   ✓ Models available:" -ForegroundColor Green
        foreach ($model in $modelList) {
            Write-Host "   • $model" -ForegroundColor Gray
        }
    } else {
        Write-Host "   ⚠ No models loaded" -ForegroundColor Yellow
        Write-Host "   Run: ollama pull nomic-embed-text" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ✗ Ollama not available" -ForegroundColor Red
}

# Project Status
Write-Host "`n📁 PROJECT STATUS:" -ForegroundColor Yellow
$projectChecks = @(
    @{File="package.json"; Desc="Project config"},
    @{File="node_modules"; Desc="Dependencies"},
    @{File=".env"; Desc="Environment"},
    @{File="src\routes\yorha-dashboard"; Desc="YoRHa UI"}
)

foreach ($check in $projectChecks) {
    if (Test-Path $check.File) {
        Write-Host "   ✓ $($check.Desc)" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $($check.Desc) missing" -ForegroundColor Red
    }
}

# Overall Status
Write-Host "`n" -NoNewline
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan

$totalServices = $services.Count
$percentOnline = [math]::Round(($onlineCount / $totalServices) * 100)

if ($onlineCount -eq $totalServices) {
    Write-Host "🎉 SYSTEM FULLY OPERATIONAL!" -ForegroundColor Green
    Write-Host "All services are running ($onlineCount/$totalServices)" -ForegroundColor Green
    Write-Host "`nAccess YoRHa Dashboard at:" -ForegroundColor Cyan
    Write-Host "http://localhost:5173" -ForegroundColor White
} elseif ($requiredMissing.Count -eq 0) {
    Write-Host "✅ SYSTEM OPERATIONAL" -ForegroundColor Green
    Write-Host "Core services running ($onlineCount/$totalServices - $percentOnline%)" -ForegroundColor Yellow
    Write-Host "`nOptional services offline but system is functional" -ForegroundColor Gray
} else {
    Write-Host "⚠️  SYSTEM NEEDS ATTENTION" -ForegroundColor Yellow
    Write-Host "Only $onlineCount/$totalServices services running ($percentOnline%)" -ForegroundColor Red
    
    if ($requiredMissing.Count -gt 0) {
        Write-Host "`n❌ REQUIRED SERVICES MISSING:" -ForegroundColor Red
        foreach ($missing in $requiredMissing) {
            Write-Host "   • $missing" -ForegroundColor Red
        }
    }
    
    Write-Host "`n🔧 TO FIX:" -ForegroundColor Cyan
    Write-Host "Run as Administrator:" -ForegroundColor Yellow
    Write-Host ".\START-NATIVE-WINDOWS-COMPLETE.ps1" -ForegroundColor White
}

Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Quick action menu
Write-Host "QUICK ACTIONS:" -ForegroundColor Cyan
Write-Host "[1] Start all services" -ForegroundColor White
Write-Host "[2] Check detailed status" -ForegroundColor White
Write-Host "[3] View logs" -ForegroundColor White
Write-Host "[4] Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Select action (1-4)"

switch ($choice) {
    "1" {
        Write-Host "`nStarting services..." -ForegroundColor Yellow
        & ".\START-NATIVE-WINDOWS-COMPLETE.ps1"
    }
    "2" {
        Write-Host "`nRunning detailed check..." -ForegroundColor Yellow
        & ".\Check-Native-Installations.ps1"
    }
    "3" {
        Write-Host "`nOpening logs folder..." -ForegroundColor Yellow
        if (Test-Path ".\logs") {
            explorer ".\logs"
        } else {
            Write-Host "No logs folder found" -ForegroundColor Red
        }
    }
    "4" {
        Write-Host "Exiting..." -ForegroundColor Gray
    }
    default {
        Write-Host "Invalid choice" -ForegroundColor Red
    }
}
