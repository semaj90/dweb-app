# Native Windows Installation Status Checker
# Run this to see what's installed and what's running

Write-Host @"
╔══════════════════════════════════════════════════════════════╗
║      NATIVE WINDOWS INSTALLATION STATUS CHECK                ║
║           Legal AI Platform - YoRHa Interface                ║
╚══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Initialize results
$global:results = @{
    Working = @()
    NotWorking = @()
    NeedsAttention = @()
}

function Write-Section {
    param($Title)
    Write-Host "`n━━━ $Title ━━━" -ForegroundColor Yellow
}

function Test-Installation {
    param(
        [string]$Name,
        [string]$Type,
        [string]$CheckCommand,
        [string]$Path,
        [int]$Port,
        [string]$ServiceName,
        [string]$ProcessName
    )
    
    $installed = $false
    $running = $false
    
    switch ($Type) {
        "Command" {
            try {
                $output = & cmd /c "$CheckCommand 2>nul"
                if ($LASTEXITCODE -eq 0 -or $output) {
                    Write-Host "✓ $Name is installed" -ForegroundColor Green
                    if ($output) {
                        $version = ($output | Select-Object -First 1).Trim()
                        Write-Host "  Version: $version" -ForegroundColor Gray
                    }
                    $installed = $true
                }
            } catch {
                Write-Host "✗ $Name is not installed" -ForegroundColor Red
            }
        }
        
        "Path" {
            if (Test-Path $Path) {
                Write-Host "✓ $Name found at: $Path" -ForegroundColor Green
                $installed = $true
            } else {
                Write-Host "✗ $Name not found at: $Path" -ForegroundColor Red
            }
        }
        
        "Port" {
            $connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
            if ($connection.TcpTestSucceeded) {
                Write-Host "✓ Port $Port is in use ($Name likely running)" -ForegroundColor Green
                $running = $true
            } else {
                Write-Host "⚠ Port $Port is available ($Name not running)" -ForegroundColor Yellow
            }
        }
        
        "Service" {
            $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
            if ($service) {
                if ($service.Status -eq 'Running') {
                    Write-Host "✓ $Name service is running" -ForegroundColor Green
                    $running = $true
                } else {
                    Write-Host "⚠ $Name service exists but is stopped" -ForegroundColor Yellow
                    $global:results.NeedsAttention += $Name
                }
                $installed = $true
            } else {
                Write-Host "✗ $Name service not found" -ForegroundColor Red
            }
        }
        
        "Process" {
            $process = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "✓ $Name is running (PID: $($process.Id))" -ForegroundColor Green
                $running = $true
            } else {
                Write-Host "⚠ $Name is not running" -ForegroundColor Yellow
            }
        }
    }
    
    # Update global results
    if ($running) {
        $global:results.Working += $Name
    } elseif ($installed -and -not $running) {
        if ($Name -notin $global:results.NeedsAttention) {
            $global:results.NeedsAttention += $Name
        }
    } elseif (-not $installed -and $Type -ne "Port") {
        $global:results.NotWorking += $Name
    }
    
    return @{
        Installed = $installed
        Running = $running
    }
}

# ============================================
# CORE DEVELOPMENT TOOLS
# ============================================
Write-Section "CORE DEVELOPMENT TOOLS"

Test-Installation -Name "Node.js" -Type "Command" -CheckCommand "node --version"
Test-Installation -Name "npm" -Type "Command" -CheckCommand "npm --version"
Test-Installation -Name "Git" -Type "Command" -CheckCommand "git --version"
Test-Installation -Name "Python" -Type "Command" -CheckCommand "python --version"

# ============================================
# DATABASES
# ============================================
Write-Section "DATABASE SERVICES"

# PostgreSQL
Write-Host "`nPostgreSQL:" -ForegroundColor Cyan
Test-Installation -Name "PostgreSQL CLI" -Type "Command" -CheckCommand "psql --version"
Test-Installation -Name "PostgreSQL 17" -Type "Service" -ServiceName "postgresql-x64-17"
if ($LASTEXITCODE -ne 0) {
    Test-Installation -Name "PostgreSQL 15" -Type "Service" -ServiceName "postgresql-x64-15"
}
Test-Installation -Name "PostgreSQL" -Type "Port" -Port 5432

# Test actual database connection
Write-Host "  Testing database connection..." -ForegroundColor Gray
try {
    $testQuery = "psql -U legal_admin -d legal_ai_db -h localhost -c 'SELECT COUNT(*) FROM cases;' 2>&1"
    $env:PGPASSWORD = "123456"
    $result = Invoke-Expression $testQuery
    if ($result -match "count") {
        Write-Host "  ✓ Database connection successful!" -ForegroundColor Green
        # Extract count if possible
        if ($result -match "(\d+)") {
            Write-Host "  ✓ Cases in database: $($matches[1])" -ForegroundColor Green
        }
    }
} catch {
    Write-Host "  ✗ Database connection failed" -ForegroundColor Red
}

# Redis
Write-Host "`nRedis:" -ForegroundColor Cyan
Test-Installation -Name "Redis" -Type "Service" -ServiceName "Redis"
Test-Installation -Name "Redis" -Type "Process" -ProcessName "redis-server"
Test-Installation -Name "Redis" -Type "Port" -Port 6379
Test-Installation -Name "Redis (C:\Redis)" -Type "Path" -Path "C:\Redis"

# Neo4j
Write-Host "`nNeo4j:" -ForegroundColor Cyan
Test-Installation -Name "Neo4j" -Type "Service" -ServiceName "neo4j"
Test-Installation -Name "Neo4j Browser" -Type "Port" -Port 7474
Test-Installation -Name "Neo4j Bolt" -Type "Port" -Port 7687
Test-Installation -Name "Neo4j (Local)" -Type "Path" -Path ".\neo4j-community-5.23.0"
Test-Installation -Name "Neo4j (C:\neo4j)" -Type "Path" -Path "C:\neo4j"

# ============================================
# STORAGE & MESSAGING
# ============================================
Write-Section "STORAGE & MESSAGING SERVICES"

# MinIO
Write-Host "`nMinIO:" -ForegroundColor Cyan
Test-Installation -Name "MinIO" -Type "Process" -ProcessName "minio"
Test-Installation -Name "MinIO API" -Type "Port" -Port 9000
Test-Installation -Name "MinIO Console" -Type "Port" -Port 9001
Test-Installation -Name "MinIO (Local)" -Type "Path" -Path ".\minio.exe"
Test-Installation -Name "MinIO (C:\minio)" -Type "Path" -Path "C:\minio\minio.exe"

# RabbitMQ
Write-Host "`nRabbitMQ:" -ForegroundColor Cyan
Test-Installation -Name "RabbitMQ" -Type "Service" -ServiceName "RabbitMQ"
Test-Installation -Name "RabbitMQ" -Type "Port" -Port 5672
Test-Installation -Name "RabbitMQ Management" -Type "Port" -Port 15672

# ============================================
# AI & ML SERVICES
# ============================================
Write-Section "AI & ML SERVICES"

# Ollama
Write-Host "`nOllama:" -ForegroundColor Cyan
$ollama = Test-Installation -Name "Ollama" -Type "Command" -CheckCommand "ollama version"
Test-Installation -Name "Ollama" -Type "Process" -ProcessName "ollama"
Test-Installation -Name "Ollama API" -Type "Port" -Port 11434

if ($ollama.Installed) {
    Write-Host "  Available models:" -ForegroundColor Gray
    $models = ollama list 2>$null | Select-Object -Skip 1
    foreach ($model in $models) {
        if ($model -and $model.Trim()) {
            $modelName = ($model -split '\s+')[0]
            if ($modelName -and $modelName -ne "NAME") {
                Write-Host "    • $modelName" -ForegroundColor Gray
            }
        }
    }
}

# CUDA/GPU Support
Write-Host "`nGPU Support:" -ForegroundColor Cyan
Test-Installation -Name "NVIDIA CUDA" -Type "Command" -CheckCommand "nvidia-smi"
Test-Installation -Name "CUDA Toolkit" -Type "Path" -Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

# ============================================
# APPLICATION SERVICES
# ============================================
Write-Section "APPLICATION SERVICES"

# Development servers
Test-Installation -Name "Vite Dev Server" -Type "Port" -Port 5173
Test-Installation -Name "Alternative Dev Server" -Type "Port" -Port 3000

# Go services
Test-Installation -Name "GPU Orchestrator" -Type "Port" -Port 8084
Test-Installation -Name "RAG Service" -Type "Port" -Port 8085

# ============================================
# PROJECT FILES
# ============================================
Write-Section "PROJECT STRUCTURE"

Test-Installation -Name "package.json" -Type "Path" -Path ".\package.json"
Test-Installation -Name "node_modules" -Type "Path" -Path ".\node_modules"
Test-Installation -Name ".env file" -Type "Path" -Path ".\.env"
Test-Installation -Name "Drizzle config" -Type "Path" -Path ".\drizzle.config.ts"
Test-Installation -Name "YoRHa Dashboard" -Type "Path" -Path ".\src\routes\yorha-dashboard"
Test-Installation -Name "Start Script" -Type "Path" -Path ".\START-LEGAL-AI.bat"

# ============================================
# SUMMARY
# ============================================
Write-Section "INSTALLATION SUMMARY"

Write-Host "`n✅ WORKING SERVICES ($($global:results.Working.Count)):" -ForegroundColor Green
foreach ($service in $global:results.Working | Select-Object -Unique) {
    Write-Host "   • $service" -ForegroundColor Green
}

if ($global:results.NeedsAttention.Count -gt 0) {
    Write-Host "`n⚠️  NEEDS ATTENTION ($($global:results.NeedsAttention.Count)):" -ForegroundColor Yellow
    foreach ($service in $global:results.NeedsAttention | Select-Object -Unique) {
        Write-Host "   • $service" -ForegroundColor Yellow
    }
}

if ($global:results.NotWorking.Count -gt 0) {
    Write-Host "`n❌ NOT INSTALLED/NOT WORKING ($($global:results.NotWorking.Count)):" -ForegroundColor Red
    foreach ($service in $global:results.NotWorking | Select-Object -Unique) {
        Write-Host "   • $service" -ForegroundColor Red
    }
}

# ============================================
# RECOMMENDATIONS
# ============================================
Write-Section "RECOMMENDED ACTIONS"

$needsAction = ($global:results.NotWorking.Count -gt 0) -or ($global:results.NeedsAttention.Count -gt 0)

if ($needsAction) {
    Write-Host "`nTo fix missing services, run as Administrator:" -ForegroundColor Cyan
    Write-Host "  .\START-NATIVE-WINDOWS-COMPLETE.ps1" -ForegroundColor White
    
    # Specific recommendations
    if ("Redis" -in $global:results.NotWorking) {
        Write-Host "`nFor Redis:" -ForegroundColor Yellow
        Write-Host "  1. Download from: https://github.com/microsoftarchive/redis/releases" -ForegroundColor Gray
        Write-Host "  2. Install to C:\Redis" -ForegroundColor Gray
        Write-Host "  3. Run: redis-server.exe" -ForegroundColor Gray
    }
    
    if ("Ollama" -in $global:results.NotWorking) {
        Write-Host "`nFor Ollama:" -ForegroundColor Yellow
        Write-Host "  1. Download from: https://ollama.ai/download" -ForegroundColor Gray
        Write-Host "  2. Install and run: ollama serve" -ForegroundColor Gray
        Write-Host "  3. Pull models: ollama pull nomic-embed-text" -ForegroundColor Gray
    }
    
    if ("Vite Dev Server" -notin $global:results.Working) {
        Write-Host "`nTo start development server:" -ForegroundColor Yellow
        Write-Host "  npm run dev" -ForegroundColor Gray
    }
} else {
    Write-Host "`n🎉 All services are running! Your system is ready." -ForegroundColor Green
    Write-Host "`nAccess YoRHa Dashboard at:" -ForegroundColor Cyan
    if ("Vite Dev Server" -in $global:results.Working) {
        Write-Host "  http://localhost:5173" -ForegroundColor White
    } else {
        Write-Host "  http://localhost:3000" -ForegroundColor White
    }
}

Write-Host "`n" -NoNewline
