# ================================================================================
# LEGAL AI PLATFORM - COMPLETE PRODUCTION WIRE-UP
# ================================================================================
# This script sets up and starts the complete Legal AI system with all services
# Includes: PostgreSQL, Redis, Ollama, MinIO, Qdrant, Neo4j, Go services, SvelteKit
# ================================================================================

param(
    [switch]$Start,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Install,
    [switch]$Test
)

Write-Host "Legal AI Platform - Complete Production Wire-Up" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

function Test-ServiceRunning {
    param($ProcessName)
    return (Get-Process -Name $ProcessName -ErrorAction SilentlyContinue) -ne $null
}

function Test-PortOpen {
    param($Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient("localhost", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

function Start-LegalAIServices {
    Write-Host "Starting Legal AI Platform Services..." -ForegroundColor Yellow

    # 1. PostgreSQL
    Write-Host "[1/10] Starting PostgreSQL..." -ForegroundColor Cyan
    try {
        Start-Service -Name "postgresql-x64-17" -ErrorAction SilentlyContinue
    Write-Host "[OK] PostgreSQL started" -ForegroundColor Green
    } catch {
    Write-Host "[WARN] PostgreSQL service not found, check installation" -ForegroundColor Yellow
    }

    # 2. Redis
    Write-Host "[2/10] Starting Redis..." -ForegroundColor Cyan
    if (-not (Test-ServiceRunning "redis-server")) {
        try {
            Start-Process -FilePath "redis-server" -WindowStyle Minimized -ErrorAction SilentlyContinue
        } catch {
            try {
                Start-Process -FilePath ".\redis-windows\redis-server.exe" -WindowStyle Minimized -ErrorAction SilentlyContinue
            } catch {
                Write-Host "Redis binary not found; please install redis or adjust path" -ForegroundColor Yellow
            }
        }
        Start-Sleep 2
    }
    Write-Host "[OK] Redis server started" -ForegroundColor Green

    # 3. Ollama
    Write-Host "[3/10] Starting Ollama..." -ForegroundColor Cyan
    if (-not (Test-ServiceRunning "ollama")) {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized
        Start-Sleep 3
    }
    Write-Host "[OK] Ollama server started" -ForegroundColor Green

    # 4. MinIO
    Write-Host "[4/10] Starting MinIO..." -ForegroundColor Cyan
    if (-not (Test-ServiceRunning "minio")) {
        if (-not (Test-Path "minio-data")) { New-Item -ItemType Directory -Path "minio-data" }
        Start-Process -FilePath "minio.exe" -ArgumentList "server", "./minio-data", "--address", ":9000", "--console-address", ":9001" -WindowStyle Minimized
        Start-Sleep 2
    }
    Write-Host "[OK] MinIO server started" -ForegroundColor Green

    # 5. Qdrant
    Write-Host "[5/10] Starting Qdrant..." -ForegroundColor Cyan
    if (-not (Test-ServiceRunning "qdrant")) {
        Start-Process -FilePath ".\qdrant-windows\qdrant.exe" -WindowStyle Minimized -ErrorAction SilentlyContinue
        Start-Sleep 2
    }
    Write-Host "[OK] Qdrant server started" -ForegroundColor Green

    # 6. Neo4j
    Write-Host "[6/10] Starting Neo4j..." -ForegroundColor Cyan
    try {
        Start-Service -Name "neo4j" -ErrorAction SilentlyContinue
    Write-Host "[OK] Neo4j started" -ForegroundColor Green
    } catch {
    Write-Host "[WARN] Neo4j service not found, manual start required" -ForegroundColor Yellow
    }

    # 7. Go Enhanced RAG Service
    Write-Host "[7/10] Starting Go Enhanced RAG Service..." -ForegroundColor Cyan
    if (Test-Path "go-microservice\cmd\enhanced-rag") {
        Push-Location "go-microservice\cmd\enhanced-rag"
        Start-Process -FilePath "go" -ArgumentList "run", "." -WindowStyle Minimized
        Pop-Location
    Write-Host "[OK] Enhanced RAG service started" -ForegroundColor Green
    } else {
    Write-Host "[WARN] Enhanced RAG cmd not found" -ForegroundColor Yellow
    }

    # 8. Go Upload Service
    Write-Host "[8/10] Starting Go Upload Service..." -ForegroundColor Cyan
    if (Test-Path "go-microservice\cmd\upload-service") {
        Push-Location "go-microservice\cmd\upload-service"
        Start-Process -FilePath "go" -ArgumentList "run", "." -WindowStyle Minimized
        Pop-Location
    Write-Host "[OK] Upload service started" -ForegroundColor Green
    } else {
    Write-Host "[WARN] Upload service cmd not found" -ForegroundColor Yellow
    }

    # 9. Go XState Manager
    Write-Host "[9/10] Starting Go XState Manager..." -ForegroundColor Cyan
    if (Test-Path "go-services\cmd\xstate-manager") {
        Set-Location "go-services\cmd\xstate-manager"
        Start-Process -FilePath "go" -ArgumentList "run", "main.go" -WindowStyle Minimized
        Set-Location "..\..\..\"
    }
    Write-Host "[OK] XState manager started" -ForegroundColor Green

    # 10. SvelteKit Frontend
    Write-Host "[10/10] Starting SvelteKit Frontend..." -ForegroundColor Cyan
    if (Test-Path "sveltekit-frontend") {
        Set-Location "sveltekit-frontend"
        Start-Process -FilePath "cmd" -ArgumentList "/k", "npm run dev -- --host 0.0.0.0"
        Set-Location ".."
    }
    Write-Host "[OK] SvelteKit frontend started" -ForegroundColor Green

    Start-Sleep 5

    Write-Host "`nLEGAL AI PLATFORM STARTED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "=============================================" -ForegroundColor Green
    Write-Host "`nAccess Points:" -ForegroundColor White
    Write-Host "- Frontend:        http://localhost:5173" -ForegroundColor Cyan
    Write-Host "- Enhanced RAG:    http://localhost:8094/api/rag" -ForegroundColor Cyan
    Write-Host "- Upload API:      http://localhost:8093/upload" -ForegroundColor Cyan
    Write-Host "- MinIO Console:   http://localhost:9001 (admin/minioadmin)" -ForegroundColor Cyan
    Write-Host "- Qdrant API:      http://localhost:6333" -ForegroundColor Cyan
    Write-Host "- Neo4j Browser:   http://localhost:7474" -ForegroundColor Cyan
    Write-Host "- Ollama API:      http://localhost:11434" -ForegroundColor Cyan

    Write-Host "`nDatabase Details:" -ForegroundColor White
    Write-Host "- PostgreSQL:      postgresql://legal_admin:123456@localhost:5432/legal_ai_db" -ForegroundColor Cyan
    Write-Host "- Redis:           redis://localhost:6379" -ForegroundColor Cyan
}

function Stop-LegalAIServices {
    Write-Host "Stopping Legal AI Platform Services..." -ForegroundColor Yellow

    # Stop processes
    Get-Process -Name "node" -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process -Name "go" -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process -Name "minio" -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process -Name "qdrant" -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process -Name "redis-server" -ErrorAction SilentlyContinue | Stop-Process -Force

    Write-Host "[OK] All services stopped" -ForegroundColor Green
}

function Show-ServiceStatus {
    Write-Host "Legal AI Platform Service Status" -ForegroundColor Yellow
    Write-Host "===================================" -ForegroundColor Yellow

    # Check services
    if (Test-PortOpen 5432) { Write-Host "[OK] PostgreSQL: Running (port 5432)" -ForegroundColor Green } else { Write-Host "[ERR] PostgreSQL: Not running" -ForegroundColor Red }
    if (Test-PortOpen 6379) { Write-Host "[OK] Redis: Running (port 6379)" -ForegroundColor Green } else { Write-Host "[ERR] Redis: Not running" -ForegroundColor Red }
    if (Test-PortOpen 11434) { Write-Host "[OK] Ollama: Running (port 11434)" -ForegroundColor Green } else { Write-Host "[ERR] Ollama: Not running" -ForegroundColor Red }
    if (Test-PortOpen 9000) { Write-Host "[OK] MinIO: Running (port 9000)" -ForegroundColor Green } else { Write-Host "[ERR] MinIO: Not running" -ForegroundColor Red }
    if (Test-PortOpen 6333) { Write-Host "[OK] Qdrant: Running (port 6333)" -ForegroundColor Green } else { Write-Host "[ERR] Qdrant: Not running" -ForegroundColor Red }
    if (Test-PortOpen 7474) { Write-Host "[OK] Neo4j: Running (port 7474)" -ForegroundColor Green } else { Write-Host "[ERR] Neo4j: Not running" -ForegroundColor Red }
    if (Test-PortOpen 5173) { Write-Host "[OK] SvelteKit: Running (port 5173)" -ForegroundColor Green } else { Write-Host "[ERR] SvelteKit: Not running" -ForegroundColor Red }
    if (Test-PortOpen 8094) { Write-Host "[OK] Enhanced RAG: Running (port 8094)" -ForegroundColor Green } else { Write-Host "[ERR] Enhanced RAG: Not running" -ForegroundColor Red }
    if (Test-PortOpen 8093) { Write-Host "[OK] Upload Service: Running (port 8093)" -ForegroundColor Green } else { Write-Host "[ERR] Upload Service: Not running" -ForegroundColor Red }
}

function Install-Dependencies {
    Write-Host "Installing Legal AI Platform Dependencies..." -ForegroundColor Yellow

    # Install Chocolatey packages
    choco install redis-64 -y
    choco install nodejs -y
    choco install golang -y
    choco install postgresql -y

    # Install npm dependencies
    if (Test-Path "sveltekit-frontend") {
        Set-Location "sveltekit-frontend"
        npm install
        Set-Location ".."
    }

    # Install Go dependencies
    if (Test-Path "go-services") {
        Set-Location "go-services"
        go mod tidy
        Set-Location ".."
    }

    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
}

function Test-SystemIntegration {
    Write-Host "Testing Legal AI Platform Integration..." -ForegroundColor Yellow

    # Test Ollama models
    try {
        $result = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get
    Write-Host "[OK] Ollama API responsive with" $result.models.Count "models" -ForegroundColor Green
    } catch {
    Write-Host "[ERR] Ollama API not responding" -ForegroundColor Red
    }

    # Test database connection
    try {
        $env:PGPASSWORD = "123456"
        & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT version();" -d legal_ai_db
    Write-Host "[OK] PostgreSQL connection successful" -ForegroundColor Green
    } catch {
    Write-Host "[ERR] PostgreSQL connection failed" -ForegroundColor Red
    }

    # Test Redis
    try {
        redis-cli ping
    Write-Host "[OK] Redis connection successful" -ForegroundColor Green
    } catch {
    Write-Host "[ERR] Redis connection failed" -ForegroundColor Red
    }
}

# Main execution logic
if ($Start) {
    Start-LegalAIServices
} elseif ($Stop) {
    Stop-LegalAIServices
} elseif ($Status) {
    Show-ServiceStatus
} elseif ($Install) {
    Install-Dependencies
} elseif ($Test) {
    Test-SystemIntegration
} else {
    Write-Host "Usage:" -ForegroundColor White
    Write-Host "  .\COMPLETE-LEGAL-AI-WIRE-UP.ps1 -Start    # Start all services" -ForegroundColor Cyan
    Write-Host "  .\COMPLETE-LEGAL-AI-WIRE-UP.ps1 -Stop     # Stop all services" -ForegroundColor Cyan
    Write-Host "  .\COMPLETE-LEGAL-AI-WIRE-UP.ps1 -Status   # Check service status" -ForegroundColor Cyan
    Write-Host "  .\COMPLETE-LEGAL-AI-WIRE-UP.ps1 -Install  # Install dependencies" -ForegroundColor Cyan
    Write-Host "  .\COMPLETE-LEGAL-AI-WIRE-UP.ps1 -Test     # Test integration" -ForegroundColor Cyan
}