#!/usr/bin/env powershell

<#
.SYNOPSIS
Fix Docker Container Health Issues for Legal AI System

.DESCRIPTION
Diagnoses and fixes unhealthy Docker containers in the Legal AI system,
specifically targeting Ollama GPU and Qdrant containers.

.EXAMPLE
.\Fix-DockerContainerHealth.ps1
#>

[CmdletBinding()]
param()

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║ $($Title.PadRight(76)) ║" -ForegroundColor Cyan  
    Write-Host "╚══════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "[$Title]" -ForegroundColor Yellow
    Write-Host ("─" * 80) -ForegroundColor Gray
}

function Test-ContainerHealth {
    param([string]$ContainerName)
    
    try {
        $health = docker inspect $ContainerName --format "{{.State.Health.Status}}" 2>$null
        return $health
    }
    catch {
        return "unknown"
    }
}

function Get-ContainerLogs {
    param([string]$ContainerName, [int]$Lines = 10)
    
    try {
        $logs = docker logs $ContainerName --tail $Lines 2>&1
        return $logs
    }
    catch {
        return "Could not retrieve logs for $ContainerName"
    }
}

function Fix-OllamaContainer {
    Write-Section "🤖 Fixing Ollama Container"
    
    $ollamaHealth = Test-ContainerHealth "legal-ollama-gpu"
    Write-Host "Current Ollama health: $ollamaHealth" -ForegroundColor $(if($ollamaHealth -eq "healthy") {"Green"} else {"Red"})
    
    if ($ollamaHealth -ne "healthy") {
        Write-Host "Diagnosing Ollama issues..." -ForegroundColor Yellow
        
        # Check if container is running
        $isRunning = docker ps --filter "name=legal-ollama-gpu" --format "{{.Names}}" 2>$null
        
        if (-not $isRunning) {
            Write-Host "❌ Ollama container is not running" -ForegroundColor Red
            Write-Host "Starting Ollama container..." -ForegroundColor Yellow
            docker start legal-ollama-gpu
            Start-Sleep 10
        }
        
        # Test GPU access
        Write-Host "Checking GPU access..." -ForegroundColor Yellow
        $gpuTest = docker exec legal-ollama-gpu nvidia-smi 2>$null
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠️ GPU not accessible, checking CPU fallback..." -ForegroundColor Yellow
            
            # Check if CPU version should be used
            $cpuContainer = docker ps --filter "name=legal-ollama-cpu" --format "{{.Names}}" 2>$null
            
            if (-not $cpuContainer) {
                Write-Host "Creating CPU-only Ollama container..." -ForegroundColor Yellow
                docker stop legal-ollama-gpu 2>$null
                docker run -d --name legal-ollama-cpu `
                    --restart unless-stopped `
                    -p 11434:11434 `
                    -v ollama:/root/.ollama `
                    ollama/ollama
                Start-Sleep 15
                Write-Host "✅ CPU Ollama container created" -ForegroundColor Green
            }
        } else {
            Write-Host "✅ GPU accessible" -ForegroundColor Green
            
            # Restart Ollama service inside container
            Write-Host "Restarting Ollama service..." -ForegroundColor Yellow
            docker exec legal-ollama-gpu pkill ollama 2>$null
            Start-Sleep 3
            docker exec legal-ollama-gpu nohup ollama serve 2>&1 &
            Start-Sleep 10
        }
        
        # Test API
        Write-Host "Testing Ollama API..." -ForegroundColor Yellow
        $apiTest = docker exec legal-ollama-gpu curl -s http://localhost:11434/api/tags 2>$null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Ollama API responding" -ForegroundColor Green
        } else {
            Write-Host "❌ Ollama API not responding" -ForegroundColor Red
            Write-Host "Recent logs:" -ForegroundColor Yellow
            Get-ContainerLogs "legal-ollama-gpu" | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        }
    } else {
        Write-Host "✅ Ollama container is healthy" -ForegroundColor Green
    }
}

function Fix-QdrantContainer {
    Write-Section "🔍 Fixing Qdrant Container"
    
    $qdrantHealth = Test-ContainerHealth "legal-qdrant-optimized"
    Write-Host "Current Qdrant health: $qdrantHealth" -ForegroundColor $(if($qdrantHealth -eq "healthy") {"Green"} else {"Red"})
    
    if ($qdrantHealth -ne "healthy") {
        Write-Host "Diagnosing Qdrant issues..." -ForegroundColor Yellow
        
        # Check if container is running
        $isRunning = docker ps --filter "name=legal-qdrant-optimized" --format "{{.Names}}" 2>$null
        
        if (-not $isRunning) {
            Write-Host "❌ Qdrant container is not running" -ForegroundColor Red
            Write-Host "Starting Qdrant container..." -ForegroundColor Yellow
            docker start legal-qdrant-optimized
            Start-Sleep 10
        }
        
        # Test internal API
        Write-Host "Testing Qdrant internal API..." -ForegroundColor Yellow
        $internalTest = docker exec legal-qdrant-optimized curl -s http://localhost:6333/health 2>$null
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Qdrant internal API not responding" -ForegroundColor Red
            Write-Host "Recent logs:" -ForegroundColor Yellow
            Get-ContainerLogs "legal-qdrant-optimized" | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
            
            Write-Host "Recreating Qdrant container..." -ForegroundColor Yellow
            docker stop legal-qdrant-optimized 2>$null
            docker rm legal-qdrant-optimized 2>$null
            docker run -d --name legal-qdrant-optimized `
                --restart unless-stopped `
                -p 6333:6333 `
                -p 6334:6334 `
                -v qdrant_storage:/qdrant/storage `
                qdrant/qdrant:latest
            Start-Sleep 15
            Write-Host "✅ Qdrant container recreated" -ForegroundColor Green
        } else {
            Write-Host "✅ Qdrant internal API responding" -ForegroundColor Green
        }
        
        # Test external API
        Write-Host "Testing Qdrant external API..." -ForegroundColor Yellow
        try {
            $externalTest = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method Get -TimeoutSec 5
            Write-Host "✅ Qdrant external API responding" -ForegroundColor Green
        } catch {
            Write-Host "❌ Qdrant external API not accessible: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "Checking port mapping..." -ForegroundColor Yellow
            docker port legal-qdrant-optimized
        }
    } else {
        Write-Host "✅ Qdrant container is healthy" -ForegroundColor Green
    }
}

function Show-ServiceStatus {
    Write-Section "📊 Service Status Summary"
    
    $containers = @("legal-postgres-optimized", "legal-redis-cluster", "legal-qdrant-optimized", "legal-ollama-gpu", "legal-ollama-cpu")
    
    Write-Host "Container Status:" -ForegroundColor Cyan
    Write-Host ("─" * 50) -ForegroundColor Gray
    
    foreach ($container in $containers) {
        $status = docker ps --filter "name=$container" --format "{{.Status}}" 2>$null
        $health = Test-ContainerHealth $container
        
        if ($status) {
            $statusColor = switch ($health) {
                "healthy" { "Green" }
                "unhealthy" { "Red" }
                default { "Yellow" }
            }
            Write-Host "✅ $container" -ForegroundColor Green -NoNewline
            Write-Host " - $health" -ForegroundColor $statusColor
        } else {
            Write-Host "❌ $container - not running" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "Service Endpoints:" -ForegroundColor Cyan
    Write-Host ("─" * 50) -ForegroundColor Gray
    Write-Host "• PostgreSQL:  localhost:5432" -ForegroundColor White
    Write-Host "• Redis:       localhost:6379" -ForegroundColor White  
    Write-Host "• Qdrant:      localhost:6333" -ForegroundColor White
    Write-Host "• Ollama:      localhost:11434" -ForegroundColor White
}

function Test-AllServices {
    Write-Section "🧪 Testing All Service Endpoints"
    
    # Test PostgreSQL
    Write-Host "Testing PostgreSQL..." -ForegroundColor Yellow -NoNewline
    $pgTest = docker exec legal-postgres-optimized pg_isready -h localhost -p 5432 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅" -ForegroundColor Green
    } else {
        Write-Host " ❌" -ForegroundColor Red
    }
    
    # Test Redis
    Write-Host "Testing Redis..." -ForegroundColor Yellow -NoNewline
    $redisTest = docker exec legal-redis-cluster redis-cli ping 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅" -ForegroundColor Green
    } else {
        Write-Host " ❌" -ForegroundColor Red
    }
    
    # Test Qdrant
    Write-Host "Testing Qdrant..." -ForegroundColor Yellow -NoNewline
    try {
        $qdrantTest = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method Get -TimeoutSec 3
        Write-Host " ✅" -ForegroundColor Green
    } catch {
        Write-Host " ❌" -ForegroundColor Red
    }
    
    # Test Ollama
    Write-Host "Testing Ollama..." -ForegroundColor Yellow -NoNewline
    try {
        $ollamaTest = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
        Write-Host " ✅" -ForegroundColor Green
    } catch {
        Write-Host " ❌" -ForegroundColor Red
    }
}

# Main execution
try {
    Write-Header "🐳 DOCKER HEALTH CHECK & FIX UTILITY"
    
    Write-Host "🔍 Diagnosing Docker container health issues..." -ForegroundColor Cyan
    
    # Show current status
    Write-Section "📋 Current Container Status"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    # Fix containers
    Fix-OllamaContainer
    Fix-QdrantContainer
    
    # Wait for health checks to update
    Write-Section "⏳ Waiting for Health Checks"
    Write-Host "Allowing time for health checks to update..." -ForegroundColor Yellow
    Start-Sleep 20
    
    # Show final status
    Show-ServiceStatus
    Test-AllServices
    
    Write-Header "✅ DOCKER HEALTH FIX COMPLETE"
    
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "• Run your SvelteKit app: npm run dev" -ForegroundColor White
    Write-Host "• Monitor logs: docker-compose logs -f" -ForegroundColor White
    Write-Host "• Check health: docker ps" -ForegroundColor White
    
    # Offer next actions
    Write-Host ""
    $choice = Read-Host "Would you like to start the SvelteKit dev server? (y/N)"
    
    if ($choice -eq 'y' -or $choice -eq 'Y') {
        Write-Host "🚀 Starting SvelteKit development server..." -ForegroundColor Green
        npm run dev
    }
    
} catch {
    Write-Host "❌ Error during health check: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please check Docker installation and container configuration." -ForegroundColor Yellow
}