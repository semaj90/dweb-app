# PowerShell script to start AI Synthesis services on Windows
# Native Windows deployment - No Docker required

param(
    [switch]$SkipHealthCheck = $false,
    [switch]$EnableDebug = $false
)

$ErrorActionPreference = "Stop"
$script:StartTime = Get-Date

Write-Host "🚀 Starting AI Synthesis System..." -ForegroundColor Cyan
Write-Host "⏰ Start Time: $($script:StartTime)" -ForegroundColor Gray

# Configuration
$env:REDIS_HOST = "localhost"
$env:REDIS_PORT = "6379"
$env:OLLAMA_URL = "http://localhost:11434"
$env:CONTEXT7_URL = "http://localhost:4000"
$env:ENHANCED_RAG_URL = "http://localhost:8094"
$env:GPU_ORCHESTRATOR_URL = "http://localhost:8095"
$env:NODE_ENV = "development"

# Service definitions
$services = @(
    @{
        Name = "Redis Cache"
        Port = 6379
        Start = {
            # Check if Redis is installed
            if (-not (Get-Command redis-server -ErrorAction SilentlyContinue)) {
                Write-Host "⚠️ Redis not found. Installing via Chocolatey..." -ForegroundColor Yellow
                choco install redis-64 -y
            }
            
            # Start Redis in background
            $job = Start-Job -ScriptBlock {
                redis-server --port 6379 --maxmemory 512mb --maxmemory-policy allkeys-lru
            }
            return $job
        }
        HealthCheck = { Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet }
    },
    @{
        Name = "Ollama AI"
        Port = 11434
        Start = {
            # Check if Ollama is running
            $ollamaRunning = Get-Process ollama -ErrorAction SilentlyContinue
            if (-not $ollamaRunning) {
                Write-Host "Starting Ollama service..." -ForegroundColor Green
                Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
                Start-Sleep -Seconds 3
                
                # Pull legal models if not present
                Write-Host "Checking for legal models..." -ForegroundColor Gray
                & ollama list | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "Pulling llama2 model..." -ForegroundColor Yellow
                    & ollama pull llama2
                }
            }
        }
        HealthCheck = {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2
                return $true
            } catch {
                return $false
            }
        }
    }
)

# Function to check service health
function Test-ServiceHealth {
    param(
        [string]$ServiceName,
        [int]$Port,
        [scriptblock]$HealthCheck
    )
    
    Write-Host "  Checking $ServiceName (Port $Port)..." -NoNewline
    
    $healthy = & $HealthCheck
    
    if ($healthy) {
        Write-Host " ✅ HEALTHY" -ForegroundColor Green
        return $true
    } else {
        Write-Host " ❌ OFFLINE" -ForegroundColor Red
        return $false
    }
}

# Function to start services
function Start-Services {
    $jobs = @{}
    
    foreach ($service in $services) {
        Write-Host "`n📦 Starting $($service.Name)..." -ForegroundColor Cyan
        
        # Check if already running
        $isHealthy = Test-ServiceHealth -ServiceName $service.Name -Port $service.Port -HealthCheck $service.HealthCheck
        
        if ($isHealthy) {
            Write-Host "  Already running, skipping..." -ForegroundColor Gray
        } else {
            # Start the service
            if ($service.Start) {
                $job = & $service.Start
                if ($job) {
                    $jobs[$service.Name] = $job
                }
                
                # Wait for service to start
                $retries = 10
                $started = $false
                
                while ($retries -gt 0 -and -not $started) {
                    Start-Sleep -Seconds 2
                    $started = Test-ServiceHealth -ServiceName $service.Name -Port $service.Port -HealthCheck $service.HealthCheck
                    $retries--
                }
                
                if (-not $started) {
                    Write-Host "  ⚠️ Failed to start $($service.Name)" -ForegroundColor Yellow
                }
            }
        }
    }
    
    return $jobs
}

# Function to integrate with existing services
function Connect-ExistingServices {
    Write-Host "`n🔗 Connecting to existing Legal AI services..." -ForegroundColor Cyan
    
    $existingServices = @(
        @{ Name = "Enhanced RAG"; Port = 8094; Url = "http://localhost:8094/health" },
        @{ Name = "GPU Orchestrator"; Port = 8095; Url = "http://localhost:8095/health" },
        @{ Name = "Context7 MCP"; Port = 4000; Url = "http://localhost:4000/health" }
    )
    
    foreach ($service in $existingServices) {
        Write-Host "  Checking $($service.Name)..." -NoNewline
        try {
            $response = Invoke-RestMethod -Uri $service.Url -Method Get -TimeoutSec 2
            Write-Host " ✅ Connected" -ForegroundColor Green
            
            # Store connection in environment
            $envName = $service.Name.Replace(" ", "_").ToUpper() + "_CONNECTED"
            [Environment]::SetEnvironmentVariable($envName, "true", "Process")
        } catch {
            Write-Host " ⚠️ Not available" -ForegroundColor Yellow
        }
    }
}

# Main execution
try {
    # Start all services
    $jobs = Start-Services
    
    # Connect to existing services
    Connect-ExistingServices
    
    # Summary
    $duration = (Get-Date) - $script:StartTime
    Write-Host "`n✨ AI Synthesis System Started Successfully!" -ForegroundColor Green
    Write-Host "⏱️ Total startup time: $($duration.TotalSeconds.ToString('F2')) seconds" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🔗 Available Endpoints:" -ForegroundColor Cyan
    Write-Host "  Main API:     http://localhost:5173/api/ai-synthesizer"
    Write-Host "  Health:       http://localhost:5173/api/ai-synthesizer/health"
    Write-Host "  Test Suite:   http://localhost:5173/api/ai-synthesizer/test"
    Write-Host "  Streaming:    http://localhost:5173/api/ai-synthesizer/stream/{id}"
    Write-Host ""
    
    if ($EnableDebug) {
        Write-Host "Debug mode enabled. Press Ctrl+C to exit..." -ForegroundColor Yellow
        while ($true) {
            Start-Sleep -Seconds 60
        }
    }
    
} catch {
    Write-Host "❌ Error during startup: $_" -ForegroundColor Red
    exit 1
}
