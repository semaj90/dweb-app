# Legal AI Asynchronous Indexing System Startup Script
# PowerShell script to orchestrate the entire indexing pipeline

param(
    [string]$Path = "C:\Users\james\Desktop\deeds-web\deeds-web-app",
    [int]$Workers = 8,
    [int]$BatchSize = 100,
    [switch]$Production,
    [switch]$Monitor,
    [switch]$SkipHealthCheck
)

# Configuration
$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

Write-Host "🚀 Legal AI Asynchronous Indexing System" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Set working directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "📁 Working Directory: $ScriptDir" -ForegroundColor Blue
Write-Host "🎯 Target Path: $Path" -ForegroundColor Blue
Write-Host "👥 Workers: $Workers" -ForegroundColor Blue
Write-Host "📦 Batch Size: $BatchSize" -ForegroundColor Blue

# Function to check if a service is running on a port
function Test-ServicePort {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Function to wait for service to be ready
function Wait-ForService {
    param([string]$Name, [string]$Url, [int]$TimeoutSeconds = 60)
    
    Write-Host "⏳ Waiting for $Name to be ready..." -ForegroundColor Yellow
    $startTime = Get-Date
    
    do {
        try {
            $response = Invoke-RestMethod -Uri $Url -Method GET -TimeoutSec 5
            Write-Host "✅ $Name is ready!" -ForegroundColor Green
            return $true
        } catch {
            Start-Sleep -Seconds 2
            $elapsed = (Get-Date) - $startTime
            if ($elapsed.TotalSeconds -gt $TimeoutSeconds) {
                Write-Host "❌ $Name failed to start within $TimeoutSeconds seconds" -ForegroundColor Red
                return $false
            }
        }
    } while ($true)
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow
    
    $missing = @()
    
    # Check Go
    try {
        $goVersion = go version 2>$null
        Write-Host "✅ Go: $goVersion" -ForegroundColor Green
    } catch {
        $missing += "Go"
    }
    
    # Check Python
    try {
        $pythonVersion = python --version 2>$null
        Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
    } catch {
        $missing += "Python"
    }
    
    # Check Node.js
    try {
        $nodeVersion = node --version 2>$null
        Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
    } catch {
        $missing += "Node.js"
    }
    
    # Check PM2
    try {
        $pm2Version = pm2 --version 2>$null
        Write-Host "✅ PM2: $pm2Version" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ PM2 not installed, installing..." -ForegroundColor Yellow
        npm install -g pm2
    }
    
    # Check zx
    try {
        $zxVersion = zx --version 2>$null
        Write-Host "✅ zx: $zxVersion" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ zx not installed, installing..." -ForegroundColor Yellow
        npm install -g zx
    }
    
    if ($missing.Count -gt 0) {
        Write-Host "❌ Missing prerequisites: $($missing -join ', ')" -ForegroundColor Red
        throw "Please install missing prerequisites"
    }
}

# Function to start Ollama if not running
function Start-Ollama {
    if (-not $SkipHealthCheck) {
        Write-Host "🔍 Checking Ollama..." -ForegroundColor Yellow
        
        if (-not (Test-ServicePort 11434)) {
            Write-Host "🚀 Starting Ollama..." -ForegroundColor Yellow
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            
            if (-not (Wait-ForService "Ollama" "http://localhost:11434/api/tags" 30)) {
                throw "Failed to start Ollama"
            }
        } else {
            Write-Host "✅ Ollama is already running" -ForegroundColor Green
        }
        
        # Ensure required models are available
        Write-Host "📥 Checking required models..." -ForegroundColor Yellow
        $models = ollama list | Out-String
        
        if (-not $models.Contains("nomic-embed-text")) {
            Write-Host "📥 Pulling nomic-embed-text model..." -ForegroundColor Yellow
            ollama pull nomic-embed-text
        }
        
        Write-Host "✅ Ollama setup complete" -ForegroundColor Green
    }
}

# Function to prepare Go service
function Initialize-GoService {
    Write-Host "🔧 Preparing Go indexing service..." -ForegroundColor Yellow
    
    if (-not (Test-Path "go.mod")) {
        Write-Host "📦 Initializing Go module..." -ForegroundColor Blue
        go mod init async-indexer
    }
    
    Write-Host "📦 Installing Go dependencies..." -ForegroundColor Blue
    go mod tidy
    
    # Build the service
    Write-Host "🔨 Building Go service..." -ForegroundColor Blue
    go build -o async-indexer.exe async-indexer.go
    
    Write-Host "✅ Go service prepared" -ForegroundColor Green
}

# Function to prepare Python environment
function Initialize-PythonEnvironment {
    Write-Host "🐍 Preparing Python environment..." -ForegroundColor Yellow
    
    # Install required packages
    $packages = @(
        "asyncio",
        "websockets", 
        "requests",
        "dataclasses"
    )
    
    foreach ($package in $packages) {
        Write-Host "📦 Installing $package..." -ForegroundColor Blue
        pip install $package --quiet
    }
    
    Write-Host "✅ Python environment prepared" -ForegroundColor Green
}

# Function to prepare Node.js environment
function Initialize-NodeEnvironment {
    Write-Host "📦 Preparing Node.js environment..." -ForegroundColor Yellow
    
    if (-not (Test-Path "package.json")) {
        Write-Host "📝 Creating package.json..." -ForegroundColor Blue
        @"
{
  "name": "legal-ai-indexing",
  "version": "1.0.0",
  "type": "module",
  "dependencies": {
    "express": "^4.18.2",
    "ws": "^8.13.0",
    "zx": "^7.2.3"
  }
}
"@ | Out-File -FilePath "package.json" -Encoding UTF8
    }
    
    Write-Host "📦 Installing Node.js dependencies..." -ForegroundColor Blue
    npm install --silent
    
    Write-Host "✅ Node.js environment prepared" -ForegroundColor Green
}

# Function to start all services with PM2
function Start-AllServices {
    Write-Host "🚀 Starting all indexing services..." -ForegroundColor Cyan
    
    # Create logs directory
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" -Force | Out-Null
    }
    
    # Update PM2 configuration with current path
    $pm2Config = Get-Content "pm2.config.js" -Raw
    $pm2Config = $pm2Config -replace "process\.cwd\(\)", "'$Path'"
    $pm2Config | Out-File -FilePath "pm2.runtime.config.js" -Encoding UTF8
    
    # Start services with PM2
    Write-Host "🎯 Starting PM2 ecosystem..." -ForegroundColor Blue
    pm2 start pm2.runtime.config.js
    
    # Wait for services to be ready
    $services = @(
        @{ Name = "Go Indexer"; Url = "http://localhost:8081/api/health" },
        @{ Name = "Monitor Dashboard"; Url = "http://localhost:8084/api/health" }
    )
    
    foreach ($service in $services) {
        Wait-ForService $service.Name $service.Url 60 | Out-Null
    }
    
    Write-Host "✅ All services started successfully!" -ForegroundColor Green
}

# Function to show service status
function Show-ServiceStatus {
    Write-Host ""
    Write-Host "📊 Service Status:" -ForegroundColor Cyan
    Write-Host "=" * 30 -ForegroundColor Cyan
    
    pm2 status
    
    Write-Host ""
    Write-Host "🌐 Access Points:" -ForegroundColor Cyan
    Write-Host "  • Monitor Dashboard: http://localhost:8084" -ForegroundColor Green
    Write-Host "  • Go Indexer API:    http://localhost:8081" -ForegroundColor Green
    Write-Host "  • WebSocket Stream:  ws://localhost:8082" -ForegroundColor Green
    Write-Host "  • AutoGen WebSocket: ws://localhost:8083" -ForegroundColor Green
}

# Function to start indexing process
function Start-IndexingProcess {
    Write-Host ""
    Write-Host "🎯 Starting indexing process..." -ForegroundColor Cyan
    
    # Start the concurrent indexer
    Write-Host "⚡ Launching concurrent indexer..." -ForegroundColor Yellow
    
    $indexerArgs = @(
        $Path,
        "--workers", $Workers,
        "--batch", $BatchSize
    )
    
    if ($Production) {
        $indexerArgs += "--production"
    }
    
    Start-Process -FilePath "node" -ArgumentList @("--loader", "@zx/loader", "concurrent-indexer.mjs") + $indexerArgs -NoNewWindow
    
    Write-Host "✅ Indexing process started!" -ForegroundColor Green
    Write-Host "📊 Monitor progress at: http://localhost:8084" -ForegroundColor Blue
}

# Main execution
try {
    Write-Host ""
    
    # Step 1: Check prerequisites
    Test-Prerequisites
    
    # Step 2: Start Ollama
    Start-Ollama
    
    # Step 3: Prepare environments
    Initialize-GoService
    Initialize-PythonEnvironment
    Initialize-NodeEnvironment
    
    # Step 4: Start all services
    Start-AllServices
    
    # Step 5: Show status
    Show-ServiceStatus
    
    # Step 6: Start indexing if requested
    if (-not $Monitor) {
        Start-IndexingProcess
    }
    
    Write-Host ""
    Write-Host "🎉 Legal AI Indexing System is fully operational!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  pm2 status          - Check service status" -ForegroundColor Gray
    Write-Host "  pm2 logs            - View all logs" -ForegroundColor Gray
    Write-Host "  pm2 stop all        - Stop all services" -ForegroundColor Gray
    Write-Host "  pm2 restart all     - Restart all services" -ForegroundColor Gray
    Write-Host ""
    
    if ($Monitor) {
        Write-Host "👀 Running in monitor-only mode. Access dashboard at: http://localhost:8084" -ForegroundColor Blue
        Write-Host "   To start indexing, run without -Monitor flag" -ForegroundColor Gray
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    
    # Cleanup on failure
    Write-Host "Cleaning up..." -ForegroundColor Yellow
    try {
        pm2 delete all 2>$null
    } catch {
        # Ignore cleanup errors
    }
    
    exit 1
}

# Keep script running if in monitor mode
if ($Monitor) {
    Write-Host "Press Ctrl+C to stop monitoring..." -ForegroundColor Gray
    try {
        while ($true) {
            Start-Sleep -Seconds 10
            # Optional: Show periodic status updates
        }
    } catch {
        Write-Host "Stopping services..." -ForegroundColor Yellow
        pm2 stop all
    }
}