# Enhanced Windows Development Environment Startup
# start-dev-windows.ps1

param(
    [switch]$SkipChecks,
    [switch]$GPUMode,
    [switch]$Debug
)

$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "Legal AI Development Environment"

# Colors for output
function Write-Status { param($msg) Write-Host "✅ $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "ℹ️  $msg" -ForegroundColor Cyan }
function Write-Warning { param($msg) Write-Host "⚠️  $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "❌ $msg" -ForegroundColor Red }
function Write-Header { 
    param($msg) 
    Write-Host "`n$('='*60)" -ForegroundColor Blue
    Write-Host $msg -ForegroundColor Blue
    Write-Host "$('='*60)`n" -ForegroundColor Blue
}

Clear-Host
Write-Header "LEGAL AI DEVELOPMENT ENVIRONMENT"

# Check Node.js version
Write-Info "Checking Node.js version..."
$nodeVersion = node --version
if ($nodeVersion -match "v(\d+)") {
    $majorVersion = [int]$matches[1]
    if ($majorVersion -lt 18) {
        Write-Error "Node.js 18+ required. Current: $nodeVersion"
        exit 1
    }
    Write-Status "Node.js $nodeVersion detected"
}

# Check Go installation
Write-Info "Checking Go installation..."
$goVersion = go version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Status "Go detected: $goVersion"
} else {
    Write-Warning "Go not installed - microservice features disabled"
}

# Check GPU if requested
if ($GPUMode) {
    Write-Info "Checking GPU availability..."
    $nvidiaCheck = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Status "GPU detected: $nvidiaCheck"
        $env:ENABLE_GPU = "true"
        $env:CUDA_VISIBLE_DEVICES = "0"
    } else {
        Write-Warning "No NVIDIA GPU detected - CPU mode enabled"
        $env:ENABLE_GPU = "false"
    }
}

# Kill existing processes on same ports
Write-Info "Checking for port conflicts..."
$ports = @(5173, 8084, 6379, 11434, 5432)
foreach ($port in $ports) {
    $process = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($process) {
        Write-Warning "Port $port is in use, attempting to free it..."
        $pid = $process.OwningProcess
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}
Write-Status "Ports cleared"

# Start Redis if available
Write-Info "Starting Redis cache..."
$redisRunning = $false
try {
    $redisCheck = redis-cli ping 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Redis already running"
        $redisRunning = $true
    }
} catch {}

if (-not $redisRunning) {
    if (Test-Path "C:\Program Files\Redis\redis-server.exe") {
        Start-Process -FilePath "C:\Program Files\Redis\redis-server.exe" -WindowStyle Hidden
        Write-Status "Redis started"
    } elseif (Get-Command redis-server -ErrorAction SilentlyContinue) {
        Start-Process redis-server -WindowStyle Hidden
        Write-Status "Redis started"
    } else {
        Write-Warning "Redis not found - using memory cache"
        $env:USE_MEMORY_CACHE = "true"
    }
}

# Start Ollama if available
Write-Info "Checking Ollama service..."
$ollamaRunning = $false
try {
    $ollamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2
    Write-Status "Ollama already running"
    $ollamaRunning = $true
} catch {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Write-Info "Starting Ollama service..."
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
        
        # Check if gemma3-legal model is available
        try {
            $models = ollama list | Out-String
            if ($models -notmatch "gemma3-legal") {
                Write-Warning "Gemma3-legal model not found. Pull it with: ollama pull gemma3-legal:latest"
            } else {
                Write-Status "Gemma3-legal model available"
            }
        } catch {}
    } else {
        Write-Warning "Ollama not installed - AI features limited"
    }
}

# Start Go microservice if available
if (Test-Path "../main.go") {
    Write-Info "Starting Go microservice..."
    
    # Set Go environment variables
    $env:PORT = "8084"
    $env:REDIS_ADDR = "localhost:6379"
    $env:OLLAMA_URL = "http://localhost:11434"
    $env:MAX_CONCURRENCY = "3"
    
    if ($GPUMode) {
        $env:GPU_MEMORY_LIMIT_MB = "6000"
        $env:ENABLE_GPU = "true"
    }
    
    # Build and start Go service
    Push-Location ..
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "go run main.go" -WindowStyle Minimized
    Pop-Location
    
    Start-Sleep -Seconds 2
    
    # Check if service is running
    try {
        $goHealth = Invoke-RestMethod -Uri "http://localhost:8084/api/health" -Method Get -TimeoutSec 2
        Write-Status "Go microservice started on port 8084"
    } catch {
        Write-Warning "Go microservice failed to start"
    }
} else {
    Write-Warning "Go microservice not found - backend features disabled"
}

# Run pre-flight checks unless skipped
if (-not $SkipChecks) {
    Write-Header "RUNNING PRE-FLIGHT CHECKS"
    
    Write-Info "TypeScript check..."
    $tsCheck = npx tsc --noEmit --skipLibCheck 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Status "TypeScript: No errors"
    } else {
        Write-Warning "TypeScript: Errors found (non-blocking)"
        if ($Debug) {
            Write-Host $tsCheck -ForegroundColor Gray
        }
    }
    
    Write-Info "Dependencies check..."
    $outdated = npm outdated --json 2>$null | ConvertFrom-Json
    if ($outdated.PSObject.Properties.Count -gt 0) {
        Write-Warning "$($outdated.PSObject.Properties.Count) packages outdated"
    } else {
        Write-Status "All packages up to date"
    }
}

# Set development environment variables
$env:NODE_ENV = "development"
$env:NODE_OPTIONS = "--max-old-space-size=4096"
$env:VITE_LEGAL_AI_API = "http://localhost:8084"
$env:VITE_OLLAMA_URL = "http://localhost:11434"
$env:VITE_REDIS_URL = "redis://localhost:6379"

Write-Header "STARTING DEVELOPMENT SERVER"

# Create a monitoring script that runs in parallel
$monitorScript = @'
while ($true) {
    Clear-Host
    Write-Host "📊 SYSTEM MONITOR" -ForegroundColor Cyan
    Write-Host "=================" -ForegroundColor Cyan
    
    # Check services
    $services = @{
        "Frontend" = 5173
        "Go API" = 8084
        "Redis" = 6379
        "Ollama" = 11434
    }
    
    foreach ($service in $services.GetEnumerator()) {
        $port = $service.Value
        try {
            $conn = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue -InformationLevel Quiet
            if ($conn) {
                Write-Host "✅ $($service.Key): Port $port" -ForegroundColor Green
            } else {
                Write-Host "❌ $($service.Key): Port $port" -ForegroundColor Red
            }
        } catch {
            Write-Host "❌ $($service.Key): Port $port" -ForegroundColor Red
        }
    }
    
    # Memory usage
    $nodeProc = Get-Process node -ErrorAction SilentlyContinue
    if ($nodeProc) {
        $memMB = [math]::Round($nodeProc.WorkingSet64 / 1MB, 0)
        Write-Host "`n💾 Node Memory: ${memMB}MB" -ForegroundColor Yellow
    }
    
    # GPU usage if available
    if ($env:ENABLE_GPU -eq "true") {
        $gpuInfo = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0) {
            $gpu = $gpuInfo -split ','
            Write-Host "🎮 GPU: $($gpu[0])% | VRAM: $($gpu[1])MB / $($gpu[2])MB" -ForegroundColor Magenta
        }
    }
    
    Write-Host "`nPress Ctrl+C to stop monitoring" -ForegroundColor Gray
    Start-Sleep -Seconds 5
}
'@

# Start monitoring in a separate window if not in debug mode
if (-not $Debug) {
    $monitorPath = Join-Path $env:TEMP "monitor-dev.ps1"
    $monitorScript | Out-File -FilePath $monitorPath -Encoding UTF8
    Start-Process powershell -ArgumentList "-NoExit", "-File", $monitorPath -WindowStyle Normal
}

Write-Info "Starting Vite development server..."
Write-Host ""
Write-Status "Development environment ready!"
Write-Host ""
Write-Host "🌐 Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🚀 API: http://localhost:8084" -ForegroundColor Green
Write-Host "📊 UnoCSS: http://localhost:5173/__unocss/" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Gray
Write-Host ""

# Start Vite with error handling
try {
    npm run dev
} catch {
    Write-Error "Vite server crashed"
    Write-Host $_.Exception.Message -ForegroundColor Red
} finally {
    # Cleanup on exit
    Write-Warning "`nShutting down services..."
    
    # Stop processes
    Get-Process node, go, redis-server, ollama -ErrorAction SilentlyContinue | Stop-Process -Force
    
    Write-Status "Development environment stopped"
}
