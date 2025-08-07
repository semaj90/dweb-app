# START-SIMD-REDIS-VITE.ps1
# PowerShell script for SIMD + Redis + Vite Integration System
# Enhanced with monitoring and error handling

param(
    [switch]$SkipTests = $false,
    [switch]$Production = $false,
    [switch]$Debug = $false,
    [int]$Workers = 0,
    [string]$RedisPath = ""
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Banner
Clear-Host
Write-Host @"
================================================================
     SIMD JSON + Redis + Vite Integration System
     PowerShell Enhanced Launcher v1.0
================================================================
"@ -ForegroundColor Magenta

# Configuration
$config = @{
    RedisPort = 6379
    SIMDPort = 8080
    VitePort = 3130
    Workers = if ($Workers -gt 0) { $Workers } else { [Environment]::ProcessorCount * 2 }
    Mode = if ($Production) { "production" } else { "development" }
}

Write-Info "`nConfiguration:"
Write-Host "  Mode: $($config.Mode)"
Write-Host "  Workers: $($config.Workers)"
Write-Host "  Redis Port: $($config.RedisPort)"
Write-Host "  SIMD Server Port: $($config.SIMDPort)"
Write-Host "  Vite Dev Port: $($config.VitePort)"

# Check prerequisites
Write-Info "`n[1/4] Checking prerequisites..."

# Check Redis
$redisInstalled = $false
$redisExe = ""

if (Get-Command redis-server -ErrorAction SilentlyContinue) {
    $redisExe = "redis-server"
    $redisInstalled = $true
    Write-Success "  ✓ Redis found in PATH"
} elseif ($RedisPath -and (Test-Path "$RedisPath\redis-server.exe")) {
    $redisExe = "$RedisPath\redis-server.exe"
    $redisInstalled = $true
    Write-Success "  ✓ Redis found at $RedisPath"
} elseif (Test-Path ".\redis-windows\redis-server.exe") {
    $redisExe = ".\redis-windows\redis-server.exe"
    $redisInstalled = $true
    Write-Success "  ✓ Redis found in local directory"
} else {
    Write-Error "  ✗ Redis not found. Please install Redis or specify path with -RedisPath"
    Write-Host "    Download from: https://github.com/microsoftarchive/redis/releases"
    exit 1
}

# Check Go
if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Error "  ✗ Go not found. Please install Go from https://golang.org/dl/"
    exit 1
}
Write-Success "  ✓ Go installed: $(go version)"

# Check Node.js
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "  ✗ Node.js not found. Please install from https://nodejs.org/"
    exit 1
}
Write-Success "  ✓ Node.js installed: $(node --version)"

# Check npm packages
Write-Info "`n[2/4] Checking npm packages..."
if (-not (Test-Path "node_modules")) {
    Write-Warning "  Installing npm dependencies..."
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Error "  ✗ Failed to install npm dependencies"
        exit 1
    }
}
Write-Success "  ✓ NPM packages ready"

# Function to start process with logging
function Start-ServiceProcess {
    param(
        [string]$Name,
        [string]$Executable,
        [string]$Arguments = "",
        [string]$WorkingDirectory = "."
    )
    
    $logFile = "logs\$Name-$(Get-Date -Format 'yyyy-MM-dd').log"
    $null = New-Item -ItemType Directory -Path "logs" -Force -ErrorAction SilentlyContinue
    
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Executable
    $psi.Arguments = $Arguments
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $false
    
    if ($Debug) {
        $psi.Environment["DEBUG"] = "*"
    }
    
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    
    # Set up output handlers
    $outputHandler = {
        if ($EventArgs.Data) {
            Add-Content -Path $logFile -Value "[$(Get-Date -Format 'HH:mm:ss')] $($EventArgs.Data)"
            if ($Debug) {
                Write-Host "[$Name] $($EventArgs.Data)" -ForegroundColor Gray
            }
        }
    }
    
    Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outputHandler | Out-Null
    Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $outputHandler | Out-Null
    
    $process.Start() | Out-Null
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
    
    return $process
}

# Start services
Write-Info "`n[3/4] Starting services..."

# Start Redis
Write-Info "  Starting Redis Server..."
$redisProcess = Start-ServiceProcess -Name "Redis" -Executable $redisExe

Start-Sleep -Seconds 2

# Test Redis connection
try {
    $redisCli = if ($RedisPath) { "$RedisPath\redis-cli.exe" } else { "redis-cli" }
    $pingResult = & $redisCli ping 2>$null
    if ($pingResult -eq "PONG") {
        Write-Success "  ✓ Redis is running"
    } else {
        throw "Redis not responding"
    }
} catch {
    Write-Error "  ✗ Redis failed to start"
    exit 1
}

# Check Redis JSON module
Write-Info "  Checking Redis JSON module..."
$redisJsonAvailable = $false
try {
    $jsonCheck = & $redisCli --raw JSON.GET test_key 2>$null
    if ($jsonCheck -notmatch "ERR unknown command") {
        $redisJsonAvailable = $true
        Write-Success "  ✓ Redis JSON module available"
    } else {
        Write-Warning "  ⚠ Redis JSON module not available (optional)"
    }
} catch {
    Write-Warning "  ⚠ Could not check Redis JSON module"
}

# Build and start Go SIMD server
Write-Info "  Building Go SIMD server..."
Push-Location go-microservice

# Check if source file exists
if (-not (Test-Path "simd-redis-vite-server.go")) {
    Write-Error "  ✗ simd-redis-vite-server.go not found"
    Write-Host "    Please ensure the file exists in go-microservice/"
    Pop-Location
    exit 1
}

# Install Go dependencies
Write-Info "  Installing Go dependencies..."
go mod tidy
if ($LASTEXITCODE -ne 0) {
    Write-Warning "  Initializing Go module..."
    go mod init simd-server
    go get github.com/gin-gonic/gin
    go get github.com/gin-contrib/cors
    go get github.com/go-redis/redis/v8
    go get github.com/valyala/fastjson
    go get github.com/gorilla/websocket
    go mod tidy
}

# Build the server
Write-Info "  Compiling SIMD server..."
$buildArgs = if ($Production) {
    "-ldflags=`"-s -w`" -o simd-redis-vite.exe simd-redis-vite-server.go"
} else {
    "-o simd-redis-vite.exe simd-redis-vite-server.go"
}

Invoke-Expression "go build $buildArgs"
if ($LASTEXITCODE -ne 0) {
    Write-Error "  ✗ Failed to compile Go server"
    Pop-Location
    exit 1
}
Write-Success "  ✓ SIMD server compiled"

# Set environment variables for SIMD server
$env:SIMD_WORKERS = $config.Workers
$env:REDIS_HOST = "localhost"
$env:REDIS_PORT = $config.RedisPort

# Start SIMD server
Write-Info "  Starting SIMD server..."
$simdProcess = Start-ServiceProcess -Name "SIMD" -Executable ".\simd-redis-vite.exe" -WorkingDirectory (Get-Location)

Pop-Location

Start-Sleep -Seconds 3

# Test SIMD server health
Write-Info "  Testing SIMD server health..."
$maxRetries = 5
$retryCount = 0
$simdHealthy = $false

while ($retryCount -lt $maxRetries -and -not $simdHealthy) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:$($config.SIMDPort)/health" -Method Get -TimeoutSec 2
        if ($response.status -eq "healthy") {
            $simdHealthy = $true
            Write-Success "  ✓ SIMD server is healthy"
            Write-Host "    Workers: $($response.workers)"
            Write-Host "    Redis: $($response.redis)"
            Write-Host "    Redis JSON: $($response.redis_json)"
        }
    } catch {
        $retryCount++
        if ($retryCount -lt $maxRetries) {
            Write-Warning "  Retry $retryCount/$maxRetries - Waiting for server..."
            Start-Sleep -Seconds 2
        }
    }
}

if (-not $simdHealthy) {
    Write-Error "  ✗ SIMD server failed to start"
    exit 1
}

# Start Vite development server
Write-Info "`n[4/4] Starting Vite development server..."
if ($Production) {
    Write-Info "  Building production bundle..."
    npm run build
    $viteProcess = Start-ServiceProcess -Name "Vite" -Executable "npm" -Arguments "run preview"
} else {
    $viteProcess = Start-ServiceProcess -Name "Vite" -Executable "npm" -Arguments "run dev"
}

Start-Sleep -Seconds 5

# Display status
Write-Success "`n================================================================"
Write-Success "              System Successfully Started!"
Write-Success "================================================================"
Write-Host ""
Write-Info "Services Running:"
Write-Host "  [Redis]      http://localhost:$($config.RedisPort)" -ForegroundColor Green
Write-Host "  [SIMD]       http://localhost:$($config.SIMDPort)" -ForegroundColor Green
Write-Host "  [Vite]       http://localhost:$($config.VitePort)" -ForegroundColor Green
Write-Host ""
Write-Info "Available Endpoints:"
Write-Host "  Health Check    : GET  http://localhost:$($config.SIMDPort)/health"
Write-Host "  SIMD Parse      : POST http://localhost:$($config.SIMDPort)/simd-parse"
Write-Host "  Batch Process   : POST http://localhost:$($config.SIMDPort)/simd-batch"
Write-Host "  Document Process: POST http://localhost:$($config.SIMDPort)/process-document"
Write-Host "  Legal Analysis  : POST http://localhost:$($config.SIMDPort)/legal/analyze"
Write-Host "  Metrics         : GET  http://localhost:$($config.SIMDPort)/metrics"
Write-Host "  WebSocket       : WS   ws://localhost:$($config.SIMDPort)/ws"
Write-Host ""
Write-Info "Performance:"
Write-Host "  Workers         : $($config.Workers)"
Write-Host "  Mode            : $($config.Mode)"
Write-Host "  Redis JSON      : $(if ($redisJsonAvailable) { 'Enabled' } else { 'Disabled' })"
Write-Host ""

# Run tests if not skipped
if (-not $SkipTests -and (Test-Path "test-simd-redis-vite.mjs")) {
    Write-Info "Running integration tests..."
    $testResult = node test-simd-redis-vite.mjs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✓ All tests passed!"
    } else {
        Write-Warning "⚠ Some tests failed. Check the output above."
    }
}

# Monitor processes
Write-Info "`nMonitoring services (Press Ctrl+C to stop all)..."
Write-Host "Logs are being written to the 'logs' directory" -ForegroundColor Gray

# Set up cleanup
$cleanupBlock = {
    Write-Warning "`nShutting down services..."
    
    if ($redisProcess -and -not $redisProcess.HasExited) {
        $redisProcess.Kill()
        Write-Host "  Redis stopped" -ForegroundColor Yellow
    }
    
    if ($simdProcess -and -not $simdProcess.HasExited) {
        $simdProcess.Kill()
        Write-Host "  SIMD server stopped" -ForegroundColor Yellow
    }
    
    if ($viteProcess -and -not $viteProcess.HasExited) {
        $viteProcess.Kill()
        Write-Host "  Vite server stopped" -ForegroundColor Yellow
    }
    
    Write-Success "All services stopped."
}

# Register cleanup on Ctrl+C
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action $cleanupBlock | Out-Null

try {
    # Keep monitoring
    while ($true) {
        # Check if processes are still running
        $allRunning = $true
        
        if ($redisProcess.HasExited) {
            Write-Error "Redis has stopped unexpectedly!"
            $allRunning = $false
        }
        
        if ($simdProcess.HasExited) {
            Write-Error "SIMD server has stopped unexpectedly!"
            $allRunning = $false
        }
        
        if ($viteProcess.HasExited) {
            Write-Error "Vite server has stopped unexpectedly!"
            $allRunning = $false
        }
        
        if (-not $allRunning) {
            Write-Error "One or more services have stopped. Shutting down..."
            & $cleanupBlock
            exit 1
        }
        
        Start-Sleep -Seconds 5
    }
} finally {
    & $cleanupBlock
}
