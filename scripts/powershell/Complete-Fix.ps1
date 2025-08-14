# Complete System Fix PowerShell Script

Write-Host @"
=====================================
🚀 LEGAL AI SYSTEM - COMPLETE FIX
=====================================
"@ -ForegroundColor Cyan

# Function to test command availability
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Step 1: Database Check
Write-Host "`n[Step 1/6] Checking Database..." -ForegroundColor Yellow
Write-Host "---------------------------------" -ForegroundColor Gray

$env:PGPASSWORD = "123456"
$psqlPath = "C:\Program Files\PostgreSQL\17\bin\psql.exe"

if (Test-Path $psqlPath) {
    $result = & $psqlPath -U legal_admin -h localhost -d legal_ai_db -t -c "SELECT 'OK'" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Database: Connected" -ForegroundColor Green
    } else {
        Write-Host "❌ Database: Failed" -ForegroundColor Red
        Write-Host "   Fix: Run FIX-POSTGRES-ADMIN.bat as Administrator" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "⚠️ PostgreSQL not found at expected path" -ForegroundColor Yellow
}

# Step 2: Redis Setup
Write-Host "`n[Step 2/6] Setting up Redis..." -ForegroundColor Yellow
Write-Host "---------------------------------" -ForegroundColor Gray

# Check if Redis is running
$redisRunning = $false

if (Test-Command "redis-cli") {
    $redisPing = redis-cli ping 2>&1
    if ($redisPing -eq "PONG") {
        $redisRunning = $true
    }
}

if (-not $redisRunning -and (Test-Path "redis-windows\redis-cli.exe")) {
    $redisPing = .\redis-windows\redis-cli.exe ping 2>&1
    if ($redisPing -eq "PONG") {
        $redisRunning = $true
    }
}

if (-not $redisRunning) {
    # Try to start Redis
    if (Test-Path "redis-windows\redis-server.exe") {
        Write-Host "Starting Redis from redis-windows..." -ForegroundColor Cyan
        
        # Create config if missing
        if (-not (Test-Path "redis-windows\redis.conf")) {
            @"
port 6379
bind 127.0.0.1
protected-mode no
daemonize no
dir ./
logfile redis.log
"@ | Out-File -FilePath "redis-windows\redis.conf" -Encoding ASCII
        }
        
        Start-Process -FilePath "redis-windows\redis-server.exe" `
                     -ArgumentList "redis.conf" `
                     -WorkingDirectory "redis-windows" `
                     -WindowStyle Hidden
        
        Start-Sleep -Seconds 3
        
        # Check again
        $redisPing = .\redis-windows\redis-cli.exe ping 2>&1
        if ($redisPing -eq "PONG") {
            $redisRunning = $true
        }
    }
}

if ($redisRunning) {
    Write-Host "✅ Redis: Running" -ForegroundColor Green
} else {
    Write-Host "⚠️ Redis: Not running (services will work without caching)" -ForegroundColor Yellow
}

# Step 3: Create Directories
Write-Host "`n[Step 3/6] Creating Directories..." -ForegroundColor Yellow
Write-Host "---------------------------------" -ForegroundColor Gray

$directories = @("logs", "uploads", "documents", "evidence", "generated_reports")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Green
    }
}

# Step 4: Build Go Services
Write-Host "`n[Step 4/6] Building Go Services..." -ForegroundColor Yellow
Write-Host "---------------------------------" -ForegroundColor Gray

Set-Location "go-microservice"

# Set Go environment
$env:GO111MODULE = "on"
$env:CGO_ENABLED = "1"

# Install dependencies
Write-Host "Installing Go dependencies..." -ForegroundColor Cyan
go get -u github.com/gin-gonic/gin 2>$null
go get -u github.com/redis/go-redis/v9 2>$null
go get -u github.com/jackc/pgx/v5/pgxpool 2>$null
go mod tidy 2>$null

# Kill existing processes
Get-Process | Where-Object {$_.Name -match "legal-processor"} | Stop-Process -Force -ErrorAction SilentlyContinue

# Build simple processor (fallback)
Write-Host "Building simple processor..." -ForegroundColor Cyan
$buildResult = go build -o legal-processor-simple.exe legal-processor-simple.go 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Simple processor built" -ForegroundColor Green
    $processor = "legal-processor-simple.exe"
} else {
    Write-Host "❌ Build failed" -ForegroundColor Red
    Write-Host $buildResult
}

Set-Location ..

# Step 5: Start Services
Write-Host "`n[Step 5/6] Starting Services..." -ForegroundColor Yellow
Write-Host "---------------------------------" -ForegroundColor Gray

if ($processor) {
    Set-Location "go-microservice"
    Write-Host "Starting $processor..." -ForegroundColor Cyan
    Start-Process -FilePath $processor -WindowStyle Hidden
    Write-Host "✅ Processor started" -ForegroundColor Green
    Set-Location ..
}

# Wait for services
Start-Sleep -Seconds 5

# Step 6: Health Check
Write-Host "`n[Step 6/6] System Health Check..." -ForegroundColor Yellow
Write-Host "---------------------------------" -ForegroundColor Gray

# Check API Service
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8080/health" -Method Get -TimeoutSec 5
    Write-Host "✅ API Service: ONLINE" -ForegroundColor Green
    Write-Host "   Status: $($health.status)" -ForegroundColor Gray
    Write-Host "   Redis: $($health.redis_connected)" -ForegroundColor Gray
    Write-Host "   Database: $($health.db_connected)" -ForegroundColor Gray
} catch {
    Write-Host "❌ API Service: OFFLINE" -ForegroundColor Red
}

Write-Host @"

=====================================
📊 SYSTEM STATUS COMPLETE
=====================================

Next Steps:
-----------
1. Run: node check-system-integration.mjs
2. Start frontend: npm run dev
3. Access: http://localhost:5173

API Endpoints:
-------------
Health: http://localhost:8080/health
Metrics: http://localhost:8080/metrics

"@ -ForegroundColor Cyan

Read-Host "Press Enter to continue"
