# Complete PostgreSQL + GPU System Setup PowerShell Script

Write-Host "🚀 Legal AI PostgreSQL + GPU Setup" -ForegroundColor Cyan
Write-Host ("=" * 50) -ForegroundColor Cyan
Write-Host ""

# Function to test PostgreSQL connection
function Test-PostgreSQLConnection {
    Write-Host "🔍 Testing PostgreSQL connection..." -ForegroundColor Yellow
    
    $env:PGPASSWORD = "123456"
    $testQuery = "SELECT current_user, current_database()"
    
    try {
        $result = & "C:\Program Files\PostgreSQL\17\bin\psql.exe" `
            -U legal_admin `
            -h localhost `
            -d legal_ai_db `
            -t -c $testQuery 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Database connection successful" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Database connection failed" -ForegroundColor Red
            Write-Host "   Error: $result" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ Failed to execute psql: $_" -ForegroundColor Red
        return $false
    }
}

# Function to setup database schema
function Setup-DatabaseSchema {
    Write-Host "`n📊 Setting up database schema..." -ForegroundColor Yellow
    
    Set-Location -Path "sveltekit-frontend"
    
    # Run schema setup
    $result = node setup-postgres-gpu.mjs --seed 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Database schema created successfully" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Schema setup failed: $result" -ForegroundColor Red
        return $false
    }
}

# Function to build Go services
function Build-GoServices {
    Write-Host "`n🔨 Building Go GPU services..." -ForegroundColor Yellow
    
    Set-Location -Path "..\go-microservice"
    
    # Set CGO environment
    $env:CGO_ENABLED = "1"
    $env:CC = "C:\Progra~1\LLVM\bin\clang.exe"
    $env:CXX = "C:\Progra~1\LLVM\bin\clang++.exe"
    
    # Install dependencies
    Write-Host "   Installing Go dependencies..." -ForegroundColor Cyan
    go get -u github.com/gin-gonic/gin | Out-Null
    go get -u github.com/redis/go-redis/v9 | Out-Null
    go get -u github.com/jackc/pgx/v5/pgxpool | Out-Null
    go mod tidy | Out-Null
    
    # Build GPU processor
    Write-Host "   Building GPU processor..." -ForegroundColor Cyan
    go build -tags=cgo -o legal-processor-gpu.exe legal_processor_gpu_simd.go
    
    if (Test-Path "legal-processor-gpu.exe") {
        Write-Host "✅ GPU processor built successfully" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ GPU processor build failed" -ForegroundColor Red
        return $false
    }
}

# Function to start services
function Start-Services {
    Write-Host "`n🚀 Starting services..." -ForegroundColor Yellow
    
    # Check Redis
    $redisService = Get-Service -Name "Redis" -ErrorAction SilentlyContinue
    if ($redisService) {
        if ($redisService.Status -ne "Running") {
            Start-Service -Name "Redis"
            Write-Host "✅ Redis service started" -ForegroundColor Green
        } else {
            Write-Host "✅ Redis already running" -ForegroundColor Green
        }
    } else {
        # Start Redis manually
        $redisPath = "redis-windows\redis-server.exe"
        if (Test-Path $redisPath) {
            Start-Process -FilePath $redisPath -ArgumentList "redis-windows\redis.conf" -WindowStyle Hidden
            Write-Host "✅ Redis started manually" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Redis not found" -ForegroundColor Yellow
        }
    }
    
    # Start GPU processor
    $gpuPath = "go-microservice\legal-processor-gpu.exe"
    if (Test-Path $gpuPath) {
        Start-Process -FilePath $gpuPath -WindowStyle Hidden
        Write-Host "✅ GPU processor started" -ForegroundColor Green
    } else {
        Write-Host "⚠️ GPU processor not found" -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds 3
}

# Function to check system health
function Test-SystemHealth {
    Write-Host "`n🔍 Running health checks..." -ForegroundColor Yellow
    
    # Test GPU service
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8080/health" -Method Get -TimeoutSec 5
        Write-Host "✅ GPU Service: ONLINE" -ForegroundColor Green
        if ($response.gpu_enabled) {
            Write-Host "   GPU Acceleration: ENABLED" -ForegroundColor Green
        }
    } catch {
        Write-Host "❌ GPU Service: OFFLINE" -ForegroundColor Red
    }
    
    # Test Redis
    try {
        $redisPing = & redis-cli ping 2>&1
        if ($redisPing -eq "PONG") {
            Write-Host "✅ Redis: ONLINE" -ForegroundColor Green
        } else {
            Write-Host "❌ Redis: OFFLINE" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Redis check failed" -ForegroundColor Red
    }
}

# Main execution
try {
    # Save current location
    $originalLocation = Get-Location
    
    # Test PostgreSQL connection
    if (-not (Test-PostgreSQLConnection)) {
        Write-Host "`n⚠️ Database connection failed!" -ForegroundColor Red
        Write-Host "Please run FIX-POSTGRES-ADMIN.bat as Administrator first" -ForegroundColor Yellow
        exit 1
    }
    
    # Setup database schema
    if (-not (Setup-DatabaseSchema)) {
        Write-Host "`n⚠️ Schema setup failed!" -ForegroundColor Red
        exit 1
    }
    
    # Build Go services
    Set-Location $originalLocation
    if (-not (Build-GoServices)) {
        Write-Host "`n⚠️ Service build failed!" -ForegroundColor Red
        exit 1
    }
    
    # Start services
    Set-Location $originalLocation
    Start-Services
    
    # Health check
    Test-SystemHealth
    
    # Success message
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 50) -ForegroundColor Green
    Write-Host "✅ System Setup Complete!" -ForegroundColor Green
    Write-Host ("=" * 50) -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Connection Details:" -ForegroundColor Cyan
    Write-Host "   PostgreSQL: legal_ai_db (legal_admin/123456)"
    Write-Host "   GPU Service: http://localhost:8080"
    Write-Host "   Redis: localhost:6379"
    Write-Host ""
    Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
    Write-Host "   1. Start frontend: cd sveltekit-frontend && npm run dev"
    Write-Host "   2. Access UI: http://localhost:5173"
    Write-Host ""
    
} catch {
    Write-Host "❌ Setup failed: $_" -ForegroundColor Red
    exit 1
} finally {
    Set-Location $originalLocation
}
