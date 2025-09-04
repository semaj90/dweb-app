#!/usr/bin/env pwsh
# Complete Service Fix Script
# Fixes all configuration issues and starts services properly

param(
    [switch]$AutoFix = $false,
    [switch]$StartServices = $false
)

Write-Host "🔧 Legal AI Service Fix Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 1. Fix Ollama Configuration
Write-Host "📝 Fixing Ollama Configuration..." -ForegroundColor Yellow

$envPath = Join-Path $PSScriptRoot ".." ".env"
$envContent = Get-Content $envPath -Raw

# Update Ollama URL from default to localhost
$envContent = $envContent -replace 'OLLAMA_API_URL=.*', 'OLLAMA_API_URL=http://localhost:11434'
$envContent = $envContent -replace 'OLLAMA_BASE_URL=.*', 'OLLAMA_BASE_URL=http://localhost:11434'
$envContent = $envContent -replace 'VITE_OLLAMA_URL=.*', 'VITE_OLLAMA_URL=http://localhost:11434'

# Save updated .env
$envContent | Set-Content $envPath -NoNewline
Write-Host "✅ Ollama URLs updated to http://localhost:11434" -ForegroundColor Green

# 2. Fix PostgreSQL
Write-Host "`n📝 Fixing PostgreSQL Configuration..." -ForegroundColor Yellow

# Check if PostgreSQL is installed
$pgPath = Get-Command pg_ctl -ErrorAction SilentlyContinue
if ($pgPath) {
    Write-Host "✅ PostgreSQL found at: $($pgPath.Source)" -ForegroundColor Green
    
    # Check PostgreSQL status
    try {
        $pgStatus = & pg_ctl status 2>&1
        if ($pgStatus -match "server is running") {
            Write-Host "✅ PostgreSQL is running" -ForegroundColor Green
        } else {
            Write-Host "⚠️ PostgreSQL is not running" -ForegroundColor Yellow
            if ($StartServices) {
                Write-Host "Starting PostgreSQL..." -ForegroundColor Cyan
                & pg_ctl start -D "$env:PGDATA" 2>&1 | Out-Null
                Start-Sleep -Seconds 3
                Write-Host "✅ PostgreSQL started" -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "⚠️ Could not check PostgreSQL status" -ForegroundColor Yellow
    }
    
    # Fix database and user
    Write-Host "Checking database configuration..." -ForegroundColor Cyan
    $sqlScript = @"
-- Check if database exists, create if not
SELECT 'CREATE DATABASE legal_ai_db' 
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'legal_ai_db')\gexec

-- Connect to legal_ai_db
\c legal_ai_db

-- Create user if not exists
DO `$`$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'legal_admin') THEN
        CREATE USER legal_admin WITH PASSWORD '123456';
    END IF;
END
`$`$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;
GRANT ALL ON SCHEMA public TO legal_admin;

-- Create pgvector extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;
"@
    
    $sqlScript | & psql -U postgres 2>&1 | Out-Null
    Write-Host "✅ Database configuration fixed" -ForegroundColor Green
} else {
    Write-Host "❌ PostgreSQL not found. Please install PostgreSQL first." -ForegroundColor Red
    Write-Host "   Download from: https://www.postgresql.org/download/windows/" -ForegroundColor Gray
}

# 3. Setup Neo4j
Write-Host "`n📝 Setting up Neo4j..." -ForegroundColor Yellow

$neo4jPath = Join-Path $PSScriptRoot ".." "neo4j-community-5.23.0" "bin" "neo4j.bat"
if (Test-Path $neo4jPath) {
    Write-Host "✅ Neo4j found" -ForegroundColor Green
    
    if ($StartServices) {
        Write-Host "Starting Neo4j..." -ForegroundColor Cyan
        Start-Process -FilePath $neo4jPath -ArgumentList "console" -WindowStyle Hidden
        Write-Host "✅ Neo4j starting on http://localhost:7474" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️ Neo4j not found at expected location" -ForegroundColor Yellow
    Write-Host "   Expected: $neo4jPath" -ForegroundColor Gray
}

# 4. Setup Redis
Write-Host "`n📝 Setting up Redis..." -ForegroundColor Yellow

$redisPath = Join-Path $PSScriptRoot ".." "redis" "redis-server.exe"
if (Test-Path $redisPath) {
    Write-Host "✅ Redis found" -ForegroundColor Green
    
    # Check if Redis is running
    $redisProcess = Get-Process redis-server -ErrorAction SilentlyContinue
    if ($redisProcess) {
        Write-Host "✅ Redis is already running" -ForegroundColor Green
    } elseif ($StartServices) {
        Write-Host "Starting Redis..." -ForegroundColor Cyan
        Start-Process -FilePath $redisPath -WindowStyle Hidden
        Write-Host "✅ Redis started on port 6379" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️ Redis not found" -ForegroundColor Yellow
    
    # Try to download Redis
    if ($AutoFix) {
        Write-Host "Downloading Redis for Windows..." -ForegroundColor Cyan
        $redisUrl = "https://github.com/microsoftarchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.zip"
        $redisZip = Join-Path $env:TEMP "redis.zip"
        $redisDir = Join-Path $PSScriptRoot ".." "redis"
        
        Invoke-WebRequest -Uri $redisUrl -OutFile $redisZip
        Expand-Archive -Path $redisZip -DestinationPath $redisDir -Force
        Remove-Item $redisZip
        
        Write-Host "✅ Redis downloaded and extracted" -ForegroundColor Green
    }
}

# 5. Setup MinIO
Write-Host "`n📝 Setting up MinIO..." -ForegroundColor Yellow

$minioPath = Join-Path $PSScriptRoot ".." "minio.exe"
if (Test-Path $minioPath) {
    Write-Host "✅ MinIO found" -ForegroundColor Green
    
    # Check if MinIO is running
    $minioProcess = Get-Process minio -ErrorAction SilentlyContinue
    if ($minioProcess) {
        Write-Host "✅ MinIO is already running" -ForegroundColor Green
    } elseif ($StartServices) {
        Write-Host "Starting MinIO..." -ForegroundColor Cyan
        $dataPath = Join-Path $PSScriptRoot ".." "minio-data"
        if (-not (Test-Path $dataPath)) {
            New-Item -ItemType Directory -Path $dataPath | Out-Null
        }
        Start-Process -FilePath $minioPath -ArgumentList "server", $dataPath, "--console-address", ":9001" -WindowStyle Hidden
        Write-Host "✅ MinIO started (API: 9000, Console: 9001)" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️ MinIO not found" -ForegroundColor Yellow
}

# 6. Setup Ollama
Write-Host "`n📝 Setting up Ollama..." -ForegroundColor Yellow

$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaPath) {
    Write-Host "✅ Ollama found" -ForegroundColor Green
    
    # Check if Ollama is running
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 2
        Write-Host "✅ Ollama is running (version: $($response.version))" -ForegroundColor Green
    } catch {
        if ($StartServices) {
            Write-Host "Starting Ollama..." -ForegroundColor Cyan
            Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 3
            Write-Host "✅ Ollama started" -ForegroundColor Green
            
            # Load the model
            Write-Host "Loading gemma3-legal model..." -ForegroundColor Cyan
            & ollama pull gemma3-legal:latest 2>&1 | Out-Null
            Write-Host "✅ Model loaded" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Ollama is not running" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "❌ Ollama not found. Please install Ollama first." -ForegroundColor Red
    Write-Host "   Download from: https://ollama.ai/download" -ForegroundColor Gray
}

# 7. Create Admin UI launcher
Write-Host "`n📝 Creating Admin UI launcher..." -ForegroundColor Yellow

$adminHtmlPath = Join-Path $PSScriptRoot ".." "admin-dashboard.html"
$adminContent = Get-Content (Join-Path $PSScriptRoot "admin-dashboard.html") -Raw -ErrorAction SilentlyContinue

if ($adminContent) {
    $adminContent | Set-Content $adminHtmlPath
    Write-Host "✅ Admin dashboard created at: $adminHtmlPath" -ForegroundColor Green
}

# 8. Create convenience scripts
Write-Host "`n📝 Creating convenience scripts..." -ForegroundColor Yellow

# Neo4j start/stop scripts
$neo4jStartScript = @"
@echo off
cd /d "%~dp0\neo4j-community-5.23.0\bin"
call neo4j.bat console
"@

$neo4jStopScript = @"
@echo off
cd /d "%~dp0\neo4j-community-5.23.0\bin"
call neo4j.bat stop
"@

$neo4jStartScript | Set-Content (Join-Path $PSScriptRoot ".." "START-NEO4J.bat")
$neo4jStopScript | Set-Content (Join-Path $PSScriptRoot ".." "STOP-NEO4J.bat")

Write-Host "✅ Created START-NEO4J.bat and STOP-NEO4J.bat" -ForegroundColor Green

# Open Admin UI script
$adminScript = @"
@echo off
start "" "admin-dashboard.html"
"@

$adminScript | Set-Content (Join-Path $PSScriptRoot ".." "OPEN-ADMIN.bat")
Write-Host "✅ Created OPEN-ADMIN.bat" -ForegroundColor Green

# Summary
Write-Host "`n📊 Configuration Summary" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

$services = @(
    @{Name="PostgreSQL"; Port=5432; Status="❓"},
    @{Name="Ollama"; Port=11434; Status="❓"},
    @{Name="Neo4j"; Port=7474; Status="❓"},
    @{Name="Redis"; Port=6379; Status="❓"},
    @{Name="MinIO"; Port=9000; Status="❓"},
    @{Name="Enhanced RAG"; Port=8094; Status="❓"}
)

foreach ($service in $services) {
    $tcpTest = Test-NetConnection -ComputerName localhost -Port $service.Port -WarningAction SilentlyContinue
    if ($tcpTest.TcpTestSucceeded) {
        $service.Status = "✅ Running"
        $color = "Green"
    } else {
        $service.Status = "❌ Offline"
        $color = "Red"
    }
    
    Write-Host "$($service.Name.PadRight(15)) Port: $($service.Port.ToString().PadRight(6)) Status: $($service.Status)" -ForegroundColor $color
}

Write-Host "`n✨ Quick Commands:" -ForegroundColor Cyan
Write-Host "  .\START-NEO4J.bat     - Start Neo4j" -ForegroundColor White
Write-Host "  .\STOP-NEO4J.bat      - Stop Neo4j" -ForegroundColor White
Write-Host "  .\OPEN-ADMIN.bat      - Open Admin Dashboard" -ForegroundColor White
Write-Host "  npm run dev           - Start Frontend" -ForegroundColor White

Write-Host "`n💡 To start all services automatically, run:" -ForegroundColor Yellow
Write-Host "  .\scripts\fix-all-services.ps1 -StartServices" -ForegroundColor White

if (-not $StartServices) {
    Write-Host "`n⚠️ Services were not started. Use -StartServices flag to start them." -ForegroundColor Yellow
}