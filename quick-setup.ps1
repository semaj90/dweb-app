#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Quick setup script for the Legal AI Assistant Platform

.DESCRIPTION
    This script sets up the complete development environment including:
    - Database initialization
    - Python NLP service setup
    - Frontend dependencies
    - Environment configuration

.PARAMETER Environment
    Target environment: dev, test, or prod

.PARAMETER SkipPython
    Skip Python service setup

.PARAMETER SkipDatabase
    Skip database setup

.PARAMETER StartServices
    Start all services after setup

.EXAMPLE
    .\quick-setup.ps1 -Environment dev -StartServices
#>

param(
    [Parameter()]
    [ValidateSet("dev", "test", "prod")]
    [string]$Environment = "dev",
    
    [switch]$SkipPython,
    [switch]$SkipDatabase,
    [switch]$StartServices
)

Write-Host "🚀 Legal AI Assistant Platform - Quick Setup" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$FrontendPath = Join-Path $ProjectRoot "web-app\sveltekit-frontend"
$PythonServicePath = Join-Path $ProjectRoot "python-masking-service"

# Function to check if command exists
function Test-Command {
    param($Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Blue

$Prerequisites = @(
    @{ Name = "Node.js"; Command = "node"; Version = "--version" },
    @{ Name = "npm"; Command = "npm"; Version = "--version" },
    @{ Name = "Python"; Command = "python"; Version = "--version" },
    @{ Name = "pip"; Command = "pip"; Version = "--version" }
)

foreach ($prereq in $Prerequisites) {
    if (Test-Command $prereq.Command) {
        $version = & $prereq.Command $prereq.Version 2>$null
        Write-Host "  ✅ $($prereq.Name): $version" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $($prereq.Name) not found" -ForegroundColor Red
        throw "$($prereq.Name) is required but not installed"
    }
}

# Setup Frontend
Write-Host "📦 Setting up frontend..." -ForegroundColor Blue
Set-Location $FrontendPath

if (!(Test-Path "node_modules")) {
    Write-Host "  Installing npm dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
} else {
    Write-Host "  ✅ Dependencies already installed" -ForegroundColor Green
}

# Setup environment file
$EnvFile = ".env.$Environment"
$TargetEnv = ".env"

if (Test-Path $EnvFile) {
    Write-Host "  Copying environment file: $EnvFile -> $TargetEnv" -ForegroundColor Yellow
    Copy-Item $EnvFile $TargetEnv -Force
} else {
    Write-Host "  Creating default environment file..." -ForegroundColor Yellow
    @"
# Legal AI Assistant Platform - $Environment Environment
DATABASE_URL=postgresql://postgres:password@localhost:5432/legal_ai_db
REDIS_URL=redis://localhost:6379
PYTHON_NLP_URL=http://localhost:8001
OLLAMA_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333
AUTH_SECRET=your-secret-key-change-in-production
PUBLIC_APP_NAME=Legal AI Assistant
PUBLIC_APP_VERSION=1.0.0
NODE_ENV=$Environment
"@ | Out-File -FilePath $TargetEnv -Encoding utf8
}

# Setup Database
if (!$SkipDatabase) {
    Write-Host "🗄️ Setting up database..." -ForegroundColor Blue
    
    # Check if Docker is available
    if (Test-Command "docker") {
        Write-Host "  Starting database services with Docker..." -ForegroundColor Yellow
        try {
            npm run db:start
            Start-Sleep -Seconds 5
            
            Write-Host "  Running database migrations..." -ForegroundColor Yellow
            npm run db:migrate
            
            Write-Host "  ✅ Database setup complete" -ForegroundColor Green
        } catch {
            Write-Host "  ⚠️ Database setup failed, please ensure PostgreSQL and Redis are running" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ⚠️ Docker not found. Please ensure PostgreSQL and Redis are running manually" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⏭️ Skipping database setup" -ForegroundColor Yellow
}

# Setup Python NLP Service
if (!$SkipPython) {
    Write-Host "🐍 Setting up Python NLP service..." -ForegroundColor Blue
    Set-Location $PythonServicePath
    
    # Check if virtual environment exists
    if (!(Test-Path "venv")) {
        Write-Host "  Creating Python virtual environment..." -ForegroundColor Yellow
        python -m venv venv
    }
    
    # Activate virtual environment
    $VenvActivate = "venv\Scripts\Activate.ps1"
    if (Test-Path $VenvActivate) {
        Write-Host "  Activating virtual environment..." -ForegroundColor Yellow
        & $VenvActivate
        
        Write-Host "  Installing Python dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
        
        Write-Host "  ✅ Python service setup complete" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Failed to create virtual environment" -ForegroundColor Red
    }
} else {
    Write-Host "  ⏭️ Skipping Python service setup" -ForegroundColor Yellow
}

# Create startup scripts
Write-Host "📝 Creating startup scripts..." -ForegroundColor Blue
Set-Location $ProjectRoot

# Frontend startup script
@"
#!/usr/bin/env pwsh
Write-Host "🚀 Starting Legal AI Assistant Frontend..." -ForegroundColor Cyan
Set-Location "$FrontendPath"
npm run dev
"@ | Out-File -FilePath "start-frontend.ps1" -Encoding utf8

# Python service startup script
@"
#!/usr/bin/env pwsh
Write-Host "🐍 Starting Python NLP Service..." -ForegroundColor Cyan
Set-Location "$PythonServicePath"
& venv\Scripts\Activate.ps1
python main.py
"@ | Out-File -FilePath "start-python-service.ps1" -Encoding utf8

# Combined startup script
@"
#!/usr/bin/env pwsh
Write-Host "🚀 Starting All Legal AI Assistant Services..." -ForegroundColor Cyan

# Start Python service in background
Write-Host "Starting Python NLP service..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-File", "$ProjectRoot\start-python-service.ps1" -WindowStyle Normal

# Wait a moment for Python service to start
Start-Sleep -Seconds 3

# Start frontend (blocking)
Write-Host "Starting frontend..." -ForegroundColor Yellow
& "$ProjectRoot\start-frontend.ps1"
"@ | Out-File -FilePath "start-all.ps1" -Encoding utf8

Write-Host "  ✅ Startup scripts created" -ForegroundColor Green

# Start services if requested
if ($StartServices) {
    Write-Host "🚀 Starting all services..." -ForegroundColor Blue
    & "$ProjectRoot\start-all.ps1"
} else {
    Write-Host "🎉 Setup complete! Next steps:" -ForegroundColor Green
    Write-Host "  1. Start all services: .\start-all.ps1" -ForegroundColor White
    Write-Host "  2. Or start individually:" -ForegroundColor White
    Write-Host "     - Frontend: .\start-frontend.ps1" -ForegroundColor White
    Write-Host "     - Python Service: .\start-python-service.ps1" -ForegroundColor White
    Write-Host "  3. Access the application at: http://localhost:5173" -ForegroundColor White
    Write-Host "  4. Python NLP service runs on: http://localhost:8001" -ForegroundColor White
}

Write-Host "`n✨ Legal AI Assistant Platform is ready!" -ForegroundColor Cyan
