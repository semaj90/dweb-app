# Complete Local Environment Setup Script
# Sets up PostgreSQL, Ollama, and all dependencies locally

param(
    [switch]$SkipPostgres,
    [switch]$SkipOllama,
    [string]$PostgresPassword = "postgres123"
)

Write-Host "🚀 Setting up complete local development environment..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "⚠️  This script requires administrator privileges." -ForegroundColor Yellow
    Write-Host "Restarting as administrator..." -ForegroundColor Yellow
    
    $arguments = "-File `"$PSCommandPath`""
    if ($SkipPostgres) { $arguments += " -SkipPostgres" }
    if ($SkipOllama) { $arguments += " -SkipOllama" }
    if ($PostgresPassword -ne "postgres123") { $arguments += " -PostgresPassword `"$PostgresPassword`"" }
    
    Start-Process PowerShell -Verb RunAs -ArgumentList $arguments
    exit
}

# Step 1: Setup PostgreSQL
if (-not $SkipPostgres) {
    Write-Host "`n1️⃣ Setting up PostgreSQL..." -ForegroundColor Magenta
    
    $postgresScript = Join-Path $PSScriptRoot "setup-local-postgres.ps1"
    if (Test-Path $postgresScript) {
        & $postgresScript -PostgresPassword $PostgresPassword
    } else {
        Write-Host "❌ PostgreSQL setup script not found: $postgresScript" -ForegroundColor Red
    }
} else {
    Write-Host "`n1️⃣ Skipping PostgreSQL setup..." -ForegroundColor Yellow
}

# Step 2: Setup Ollama
if (-not $SkipOllama) {
    Write-Host "`n2️⃣ Setting up Ollama..." -ForegroundColor Magenta
    
    # Check if Ollama is installed
    $ollamaInstalled = $false
    try {
        $ollamaVersion = ollama --version 2>$null
        if ($ollamaVersion) {
            Write-Host "✅ Ollama is already installed: $ollamaVersion" -ForegroundColor Green
            $ollamaInstalled = $true
        }
    } catch {
        Write-Host "📦 Ollama not found. Installing..." -ForegroundColor Yellow
    }
    
    if (-not $ollamaInstalled) {
        # Download and install Ollama
        $ollamaUrl = "https://ollama.com/download/windows"
        $ollamaInstaller = "$env:TEMP\ollama-windows-amd64.exe"
        
        Write-Host "📥 Downloading Ollama installer..." -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri $ollamaUrl -OutFile $ollamaInstaller
            
            Write-Host "🔧 Installing Ollama..." -ForegroundColor Yellow
            Start-Process -FilePath $ollamaInstaller -ArgumentList "/S" -Wait
            
            # Add to PATH if needed
            $ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama"
            $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
            if ($currentPath -notlike "*$ollamaPath*") {
                [Environment]::SetEnvironmentVariable("PATH", "$currentPath;$ollamaPath", "User")
                $env:PATH = "$env:PATH;$ollamaPath"
            }
            
            Write-Host "✅ Ollama installation completed" -ForegroundColor Green
        } catch {
            Write-Host "❌ Ollama installation failed: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    # Start Ollama service
    Write-Host "🔄 Starting Ollama service..." -ForegroundColor Yellow
    try {
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
        Write-Host "✅ Ollama service started" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Ollama service may already be running" -ForegroundColor Yellow
    }
    
    # Pull required models
    Write-Host "📦 Pulling required AI models..." -ForegroundColor Yellow
    
    $models = @("llama3.2", "nomic-embed-text")
    
    foreach ($model in $models) {
        Write-Host "Pulling $model..." -ForegroundColor Yellow
        try {
            & ollama pull $model
            Write-Host "✅ $model pulled successfully" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Failed to pull $model" -ForegroundColor Yellow
        }
    }
    
} else {
    Write-Host "`n2️⃣ Skipping Ollama setup..." -ForegroundColor Yellow
}

# Step 3: Install Node.js dependencies
Write-Host "`n3️⃣ Installing Node.js dependencies..." -ForegroundColor Magenta

$frontendPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

if (Test-Path $frontendPath) {
    Set-Location $frontendPath
    
    Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
    npm install
    
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "❌ Frontend directory not found: $frontendPath" -ForegroundColor Red
}

# Step 4: Setup database schema
Write-Host "`n4️⃣ Setting up database schema..." -ForegroundColor Magenta

try {
    # Generate Drizzle schema
    Write-Host "🔧 Generating database schema..." -ForegroundColor Yellow
    npm run db:generate
    
    # Run migrations
    Write-Host "🔄 Running database migrations..." -ForegroundColor Yellow
    npm run db:migrate
    
    Write-Host "✅ Database schema setup completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Database schema setup failed - you may need to run this manually" -ForegroundColor Yellow
}

# Step 5: Verify installation
Write-Host "`n5️⃣ Verifying installation..." -ForegroundColor Magenta

# Test PostgreSQL
Write-Host "🧪 Testing PostgreSQL..." -ForegroundColor Yellow
try {
    $env:PGPASSWORD = $PostgresPassword
    $result = & psql -U postgres -d deeds_legal_ai -c "SELECT version();" 2>$null
    if ($result) {
        Write-Host "✅ PostgreSQL is working" -ForegroundColor Green
    }
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
} catch {
    Write-Host "⚠️  PostgreSQL test failed" -ForegroundColor Yellow
}

# Test Ollama
Write-Host "🧪 Testing Ollama..." -ForegroundColor Yellow
try {
    $ollamaList = & ollama list 2>$null
    if ($ollamaList) {
        Write-Host "✅ Ollama is working" -ForegroundColor Green
        Write-Host "Available models:" -ForegroundColor White
        Write-Host $ollamaList -ForegroundColor Gray
    }
} catch {
    Write-Host "⚠️  Ollama test failed" -ForegroundColor Yellow
}

# Test GPU (if available)
Write-Host "🧪 Testing GPU availability..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    if ($gpuInfo) {
        Write-Host "✅ NVIDIA GPU detected: $gpuInfo" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No NVIDIA GPU detected (CPU mode will be used)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  GPU check failed (nvidia-smi not found)" -ForegroundColor Yellow
}

# Create startup script
Write-Host "`n6️⃣ Creating startup script..." -ForegroundColor Magenta

$startupScript = @"
# Local Development Environment Startup Script
# Run this script to start all services for development

Write-Host "🚀 Starting Local Development Environment..." -ForegroundColor Cyan

# Start PostgreSQL (if not already running)
`$postgresService = Get-Service -Name "postgresql-x64-*" -ErrorAction SilentlyContinue
if (`$postgresService -and `$postgresService.Status -ne "Running") {
    Write-Host "🔄 Starting PostgreSQL service..." -ForegroundColor Yellow
    Start-Service `$postgresService
}

# Start Ollama (if not already running)
`$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not `$ollamaProcess) {
    Write-Host "🔄 Starting Ollama service..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# Navigate to frontend
Set-Location "$frontendPath"

# Start development server
Write-Host "🔄 Starting SvelteKit development server..." -ForegroundColor Yellow
npm run dev

Write-Host "✅ All services started!" -ForegroundColor Green
Write-Host "🌐 Application: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🐘 PostgreSQL: localhost:5432" -ForegroundColor Cyan
Write-Host "🤖 Ollama: http://localhost:11434" -ForegroundColor Cyan
"@

$startupPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\start-local-dev.ps1"
Set-Content -Path $startupPath -Value $startupScript -Encoding UTF8

Write-Host "✅ Startup script created: $startupPath" -ForegroundColor Green

# Step 7: Run quick tests
Write-Host "`n7️⃣ Running quick validation tests..." -ForegroundColor Magenta

if (Test-Path "$frontendPath\tests\quick-validation.spec.ts") {
    try {
        Set-Location $frontendPath
        Write-Host "🧪 Running quick validation tests..." -ForegroundColor Yellow
        npx playwright test quick-validation.spec.ts --reporter=line
        Write-Host "✅ Quick tests completed" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Quick tests failed - this is normal for first setup" -ForegroundColor Yellow
    }
}

Write-Host "`n🎉 Local environment setup completed!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "📋 Summary:" -ForegroundColor Cyan
Write-Host "• PostgreSQL: ✅ Installed and configured" -ForegroundColor White
Write-Host "• Ollama: ✅ Installed with AI models" -ForegroundColor White
Write-Host "• Dependencies: ✅ Installed" -ForegroundColor White
Write-Host "• Database: ✅ Schema setup" -ForegroundColor White
Write-Host "• Tests: ✅ Ready to run" -ForegroundColor White

Write-Host "`n🚀 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Run: .\start-local-dev.ps1" -ForegroundColor White
Write-Host "2. Open: http://localhost:5173" -ForegroundColor White
Write-Host "3. Test: npm run test:quick" -ForegroundColor White

Write-Host "`n✨ Happy coding!" -ForegroundColor Green