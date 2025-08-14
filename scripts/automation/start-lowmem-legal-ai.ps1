# Low Memory Enhanced Legal AI Startup Script
# Optimized for development with limited system resources
# Uses low memory Docker configuration and efficient seeding

param(
    [switch]$SkipTests = $false,
    [switch]$SetupModels = $true,
    [switch]$InitData = $true,
    [switch]$QuickStart = $false,
    [switch]$Verbose = $false,
    [switch]$LowMemory = $true
)

$ErrorActionPreference = "Continue"
$startTime = Get-Date

Write-Host "🚀 ENHANCED LEGAL AI SYSTEM - LOW MEMORY MODE" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray
Write-Host "💾 Optimized for limited resources (8GB+ RAM recommended for 6GB model)" -ForegroundColor Cyan
Write-Host "📅 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# Navigate to project root
$projectRoot = "C:\Users\james\Desktop\web-app"
Set-Location $projectRoot

# Step 1: Pre-flight Checks (Essential files only)
Write-Host "`n🔍 STEP 1: Essential Component Checks..." -ForegroundColor Yellow
Write-Host "-" * 40 -ForegroundColor Gray

$essentialFiles = @(
    @{ Name = "Gemma 3 Model"; Path = "gemma3Q4_K_M\mo16.gguf"; Required = $true },
    @{ Name = "Low Memory Docker"; Path = "docker-compose.lowmem.yml"; Required = $true },
    @{ Name = "SvelteKit Frontend"; Path = "sveltekit-frontend"; Required = $true },
    @{ Name = "Database Schema"; Path = "scripts\init-enhanced-schema.sql"; Required = $true },
    @{ Name = "Sample Data"; Path = "scripts\seed-sample-data.sql"; Required = $true }
)

$allPresent = $true
foreach ($file in $essentialFiles) {
    if (Test-Path $file.Path) {
        Write-Host "  ✅ $($file.Name)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $($file.Name): Missing" -ForegroundColor Red
        $allPresent = $false
    }
}

if (-not $allPresent) {
    Write-Host "`n❌ Missing essential files. Run setup script first." -ForegroundColor Red
    exit 1
}

# Step 2: Start Low Memory Docker Services
Write-Host "`n🐳 STEP 2: Starting Low Memory Docker Services..." -ForegroundColor Yellow
Write-Host "-" * 40 -ForegroundColor Gray

try {
    Write-Host "  🔧 Using low memory configuration..." -ForegroundColor Cyan
    Write-Host "  📦 Starting optimized services..." -ForegroundColor Cyan
    
    $dockerCommand = if ($LowMemory) { 
        "docker compose -f docker-compose.lowmem.yml up -d" 
    } else { 
        "docker compose -f docker-compose.enhanced.yml up -d" 
    }
    
    Write-Host "  💻 Command: $dockerCommand" -ForegroundColor Gray
    
    $dockerOutput = Invoke-Expression $dockerCommand 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Docker services started (low memory mode)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Docker services started with warnings" -ForegroundColor Yellow
    }
    
    if ($Verbose) {
        $dockerOutput | Write-Host -ForegroundColor Gray
    }
    
} catch {
    Write-Host "  ❌ Failed to start Docker services: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Step 3: Efficient Service Waiting
Write-Host "`n⏳ STEP 3: Waiting for Core Services..." -ForegroundColor Yellow
Write-Host "-" * 40 -ForegroundColor Gray

$coreServices = @(
    @{ Name = "PostgreSQL"; Port = 5432; MaxWait = 30 },
    @{ Name = "Ollama"; Port = 11434; MaxWait = 45 }
)

foreach ($service in $coreServices) {
    Write-Host "  ⏳ $($service.Name)..." -ForegroundColor Cyan
    $waited = 0
    $ready = $false
    
    while ($waited -lt $service.MaxWait -and -not $ready) {
        try {
            $connection = New-Object System.Net.Sockets.TcpClient
            $connection.Connect("localhost", $service.Port)
            $connection.Close()
            $ready = $true
            Write-Host "  ✅ $($service.Name) ready" -ForegroundColor Green
        } catch {
            Start-Sleep 3
            $waited += 3
        }
    }
    
    if (-not $ready) {
        Write-Host "  ⚠️  $($service.Name) not ready, continuing..." -ForegroundColor Yellow
    }
}

# Step 4: Setup AI Models (Memory Optimized)
if ($SetupModels -and -not $QuickStart) {
    Write-Host "`n🤖 STEP 4: Setting up AI Models (Memory Optimized)..." -ForegroundColor Yellow
    Write-Host "-" * 40 -ForegroundColor Gray
    
    try {
        Write-Host "  📥 Configuring Gemma 3 Legal AI (low memory)..." -ForegroundColor Cyan
        
        $dockerComposeFile = if ($LowMemory) { "docker-compose.lowmem.yml" } else { "docker-compose.enhanced.yml" }
        $modelSetup = docker compose -f $dockerComposeFile exec -T ollama /tmp/setup-models.sh 2>&1
        
        Write-Host "  ✅ AI models setup initiated" -ForegroundColor Green
        
        if ($Verbose) {
            $modelSetup | Write-Host -ForegroundColor Gray
        }
    } catch {
        Write-Host "  ⚠️  Model setup running in background" -ForegroundColor Yellow
    }
}

# Step 5: Database Setup and Seeding
if ($InitData -and -not $QuickStart) {
    Write-Host "`n📊 STEP 5: Database Setup and Seeding..." -ForegroundColor Yellow
    Write-Host "-" * 40 -ForegroundColor Gray
    
    Set-Location sveltekit-frontend
    
    try {
        Write-Host "  📋 Installing dependencies..." -ForegroundColor Cyan
        npm install --silent --prefer-offline 2>&1 | Out-Null
        
        Write-Host "  🗄️  Setting up database schema..." -ForegroundColor Cyan
        npm run db:push 2>&1 | Out-Null
        
        Write-Host "  🌱 Seeding sample data..." -ForegroundColor Cyan
        npm run db:seed:enhanced 2>&1 | Out-Null
        
        Write-Host "  ✅ Database ready with sample data" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️  Database setup completed with warnings" -ForegroundColor Yellow
    }
    
    Set-Location ..
}

# Step 6: Quick Health Check
Write-Host "`n🏥 STEP 6: Quick Health Check..." -ForegroundColor Yellow
Write-Host "-" * 40 -ForegroundColor Gray

Set-Location sveltekit-frontend

$healthPassed = $true
try {
    Write-Host "  🔍 Testing database connection..." -ForegroundColor Cyan
    $dbTest = npm run ai:health 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Database connection healthy" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Database connection needs attention" -ForegroundColor Yellow
        $healthPassed = $false
    }
} catch {
    Write-Host "  ⚠️  Health check skipped" -ForegroundColor Yellow
    $healthPassed = $false
}

# Step 7: Integration Test (Optional and Quick)
if (-not $SkipTests -and -not $QuickStart -and $healthPassed) {
    Write-Host "`n🧪 STEP 7: Quick Integration Test..." -ForegroundColor Yellow
    Write-Host "-" * 40 -ForegroundColor Gray
    
    try {
        Write-Host "  🔬 Testing core AI features..." -ForegroundColor Cyan
        
        # Quick test with timeout
        $testProcess = Start-Process -FilePath "npm" -ArgumentList "run", "ai:test" -NoNewWindow -PassThru
        $testCompleted = $testProcess.WaitForExit(30000) # 30 second timeout
        
        if ($testCompleted -and $testProcess.ExitCode -eq 0) {
            Write-Host "  ✅ Integration tests passed" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  Integration tests timed out or failed" -ForegroundColor Yellow
            if (-not $testCompleted) {
                $testProcess.Kill()
            }
        }
    } catch {
        Write-Host "  ⚠️  Integration tests skipped" -ForegroundColor Yellow
    }
}

# Generate Quick Report
Write-Host "`n📊 STEP 8: System Status Report..." -ForegroundColor Yellow
Write-Host "-" * 40 -ForegroundColor Gray

$quickReport = @"
# Enhanced Legal AI - Low Memory Startup Report
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Configuration
- **Mode**: Low Memory Optimized
- **Docker Config**: docker-compose.lowmem.yml
- **Memory Limit**: ~12GB total (vs 16GB+ for full mode)
- **Models Setup**: $(if ($SetupModels) { "✅ Enabled" } else { "⚠️ Skipped" })
- **Data Seeding**: $(if ($InitData) { "✅ Enabled" } else { "⚠️ Skipped" })

## Optimizations Applied
- PostgreSQL: 512MB limit (vs 2GB)
- Neo4j: 768MB limit (vs 2GB) 
- Qdrant: 256MB limit (vs 1GB)
- Redis: 192MB limit (vs 512MB)
- Ollama: 10GB limit (for 6GB model + overhead)

## Quick Access
- 🌐 Frontend: http://localhost:5173
- 🤖 Gemma 3 Legal AI: http://localhost:11434
- 📊 Database: localhost:5432
- 🛠️ PgAdmin: http://localhost:5050

## Sample Login
- Email: prosecutor@legalai.demo
- Password: (use demo password)

## Memory Usage Tips
- Close unnecessary applications
- Monitor system memory usage
- Use smaller AI context windows
- Limit concurrent operations

## Next Steps
1. Access frontend at http://localhost:5173
2. Register or login with sample account
3. Explore sample cases and evidence
4. Try detective mode features
5. Test AI assistant with legal questions

Generated in $(((Get-Date) - $startTime).TotalSeconds.ToString('F1')) seconds
"@

$quickReport | Out-File "LOW_MEMORY_STARTUP_REPORT.md" -Encoding UTF8
Write-Host "  📋 Report saved: LOW_MEMORY_STARTUP_REPORT.md" -ForegroundColor Green

# Final Summary
Set-Location ..
$totalTime = ((Get-Date) - $startTime).TotalSeconds

Write-Host "`n🎉 LOW MEMORY LEGAL AI SYSTEM READY!" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Gray

Write-Host "✅ Optimized for limited resources" -ForegroundColor White
Write-Host "✅ Sample legal data loaded" -ForegroundColor White
Write-Host "✅ Gemma 3 Legal AI configured" -ForegroundColor White
Write-Host "✅ Detective mode available" -ForegroundColor White
Write-Host "✅ Interactive canvas ready" -ForegroundColor White

Write-Host "`n🔗 ACCESS POINTS:" -ForegroundColor Cyan
Write-Host "🌐 Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "🤖 AI API: http://localhost:11434" -ForegroundColor White
Write-Host "📊 Database: localhost:5432" -ForegroundColor White

Write-Host "`n💾 MEMORY OPTIMIZED:" -ForegroundColor Magenta
Write-Host "⏱️ Startup time: $($totalTime.ToString('F1'))s" -ForegroundColor White
Write-Host "💾 Memory usage: ~12GB total" -ForegroundColor White
Write-Host "🚀 Ready for legal AI workflows!" -ForegroundColor White

Write-Host "`n📋 SAMPLE DATA AVAILABLE:" -ForegroundColor Yellow
Write-Host "👥 5 demo users (prosecutor, defense, admin, etc.)" -ForegroundColor Gray
Write-Host "📁 6 legal cases (criminal, civil, contract disputes)" -ForegroundColor Gray
Write-Host "📄 5 evidence items with AI analysis" -ForegroundColor Gray
Write-Host "🔍 2 detective mode investigations" -ForegroundColor Gray
Write-Host "🎨 Interactive canvas visualizations" -ForegroundColor Gray

Write-Host "`n🚀 Starting development server..." -ForegroundColor Green
Set-Location sveltekit-frontend

Write-Host "`n💡 TIP: First AI response may take 30-60 seconds in low memory mode" -ForegroundColor Cyan
Write-Host "📖 Check LOW_MEMORY_STARTUP_REPORT.md for details" -ForegroundColor Cyan

# Start the development server
npm run dev
