# Local Development Environment Startup Script
# Run this script to start all services for development

Write-Host "🚀 Starting Local Development Environment..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Function to check if a port is in use
function Test-PortInUse {
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

# Check and start PostgreSQL
Write-Host "`n1️⃣ Checking PostgreSQL..." -ForegroundColor Magenta

$postgresService = Get-Service -Name "postgresql-x64-*" -ErrorAction SilentlyContinue
if ($postgresService) {
    if ($postgresService.Status -eq "Running") {
        Write-Host "✅ PostgreSQL service is running" -ForegroundColor Green
    } else {
        Write-Host "🔄 Starting PostgreSQL service..." -ForegroundColor Yellow
        Start-Service $postgresService
        Start-Sleep -Seconds 3
        Write-Host "✅ PostgreSQL service started" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️  PostgreSQL service not found - please run setup first" -ForegroundColor Yellow
}

# Check database connection
if (Test-PortInUse -Port 5432) {
    Write-Host "✅ PostgreSQL is accessible on port 5432" -ForegroundColor Green
} else {
    Write-Host "❌ PostgreSQL is not accessible on port 5432" -ForegroundColor Red
}

# Check and start Ollama
Write-Host "`n2️⃣ Checking Ollama..." -ForegroundColor Magenta

if (Test-PortInUse -Port 11434) {
    Write-Host "✅ Ollama is already running on port 11434" -ForegroundColor Green
} else {
    Write-Host "🔄 Starting Ollama service..." -ForegroundColor Yellow
    
    try {
        # Try to start Ollama
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
        
        # Wait for service to start
        $attempts = 0
        $maxAttempts = 10
        
        while ($attempts -lt $maxAttempts -and -not (Test-PortInUse -Port 11434)) {
            Start-Sleep -Seconds 2
            $attempts++
            Write-Host "⏳ Waiting for Ollama to start... ($attempts/$maxAttempts)" -ForegroundColor Yellow
        }
        
        if (Test-PortInUse -Port 11434) {
            Write-Host "✅ Ollama service started successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Ollama failed to start" -ForegroundColor Red
        }
        
    } catch {
        Write-Host "❌ Failed to start Ollama: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "💡 Try running: ollama serve" -ForegroundColor Yellow
    }
}

# Check available models
Write-Host "`n3️⃣ Checking AI models..." -ForegroundColor Magenta

try {
    $models = & ollama list 2>$null
    if ($models) {
        Write-Host "✅ AI models available:" -ForegroundColor Green
        Write-Host $models -ForegroundColor Gray
    } else {
        Write-Host "⚠️  No AI models found - pulling required models..." -ForegroundColor Yellow
        
        $requiredModels = @("llama3.2", "nomic-embed-text")
        foreach ($model in $requiredModels) {
            Write-Host "📦 Pulling $model..." -ForegroundColor Yellow
            & ollama pull $model
        }
    }
} catch {
    Write-Host "❌ Failed to check models: $($_.Exception.Message)" -ForegroundColor Red
}

# Check GPU availability
Write-Host "`n4️⃣ Checking GPU..." -ForegroundColor Magenta

try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>$null
    if ($gpuInfo) {
        Write-Host "✅ NVIDIA GPU available:" -ForegroundColor Green
        $gpuInfo | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
        
        # Set GPU environment variables for Ollama
        $env:OLLAMA_GPU_ENABLED = "true"
        $env:CUDA_VISIBLE_DEVICES = "0"
        
    } else {
        Write-Host "⚠️  No NVIDIA GPU detected - using CPU mode" -ForegroundColor Yellow
        $env:OLLAMA_GPU_ENABLED = "false"
    }
} catch {
    Write-Host "⚠️  GPU check failed - using CPU mode" -ForegroundColor Yellow
    $env:OLLAMA_GPU_ENABLED = "false"
}

# Navigate to frontend directory
$frontendPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

if (Test-Path $frontendPath) {
    Set-Location $frontendPath
    Write-Host "`n5️⃣ Starting SvelteKit development server..." -ForegroundColor Magenta
    
    # Check if dependencies are installed
    if (-not (Test-Path "node_modules")) {
        Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
        npm install
    }
    
    # Check if dev server is already running
    if (Test-PortInUse -Port 5173) {
        Write-Host "⚠️  Port 5173 is already in use" -ForegroundColor Yellow
        Write-Host "🌐 SvelteKit may already be running at: http://localhost:5173" -ForegroundColor Cyan
    } else {
        Write-Host "🔄 Starting development server..." -ForegroundColor Yellow
        
        # Start the development server
        Write-Host "`n✅ All services started!" -ForegroundColor Green
        Write-Host "==============================================" -ForegroundColor Cyan
        Write-Host "🌐 Application: http://localhost:5173" -ForegroundColor Cyan
        Write-Host "🐘 PostgreSQL: localhost:5432" -ForegroundColor Cyan
        Write-Host "🤖 Ollama: http://localhost:11434" -ForegroundColor Cyan
        Write-Host "📊 Database Admin: pgAdmin (from Start Menu)" -ForegroundColor Cyan
        Write-Host "`n💡 Test commands:" -ForegroundColor Yellow
        Write-Host "   npm run test:quick       # Quick validation" -ForegroundColor Gray
        Write-Host "   npm run test:auth        # Authentication tests" -ForegroundColor Gray
        Write-Host "   npm run test:comprehensive # Full test suite" -ForegroundColor Gray
        Write-Host "`n🛑 Press Ctrl+C to stop all services" -ForegroundColor Yellow
        Write-Host "==============================================" -ForegroundColor Cyan
        
        # Start the development server (this will block)
        npm run dev
    }
    
} else {
    Write-Host "❌ Frontend directory not found: $frontendPath" -ForegroundColor Red
    Write-Host "Please make sure the project is in the correct location." -ForegroundColor Yellow
}

Write-Host "`n🛑 Development server stopped." -ForegroundColor Yellow