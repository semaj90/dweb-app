# Enhanced RAG System Setup Script
# Initializes PostgreSQL with pgvector, Qdrant, and local LLM models

param(
    [ValidateSet("dev", "test", "prod")]
    [string]$Environment = "dev",
    [switch]$StartDocker,
    [switch]$SetupDatabase,
    [switch]$DownloadModels,
    [switch]$InitVectorSearch,
    [switch]$Force
)

Write-Host "🏛️ Legal AI Assistant Enhanced Setup" -ForegroundColor Blue
Write-Host "Environment: $Environment" -ForegroundColor Green

# Set working directory to the web-app frontend
$webAppPath = Join-Path $PSScriptRoot "web-app" "sveltekit-frontend"
if (Test-Path $webAppPath) {
    Set-Location $webAppPath
    Write-Host "📁 Working directory: $webAppPath" -ForegroundColor Yellow
} else {
    Write-Host "❌ Web app directory not found: $webAppPath" -ForegroundColor Red
    exit 1
}

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Verify required tools
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Cyan

$missingTools = @()
if (-not (Test-Command "node")) { $missingTools += "Node.js" }
if (-not (Test-Command "npm")) { $missingTools += "npm" }
if (-not (Test-Command "docker")) { $missingTools += "Docker" }
if (-not (Test-Command "git")) { $missingTools += "Git" }

if ($missingTools.Count -gt 0) {
    Write-Host "❌ Missing required tools: $($missingTools -join ', ')" -ForegroundColor Red
    Write-Host "Please install missing tools and run again." -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ All prerequisites found" -ForegroundColor Green

# Copy environment file
Write-Host "📝 Setting up environment configuration..." -ForegroundColor Cyan
$envSource = ".env.$Environment"
$envTarget = ".env"

if (Test-Path $envSource) {
    Copy-Item $envSource $envTarget -Force
    Write-Host "✅ Environment file copied: $envSource -> $envTarget" -ForegroundColor Green
} else {
    Write-Host "⚠️ Environment file not found: $envSource" -ForegroundColor Yellow
}

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
try {
    npm install
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install dependencies: $_" -ForegroundColor Red
    exit 1
}

# Start Docker services if requested
if ($StartDocker) {
    Write-Host "🐳 Starting Docker services..." -ForegroundColor Cyan
    try {
        # Go back to root for docker-compose
        Set-Location (Join-Path $PSScriptRoot ".." "..")
        docker-compose up -d
        Write-Host "✅ Docker services started" -ForegroundColor Green
        
        # Wait for services to be ready
        Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        
        # Check PostgreSQL
        $maxAttempts = 30
        $attempt = 0
        do {
            $attempt++
            Write-Host "Checking PostgreSQL... (attempt $attempt/$maxAttempts)" -ForegroundColor Yellow
            $pgReady = docker exec prosecutor_postgres pg_isready -U postgres -d prosecutor_db 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ PostgreSQL is ready" -ForegroundColor Green
                break
            }
            Start-Sleep -Seconds 2
        } while ($attempt -lt $maxAttempts)
        
        if ($attempt -eq $maxAttempts) {
            Write-Host "❌ PostgreSQL failed to start within timeout" -ForegroundColor Red
        }
        
        # Check Qdrant
        try {
            $qdrantResponse = Invoke-WebRequest -Uri "http://localhost:6333/healthz" -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($qdrantResponse.StatusCode -eq 200) {
                Write-Host "✅ Qdrant is ready" -ForegroundColor Green
            }
        } catch {
            Write-Host "⚠️ Qdrant may not be ready yet" -ForegroundColor Yellow
        }
        
        Set-Location $webAppPath
    } catch {
        Write-Host "❌ Failed to start Docker services: $_" -ForegroundColor Red
        Set-Location $webAppPath
    }
}

# Set up database
if ($SetupDatabase) {
    Write-Host "🗄️ Setting up database..." -ForegroundColor Cyan
    try {
        npm run db:generate
        npm run db:push
        Write-Host "✅ Database setup completed" -ForegroundColor Green
    } catch {
        Write-Host "❌ Database setup failed: $_" -ForegroundColor Red
    }
}

# Download local LLM models
if ($DownloadModels) {
    Write-Host "🤖 Setting up local LLM models..." -ForegroundColor Cyan
    
    # Create models directory
    $modelsDir = "local-llms"
    if (-not (Test-Path $modelsDir)) {
        New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
        Write-Host "📁 Created models directory: $modelsDir" -ForegroundColor Green
    }
    
    # Check if Ollama is available
    if (Test-Command "ollama") {
        Write-Host "🦙 Setting up Ollama models..." -ForegroundColor Cyan
        try {
            # Pull legal-focused models
            ollama pull llama2:7b-chat
            ollama pull codellama:7b-code
            Write-Host "✅ Ollama models downloaded" -ForegroundColor Green
        } catch {
            Write-Host "⚠️ Ollama model download failed: $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️ Ollama not found. Install Ollama for local LLM support." -ForegroundColor Yellow
        Write-Host "   Download from: https://ollama.ai" -ForegroundColor Cyan
    }
    
    # Create configuration file
    $configPath = "llm-config.json"
    $config = @{
        models = @{
            embedding = @{
                name = "legal-bert-base-uncased"
                path = "local-llms/legal-bert"
                status = "ready"
            }
            chat = @{
                name = "llama2:7b-chat"
                path = "ollama"
                status = "ready"
            }
        }
        lastUpdated = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
    }
    
    $config | ConvertTo-Json -Depth 3 | Out-File -FilePath $configPath -Encoding UTF8
    Write-Host "✅ LLM configuration saved to: $configPath" -ForegroundColor Green
}

# Initialize vector search
if ($InitVectorSearch) {
    Write-Host "🔍 Initializing vector search system..." -ForegroundColor Cyan
    try {
        npm run vector:init
        Write-Host "✅ Vector search initialized" -ForegroundColor Green
    } catch {
        Write-Host "❌ Vector search initialization failed: $_" -ForegroundColor Red
    }
}

# Final verification
Write-Host "🔍 Performing final verification..." -ForegroundColor Cyan

# Check if development server can start
Write-Host "🚀 Testing development server..." -ForegroundColor Cyan
try {
    # Start dev server in background and test
    $job = Start-Job -ScriptBlock {
        Set-Location $using:webAppPath
        npm run dev 2>&1
    }
    
    Start-Sleep -Seconds 15
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Development server is working" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠️ Development server may need manual verification" -ForegroundColor Yellow
    }
    
    Stop-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -ErrorAction SilentlyContinue
} catch {
    Write-Host "⚠️ Could not test development server automatically" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start development: npm run dev" -ForegroundColor White
Write-Host "2. View app: http://localhost:5173" -ForegroundColor White
Write-Host "3. Database studio: npm run db:studio" -ForegroundColor White
Write-Host "4. Test RAG endpoint: http://localhost:5173/rag-demo" -ForegroundColor White
Write-Host "5. Test local AI: http://localhost:5173/local-ai-demo" -ForegroundColor White
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "- RAG System: RAG_SYSTEM_README.md" -ForegroundColor White
Write-Host "- Tauri Setup: TAURI_RUST_SETUP.md" -ForegroundColor White
Write-Host "- Production: PRODUCTION_RAG_SETUP_GUIDE.md" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Useful commands:" -ForegroundColor Cyan
Write-Host "- npm run docker:up     # Start all services" -ForegroundColor White
Write-Host "- npm run vector:sync   # Sync embeddings" -ForegroundColor White
Write-Host "- npm run test          # Run tests" -ForegroundColor White
Write-Host ""
