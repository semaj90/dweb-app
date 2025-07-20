# Enhanced RAG System Setup for Windows/WSL2
# Complete Docker + local LLM initialization

Write-Host "🚀 Starting Enhanced RAG System Setup..." -ForegroundColor Green

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        $null = docker info 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

# Function to start Docker Desktop on Windows
function Start-DockerDesktop {
    Write-Host "🐳 Starting Docker Desktop..." -ForegroundColor Yellow
    try {
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        Write-Host "⏳ Waiting for Docker Desktop to start..." -ForegroundColor Yellow
        
        # Wait up to 60 seconds for Docker to start
        $timeout = 60
        $timer = 0
        do {
            Start-Sleep -Seconds 2
            $timer += 2
            if (Test-DockerRunning) {
                Write-Host "✅ Docker Desktop is now running!" -ForegroundColor Green
                return $true
            }
        } while ($timer -lt $timeout)
        
        Write-Host "⚠️ Docker Desktop failed to start within 60 seconds" -ForegroundColor Red
        return $false
    } catch {
        Write-Host "❌ Failed to start Docker Desktop: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to setup local LLM directories
function Setup-LocalLLMDirectories {
    Write-Host "📁 Setting up local LLM directories..." -ForegroundColor Blue
    
    # Create models directory structure
    $modelDirs = @(
        "models/embeddings",
        "models/chat",
        "models/classification", 
        "cache/embeddings",
        "cache/responses",
        "config/llm"
    )
    
    foreach ($dir in $modelDirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "✅ Created: $dir" -ForegroundColor Green
        }
    }
    
    # Create LLM configuration file
    $llmConfig = @{
        models = @{
            embedding = @{
                name = "legal-bert-base-uncased"
                path = "./legal-bert"
                dimensions = 768
                available = (Test-Path "./legal-bert")
            }
            chat = @{
                name = "gemma-2-2b-q4"
                path = "./gemma3Q4_K_M"  
                contextLength = 2048
                available = (Test-Path "./gemma3Q4_K_M")
            }
        }
        development = @{
            preferLocal = $true
            fallbackToCloud = $true
            autoDownload = $false
        }
    }
    
    $llmConfigJson = $llmConfig | ConvertTo-Json -Depth 4
    $llmConfigJson | Out-File -FilePath "config/llm/local-config.json" -Encoding UTF8
    Write-Host "✅ Created LLM configuration file" -ForegroundColor Green
}

# Function to setup environment files
function Setup-EnvironmentFiles {
    Write-Host "🔧 Setting up environment configuration..." -ForegroundColor Blue
    
    # Copy development environment to main .env if it doesn't exist
    if (-not (Test-Path ".env") -and (Test-Path "web-app/sveltekit-frontend/.env.development")) {
        Copy-Item "web-app/sveltekit-frontend/.env.development" ".env"
        Write-Host "✅ Created .env from development template" -ForegroundColor Green
    }
    
    # Update environment with local model paths
    $envPath = "./.env"
    if (Test-Path $envPath) {
        $envContent = Get-Content $envPath
        $envContent = $envContent -replace "OLLAMA_URL=.*", "OLLAMA_URL=http://localhost:11434"
        $envContent = $envContent -replace "EMBEDDING_MODEL=.*", "EMBEDDING_MODEL=legal-bert-base-uncased"
        $envContent | Set-Content $envPath
        Write-Host "✅ Updated environment configuration" -ForegroundColor Green
    }
}

# Main setup process
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Enhanced RAG System with Local LLM Setup" -ForegroundColor Cyan  
Write-Host "=" * 60 -ForegroundColor Cyan

# Step 1: Setup local LLM directories and config
Setup-LocalLLMDirectories

# Step 2: Setup environment files  
Setup-EnvironmentFiles

# Step 3: Check for Docker and start if needed
Write-Host "🐳 Checking Docker status..." -ForegroundColor Blue
if (-not (Test-DockerRunning)) {
    Write-Host "⚠️ Docker is not running. Attempting to start Docker Desktop..." -ForegroundColor Yellow
    
    if (Start-DockerDesktop) {
        Write-Host "✅ Docker Desktop started successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Could not start Docker Desktop automatically." -ForegroundColor Red
        Write-Host "Please start Docker Desktop manually and run this script again." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Alternative: Use WSL2 with Docker installed directly:" -ForegroundColor Cyan
        Write-Host "  wsl --install" -ForegroundColor Gray
        Write-Host "  # In WSL2:" -ForegroundColor Gray
        Write-Host "  curl -fsSL https://get.docker.com -o get-docker.sh" -ForegroundColor Gray
        Write-Host "  sudo sh get-docker.sh" -ForegroundColor Gray
        return
    }
}

# Step 4: Start Docker services
Write-Host "🚀 Starting Docker services..." -ForegroundColor Blue
try {
    docker compose down 2>$null
    docker compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker services started successfully!" -ForegroundColor Green
        
        # Wait for services to be ready
        Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        
        # Check service health
        $services = @("postgres", "qdrant", "redis")
        foreach ($service in $services) {
            $status = docker compose ps --services --filter "status=running" | Select-String $service
            if ($status) {
                Write-Host "✅ $service is running" -ForegroundColor Green
            } else {
                Write-Host "⚠️ $service may not be running correctly" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "❌ Failed to start Docker services" -ForegroundColor Red
        Write-Host "Check docker-compose.yml configuration" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Error starting Docker services: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 5: Initialize database and vector search
Write-Host "🗄️ Initializing database and vector search..." -ForegroundColor Blue
try {
    # Change to web app directory
    Push-Location "web-app/sveltekit-frontend"
    
    # Install dependencies if needed
    if (-not (Test-Path "node_modules")) {
        Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
        npm install
    }
    
    # Run database migrations
    Write-Host "🔄 Running database migrations..." -ForegroundColor Yellow
    npm run db:migrate
    
    # Initialize vector search
    Write-Host "🔍 Initializing vector search..." -ForegroundColor Yellow
    npm run vector:init
    
    Write-Host "✅ Database and vector search initialized!" -ForegroundColor Green
    
    Pop-Location
} catch {
    Write-Host "❌ Error initializing database: $($_.Exception.Message)" -ForegroundColor Red
    Pop-Location
}

# Step 6: Final setup and instructions
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "  🎉 RAG System Setup Complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start the development server:" -ForegroundColor White
Write-Host "   cd web-app/sveltekit-frontend" -ForegroundColor Gray
Write-Host "   npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Available endpoints:" -ForegroundColor White
Write-Host "   - Web App: http://localhost:5173" -ForegroundColor Gray
Write-Host "   - PostgreSQL: localhost:5432" -ForegroundColor Gray
Write-Host "   - Qdrant: http://localhost:6333" -ForegroundColor Gray
Write-Host "   - Redis: localhost:6379" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Local LLM models detected:" -ForegroundColor White
if (Test-Path "./legal-bert") {
    Write-Host "   ✅ Legal-BERT (embeddings)" -ForegroundColor Green
} else {
    Write-Host "   ⚠️ Legal-BERT not found in ./legal-bert/" -ForegroundColor Yellow
}
if (Test-Path "./gemma3Q4_K_M") {
    Write-Host "   ✅ Gemma-2 (chat)" -ForegroundColor Green
} else {
    Write-Host "   ⚠️ Gemma-2 not found in ./gemma3Q4_K_M/" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "4. Test the RAG system:" -ForegroundColor White
Write-Host "   - Visit: http://localhost:5173/rag-demo" -ForegroundColor Gray
Write-Host "   - Or: http://localhost:5173/local-ai-demo" -ForegroundColor Gray
Write-Host ""
Write-Host "🔧 For troubleshooting, check:" -ForegroundColor Cyan
Write-Host "   - Docker: docker compose logs" -ForegroundColor Gray
Write-Host "   - Database: npm run db:studio" -ForegroundColor Gray
Write-Host "   - Vector search: npm run vector:sync" -ForegroundColor Gray
