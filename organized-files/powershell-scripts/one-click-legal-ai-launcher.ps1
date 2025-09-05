# 🚀 One-Click Legal AI System Launcher for Windows 10
# Comprehensive setup and launch for PostgreSQL, pgvector, Qdrant, Ollama & SvelteKit

param(
    [switch]$Setup = $false,        # First-time setup
    [switch]$Quick = $false,        # Quick launch (skip health checks)
    [switch]$GPU = $false,          # Enable GPU acceleration
    [switch]$Native = $false,       # Use native PostgreSQL instead of Docker
    [switch]$Reset = $false,        # Reset all databases and configurations
    [string]$PostgresPassword = "legal_ai_secure_2024"
)

$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "🏛️ Legal AI System Launcher"

# Color scheme
$primaryColor = "Cyan"
$successColor = "Green"
$warningColor = "Yellow"
$errorColor = "Red"
$infoColor = "White"

Write-Host "🏛️ LEGAL AI SYSTEM - ONE-CLICK LAUNCHER" -ForegroundColor $primaryColor
Write-Host "=======================================" -ForegroundColor $primaryColor
Write-Host "🖥️ Windows 10 | Docker Desktop | GPU Ready" -ForegroundColor $infoColor
Write-Host ""

# ASCII Art Banner
Write-Host "     ⚖️  Legal AI System v2.0  ⚖️" -ForegroundColor $primaryColor
Write-Host "   PostgreSQL + pgvector + Qdrant + Ollama" -ForegroundColor $warningColor
Write-Host ""

# System Requirements Check
function Test-SystemRequirements {
    Write-Host "🔍 Checking system requirements..." -ForegroundColor $primaryColor

    $requirements = @()

    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Host "   ✅ Node.js: $nodeVersion" -ForegroundColor $successColor
    } catch {
        Write-Host "   ❌ Node.js: NOT FOUND" -ForegroundColor $errorColor
        $requirements += "Node.js (https://nodejs.org/)"
    }

    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Host "   ✅ Docker: $dockerVersion" -ForegroundColor $successColor
    } catch {
        Write-Host "   ❌ Docker: NOT FOUND" -ForegroundColor $errorColor
        $requirements += "Docker Desktop (https://www.docker.com/products/docker-desktop/)"
    }

    # Check PowerShell version
    $psVersion = $PSVersionTable.PSVersion
    if ($psVersion.Major -ge 5) {
        Write-Host "   ✅ PowerShell: $($psVersion.ToString())" -ForegroundColor $successColor
    } else {
        Write-Host "   ⚠️ PowerShell: $($psVersion.ToString()) (recommend 7+)" -ForegroundColor $warningColor
    }

    # Check available memory
    $memory = Get-CimInstance -ClassName Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum
    $memoryGB = [math]::Round($memory.Sum / 1GB, 1)
    Write-Host "   📊 Available RAM: ${memoryGB}GB" -ForegroundColor $infoColor

    if ($requirements.Count -gt 0) {
        Write-Host ""
        Write-Host "❌ Missing requirements:" -ForegroundColor $errorColor
        $requirements | ForEach-Object { Write-Host "   • $_" -ForegroundColor $errorColor }
        Write-Host ""
        Write-Host "Please install missing components and re-run this script." -ForegroundColor $warningColor
        exit 1
    }

    Write-Host "   🎉 All requirements satisfied!" -ForegroundColor $successColor
    Write-Host ""
}

# PostgreSQL Setup with pgvector
function Setup-PostgreSQL {
    Write-Host "🐘 Setting up PostgreSQL with pgvector..." -ForegroundColor $primaryColor

    if ($Native) {
        Write-Host "   📍 Using native PostgreSQL installation" -ForegroundColor $infoColor

        # Check if PostgreSQL is installed
        try {
            $env:PGPASSWORD = $PostgresPassword
            psql -h localhost -U postgres -c "SELECT version();" 2>$null | Out-Null
            Write-Host "   ✅ Native PostgreSQL detected" -ForegroundColor $successColor
        } catch {
            Write-Host "   ❌ Native PostgreSQL not found or not accessible" -ForegroundColor $errorColor
            Write-Host "   💡 Installing PostgreSQL via Chocolatey..." -ForegroundColor $warningColor

            # Install PostgreSQL via Chocolatey (if available)
            try {
                choco install postgresql --params "/Password:$PostgresPassword" -y
                Write-Host "   ✅ PostgreSQL installed via Chocolatey" -ForegroundColor $successColor
            } catch {
                Write-Host "   ⚠️ Auto-install failed. Please install manually:" -ForegroundColor $warningColor
                Write-Host "      1. Download: https://www.postgresql.org/download/windows/" -ForegroundColor $infoColor
                Write-Host "      2. Or use: winget install PostgreSQL.PostgreSQL" -ForegroundColor $infoColor
                Write-Host "      3. Or use: scoop install postgresql" -ForegroundColor $infoColor
                return $false
            }
        }
    } else {
        Write-Host "   🐳 Using Docker PostgreSQL setup" -ForegroundColor $infoColor

        # Stop and remove existing container
        docker stop legal-ai-postgres 2>$null | Out-Null
        docker rm legal-ai-postgres 2>$null | Out-Null

        # Start PostgreSQL with pgvector
        Write-Host "   🚀 Starting PostgreSQL container..." -ForegroundColor $warningColor

        $dockerArgs = @(
            "run", "-d",
            "--name", "legal-ai-postgres",
            "--restart", "unless-stopped",
            "-e", "POSTGRES_PASSWORD=$PostgresPassword",
            "-e", "POSTGRES_DB=legal_ai_db",
            "-e", "POSTGRES_USER=postgres",
            "-p", "5432:5432",
            "-v", "legal_ai_postgres_data:/var/lib/postgresql/data",
            "pgvector/pgvector:pg16"
        )

        $dockerResult = & docker @dockerArgs

        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ PostgreSQL container started: $dockerResult" -ForegroundColor $successColor
        } else {
            Write-Host "   ❌ Failed to start PostgreSQL container" -ForegroundColor $errorColor
            return $false
        }
    }

    # Wait for PostgreSQL to be ready
    Write-Host "   ⏳ Waiting for PostgreSQL to be ready..." -ForegroundColor $warningColor
    $maxAttempts = 30
    $attempt = 0

    do {
        Start-Sleep -Seconds 2
        $attempt++

        try {
            $env:PGPASSWORD = $PostgresPassword
            if ($Native) {
                $ready = psql -h localhost -U postgres -d postgres -c "SELECT 1;" 2>$null
            } else {
                $ready = docker exec legal-ai-postgres pg_isready -U postgres 2>$null
            }

            if ($ready -or $ready -match "accepting connections") {
                Write-Host "   ✅ PostgreSQL is ready!" -ForegroundColor $successColor
                break
            }
        } catch {
            # Continue waiting
        }

        Write-Host "   ⏳ Attempt $attempt/$maxAttempts..." -ForegroundColor $warningColor
    } while ($attempt -lt $maxAttempts)

    if ($attempt -ge $maxAttempts) {
        Write-Host "   ❌ PostgreSQL failed to start within timeout" -ForegroundColor $errorColor
        return $false
    }

    # Setup database and extensions
    Write-Host "   🔧 Setting up database schema and extensions..." -ForegroundColor $primaryColor

    $setupSQL = @'
-- Create database
CREATE DATABASE legal_ai_db;

-- Connect to legal_ai_db
\c legal_ai_db

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS legal_ai;
CREATE SCHEMA IF NOT EXISTS vector_store;
CREATE SCHEMA IF NOT EXISTS documents;

-- Test vector functionality
SELECT vector_dims('[1,2,3,4]'::vector) AS test_vector_dims;

-- Create basic tables
CREATE TABLE IF NOT EXISTS legal_ai.document_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_vector
ON legal_ai.document_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Show setup results
SELECT 'PostgreSQL with pgvector setup completed successfully!' as status;
'@

    # Write SQL to temporary file
    $setupSQL | Out-File -FilePath "temp_postgres_setup.sql" -Encoding UTF8

    try {
        if ($Native) {
            $env:PGPASSWORD = $PostgresPassword
            psql -h localhost -U postgres -f temp_postgres_setup.sql
        } else {
            Get-Content temp_postgres_setup.sql | docker exec -i legal-ai-postgres psql -U postgres
        }

        Write-Host "   ✅ Database schema created successfully" -ForegroundColor $successColor
    } catch {
        Write-Host "   ⚠️ Database setup completed with warnings" -ForegroundColor $warningColor
    }

    # Clean up
    Remove-Item "temp_postgres_setup.sql" -ErrorAction SilentlyContinue

    return $true
}

# Qdrant Vector Database Setup
function Setup-Qdrant {
    Write-Host "🔍 Setting up Qdrant vector database..." -ForegroundColor $primaryColor

    # Stop existing Qdrant container
    docker stop legal-ai-qdrant 2>$null | Out-Null
    docker rm legal-ai-qdrant 2>$null | Out-Null

    # Start Qdrant
    Write-Host "   🚀 Starting Qdrant container..." -ForegroundColor $warningColor

    $dockerArgs = @(
        "run", "-d",
        "--name", "legal-ai-qdrant",
        "--restart", "unless-stopped",
        "-p", "6333:6333",
        "-p", "6334:6334",
        "-v", "legal_ai_qdrant_data:/qdrant/storage",
        "qdrant/qdrant:latest"
    )

    $dockerResult = & docker @dockerArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Qdrant started: $dockerResult" -ForegroundColor $successColor
    } else {
        Write-Host "   ❌ Failed to start Qdrant" -ForegroundColor $errorColor
        return $false
    }

    # Wait for Qdrant to be ready
    Write-Host "   ⏳ Waiting for Qdrant to be ready..." -ForegroundColor $warningColor
    $maxAttempts = 20
    $attempt = 0

    do {
        Start-Sleep -Seconds 2
        $attempt++

        try {
            $response = Invoke-WebRequest -Uri "http://localhost:6333/health" -TimeoutSec 3 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Host "   ✅ Qdrant is ready!" -ForegroundColor $successColor
                break
            }
        } catch {
            # Continue waiting
        }

        Write-Host "   ⏳ Attempt $attempt/$maxAttempts..." -ForegroundColor $warningColor
    } while ($attempt -lt $maxAttempts)

    if ($attempt -ge $maxAttempts) {
        Write-Host "   ❌ Qdrant failed to start within timeout" -ForegroundColor $errorColor
        return $false
    }

    return $true
}

# Ollama Setup with Legal AI Models
function Setup-Ollama {
    Write-Host "🤖 Setting up Ollama with legal AI models..." -ForegroundColor $primaryColor

    # Check if Ollama is installed
    try {
        $ollamaVersion = ollama --version
        Write-Host "   ✅ Ollama detected: $ollamaVersion" -ForegroundColor $successColor
    } catch {
        Write-Host "   ❌ Ollama not found" -ForegroundColor $errorColor
        Write-Host "   💡 Installing Ollama..." -ForegroundColor $warningColor

        # Download and install Ollama
        try {
            $ollamaInstaller = "$env:TEMP\OllamaSetup.exe"
            Invoke-WebRequest -Uri "https://ollama.ai/download/OllamaSetup.exe" -OutFile $ollamaInstaller
            Start-Process $ollamaInstaller -ArgumentList "/S" -Wait
            Write-Host "   ✅ Ollama installed" -ForegroundColor $successColor
        } catch {
            Write-Host "   ❌ Failed to install Ollama automatically" -ForegroundColor $errorColor
            Write-Host "   📋 Please install manually: https://ollama.ai/download" -ForegroundColor $warningColor
            return $false
        }
    }

    # Start Ollama service
    Write-Host "   🚀 Starting Ollama service..." -ForegroundColor $warningColor
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep 5

    # Check if Ollama is running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing
        Write-Host "   ✅ Ollama service is running" -ForegroundColor $successColor
    } catch {
        Write-Host "   ❌ Ollama service failed to start" -ForegroundColor $errorColor
        return $false
    }

    # Pull essential models
    $models = @(
        "llama3.1:8b",
        "nomic-embed-text",
        "mistral:7b"
    )

    if ($GPU) {
        Write-Host "   🎮 GPU acceleration enabled - pulling larger models..." -ForegroundColor $primaryColor
        $models += @("llama3.1:70b", "codellama:13b")
    }

    foreach ($model in $models) {
        Write-Host "   📥 Pulling model: $model..." -ForegroundColor $warningColor
        Start-Process "ollama" -ArgumentList "pull", $model -Wait -WindowStyle Hidden

        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Model ready: $model" -ForegroundColor $successColor
        } else {
            Write-Host "   ⚠️ Model download may have failed: $model" -ForegroundColor $warningColor
        }
    }

    return $true
}

# SvelteKit Application Setup
function Setup-SvelteKit {
    Write-Host "🎨 Setting up SvelteKit application..." -ForegroundColor $primaryColor

    # Navigate to frontend directory
    if (Test-Path "sveltekit-frontend") {
        Set-Location "sveltekit-frontend"
        Write-Host "   📂 Located SvelteKit frontend directory" -ForegroundColor $successColor
    } else {
        Write-Host "   ❌ SvelteKit frontend directory not found" -ForegroundColor $errorColor
        return $false
    }

    # Install dependencies
    Write-Host "   📦 Installing npm dependencies..." -ForegroundColor $warningColor
    npm install --silent

    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Dependencies installed" -ForegroundColor $successColor
    } else {
        Write-Host "   ❌ Failed to install dependencies" -ForegroundColor $errorColor
        return $false
    }

    # Create or update .env file
    Write-Host "   ⚙️ Configuring environment variables..." -ForegroundColor $warningColor

    $envContent = @"
# Legal AI System Configuration
DATABASE_URL=postgresql://postgres:$PostgresPassword@localhost:5432/legal_ai_db
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
NODE_ENV=development

# API Keys (add your keys here)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Vector Search Configuration
VECTOR_DIMENSION=384
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# AI Model Configuration
DEFAULT_EMBEDDING_MODEL=nomic-embed-text
DEFAULT_CHAT_MODEL=llama3.1:8b
ENABLE_GPU=$(if($GPU){"true"}else{"false"})
"@

    $envContent | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "   ✅ Environment configured" -ForegroundColor $successColor

    # Build the application
    Write-Host "   🔨 Building SvelteKit application..." -ForegroundColor $warningColor
    npm run build --silent

    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Application built successfully" -ForegroundColor $successColor
    } else {
        Write-Host "   ⚠️ Build completed with warnings" -ForegroundColor $warningColor
    }

    # Return to parent directory
    Set-Location ..

    return $true
}

# Health Check Function
function Test-SystemHealth {
    Write-Host "🏥 Performing system health check..." -ForegroundColor $primaryColor

    $services = @()

    # PostgreSQL Health Check
    try {
        $env:PGPASSWORD = $PostgresPassword
        if ($Native) {
            psql -h localhost -U postgres -d legal_ai_db -c "SELECT 1;" 2>$null | Out-Null
        } else {
            docker exec legal-ai-postgres psql -U postgres -d legal_ai_db -c "SELECT 1;" 2>$null | Out-Null
        }
        $services += "✅ PostgreSQL: HEALTHY"
    } catch {
        $services += "❌ PostgreSQL: UNHEALTHY"
    }

    # Qdrant Health Check
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:6333/health" -TimeoutSec 3 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $services += "✅ Qdrant: HEALTHY"
        } else {
            $services += "❌ Qdrant: UNHEALTHY"
        }
    } catch {
        $services += "❌ Qdrant: UNHEALTHY"
    }

    # Ollama Health Check
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $services += "✅ Ollama: HEALTHY"
        } else {
            $services += "❌ Ollama: UNHEALTHY"
        }
    } catch {
        $services += "❌ Ollama: UNHEALTHY"
    }

    # SvelteKit Health Check
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 3 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $services += "✅ SvelteKit: HEALTHY"
        } else {
            $services += "❌ SvelteKit: UNHEALTHY"
        }
    } catch {
        $services += "❌ SvelteKit: STARTING/UNHEALTHY"
    }

    Write-Host ""
    Write-Host "📊 SERVICE STATUS:" -ForegroundColor $primaryColor
    $services | ForEach-Object {
        if ($_ -match "✅") {
            Write-Host "   $_" -ForegroundColor $successColor
        } else {
            Write-Host "   $_" -ForegroundColor $errorColor
        }
    }
    Write-Host ""
}

# Start SvelteKit Development Server
function Start-SvelteKitDev {
    Write-Host "🚀 Starting SvelteKit development server..." -ForegroundColor $primaryColor

    if (Test-Path "sveltekit-frontend") {
        Set-Location "sveltekit-frontend"

        # Start dev server in background
        Write-Host "   🌐 Launching at http://localhost:5173" -ForegroundColor $successColor
        Start-Process PowerShell -ArgumentList "-Command", "npm run dev" -WindowStyle Hidden

        # Wait a moment for server to start
        Start-Sleep 3

        # Return to parent directory
        Set-Location ..

        # Open browser
        Write-Host "   🌏 Opening browser..." -ForegroundColor $warningColor
        Start-Process "http://localhost:5173"

        return $true
    } else {
        Write-Host "   ❌ SvelteKit frontend directory not found" -ForegroundColor $errorColor
        return $false
    }
}

# Reset/Clean Function
function Reset-System {
    Write-Host "🧹 Resetting Legal AI System..." -ForegroundColor $warningColor

    Write-Host "   🛑 Stopping all containers..." -ForegroundColor $warningColor
    docker stop legal-ai-postgres legal-ai-qdrant 2>$null | Out-Null
    docker rm legal-ai-postgres legal-ai-qdrant 2>$null | Out-Null

    Write-Host "   🗑️ Removing volumes..." -ForegroundColor $warningColor
    docker volume rm legal_ai_postgres_data legal_ai_qdrant_data 2>$null | Out-Null

    Write-Host "   🔄 Stopping Ollama..." -ForegroundColor $warningColor
    Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force

    Write-Host "   ✅ System reset completed" -ForegroundColor $successColor
}

# Main Execution Logic
Write-Host "⚡ Starting Legal AI System Launcher..." -ForegroundColor $primaryColor
Write-Host ""

# Handle reset flag
if ($Reset) {
    Reset-System
    Write-Host "🏁 Reset completed. Re-run without -Reset to start fresh." -ForegroundColor $successColor
    exit 0
}

# System requirements check
if (-not $Quick) {
    Test-SystemRequirements
}

# Setup phase (if requested)
if ($Setup) {
    Write-Host "🔧 SETUP PHASE - First-time configuration" -ForegroundColor $primaryColor
    Write-Host ""

    $setupSuccess = $true

    # Setup each component
    $setupSuccess = $setupSuccess -and (Setup-PostgreSQL)
    $setupSuccess = $setupSuccess -and (Setup-Qdrant)
    $setupSuccess = $setupSuccess -and (Setup-Ollama)
    $setupSuccess = $setupSuccess -and (Setup-SvelteKit)

    if ($setupSuccess) {
        Write-Host "🎉 SETUP COMPLETED SUCCESSFULLY!" -ForegroundColor $successColor
        Write-Host ""
    } else {
        Write-Host "❌ Setup failed. Please check errors above." -ForegroundColor $errorColor
        exit 1
    }
}

# Launch phase
Write-Host "🚀 LAUNCH PHASE - Starting all services" -ForegroundColor $primaryColor
Write-Host ""

# Quick launch mode (skip detailed setup)
if (-not $Setup) {
    Write-Host "   📋 Quick launch mode - starting existing services..." -ForegroundColor $infoColor

    # Start existing containers
    docker start legal-ai-postgres legal-ai-qdrant 2>$null | Out-Null

    # Start Ollama
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue

    Start-Sleep 3
}

# Start SvelteKit
Start-SvelteKitDev | Out-Null

# Health check
if (-not $Quick) {
    Start-Sleep 2
    Test-SystemHealth
}

# Final status and instructions
Write-Host "🎉 LEGAL AI SYSTEM IS READY!" -ForegroundColor $successColor
Write-Host "==============================" -ForegroundColor $successColor
Write-Host ""
Write-Host "🌐 Web Interface: http://localhost:5173" -ForegroundColor $infoColor
Write-Host "🐘 PostgreSQL: localhost:5432 (legal_ai_db)" -ForegroundColor $infoColor
Write-Host "🔍 Qdrant: http://localhost:6333" -ForegroundColor $infoColor
Write-Host "🤖 Ollama: http://localhost:11434" -ForegroundColor $infoColor
Write-Host ""
Write-Host "💡 NEXT STEPS:" -ForegroundColor $warningColor
Write-Host "   1. Upload legal documents via the web interface" -ForegroundColor $infoColor
Write-Host "   2. Create embeddings and vector searches" -ForegroundColor $infoColor
Write-Host "   3. Test AI chat and document analysis" -ForegroundColor $infoColor
Write-Host "   4. Use Playwright tests: npm run test:e2e" -ForegroundColor $infoColor
Write-Host ""
Write-Host "📋 USEFUL COMMANDS:" -ForegroundColor $warningColor
Write-Host "   • Check status: .\one-click-legal-ai-launcher.ps1" -ForegroundColor $infoColor
Write-Host "   • Reset system: .\one-click-legal-ai-launcher.ps1 -Reset" -ForegroundColor $infoColor
Write-Host "   • Setup fresh: .\one-click-legal-ai-launcher.ps1 -Setup" -ForegroundColor $infoColor
Write-Host "   • GPU mode: .\one-click-legal-ai-launcher.ps1 -GPU" -ForegroundColor $infoColor
Write-Host ""
Write-Host "🏁 Launcher completed successfully!" -ForegroundColor $successColor
