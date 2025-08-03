#!/usr/bin/env powershell

# SvelteKit + Ollama Development Starter
# Comprehensive script to start the legal AI development environment

param(
    [switch]$GPU,
    [switch]$Quick,
    [switch]$Production,
    [switch]$SkipDocker,
    [string]$Model = "gemma3-legal"
)

# Colors for output
function Write-Header {
    param($message)
    Write-Host "`n🚀 $message" -ForegroundColor Cyan
    Write-Host ("=" * ($message.Length + 3)) -ForegroundColor Cyan
}

function Write-Success {
    param($message)
    Write-Host "✅ $message" -ForegroundColor Green
}

function Write-Warning {
    param($message)
    Write-Host "⚠️  $message" -ForegroundColor Yellow
}

function Write-Error {
    param($message)
    Write-Host "❌ $message" -ForegroundColor Red
}

function Write-Info {
    param($message)
    Write-Host "ℹ️  $message" -ForegroundColor Blue
}

Write-Header "Legal AI Development Environment Starter"

if ($Quick) {
    Write-Info "Quick mode: Starting frontend only (assuming services are running)"
} elseif ($Production) {
    Write-Info "Production mode: Full deployment with optimizations"
} elseif ($GPU) {
    Write-Info "GPU mode: Enhanced performance with GPU acceleration"
} else {
    Write-Info "Development mode: Full local stack"
}

# Step 1: Check prerequisites
Write-Header "Checking Prerequisites"

$prerequisites = @(
    @{ Name = "Node.js"; Command = "node --version" },
    @{ Name = "NPM"; Command = "npm --version" },
    @{ Name = "Docker"; Command = "docker --version"; Required = !$SkipDocker }
)

foreach ($prereq in $prerequisites) {
    try {
        $version = & cmd /c $prereq.Command 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "$($prereq.Name): $version"
        } else {
            if ($prereq.Required) {
                Write-Error "$($prereq.Name) is required but not available"
                exit 1
            } else {
                Write-Warning "$($prereq.Name) not available (optional)"
            }
        }
    } catch {
        if ($prereq.Required) {
            Write-Error "$($prereq.Name) check failed"
            exit 1
        } else {
            Write-Warning "$($prereq.Name) not available (optional)"
        }
    }
}

# Step 2: Install dependencies
if (!$Quick) {
    Write-Header "Installing Dependencies"

    Write-Info "Installing root dependencies..."
    npm install

    if (Test-Path "sveltekit-frontend") {
        Write-Info "Installing SvelteKit frontend dependencies..."
        Set-Location "sveltekit-frontend"
        npm install
        Set-Location ".."
    }

    Write-Success "Dependencies installed"
}

# Step 3: Start Docker services (unless skipped or quick mode)
if (!$Quick -and !$SkipDocker) {
    Write-Header "Starting Docker Services"

    # Check if Docker is running
    try {
        docker ps | Out-Null
        Write-Success "Docker is running"

        if ($GPU) {
            Write-Info "Starting GPU-accelerated services..."
            docker-compose -f docker-compose.gpu.yml up -d
        } elseif ($Production) {
            Write-Info "Starting production services..."
            docker-compose -f docker-compose.production.yml up -d
        } else {
            Write-Info "Starting development services..."
            docker-compose up -d
        }

        Write-Success "Docker services started"

        # Wait for services to be ready
        Write-Info "Waiting for services to be ready..."
        Start-Sleep 10

    } catch {
        Write-Warning "Docker not available or not running"
        Write-Info "Continuing without Docker services..."
    }
}

# Step 4: Start Ollama service
Write-Header "Starting Ollama Service"

if ($GPU) {
    Write-Info "Starting Ollama with GPU acceleration..."
    & npm run ollama:gpu
} else {
    Write-Info "Starting Ollama service..."
    & npm run ollama:start
}

# Wait for Ollama to be ready
Write-Info "Waiting for Ollama to be ready..."
$ollamaReady = $false
$attempts = 0
$maxAttempts = 12

do {
    Start-Sleep 5
    $attempts++
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
        $ollamaReady = $true
        Write-Success "Ollama is ready!"
    } catch {
        Write-Info "Ollama not ready yet... (attempt $attempts/$maxAttempts)"
    }
} while (!$ollamaReady -and $attempts -lt $maxAttempts)

if (!$ollamaReady) {
    Write-Warning "Ollama service may not be fully ready, but continuing..."
}

# Step 5: Setup AI models (if not quick mode)
if (!$Quick) {
    Write-Header "Setting Up AI Models"

    Write-Info "Setting up legal AI models..."
    & npm run ollama:setup

    # Verify model availability
    try {
        $models = & npm run ollama:models --silent
        Write-Success "Available models: $models"
    } catch {
        Write-Warning "Could not verify model availability"
    }
}

# Step 6: Start SvelteKit development server
Write-Header "Starting SvelteKit Development Server"

Write-Info "Starting SvelteKit frontend on http://localhost:5173"
Write-Info "AI Demo available at: http://localhost:5173/ai-demo"

if (Test-Path "sveltekit-frontend") {
    Set-Location "sveltekit-frontend"
}

# Start the development server
Write-Success "🎉 Starting development server..."
Write-Host ""
Write-Host "📚 Quick Access URLs:" -ForegroundColor Yellow
Write-Host "   Frontend:     http://localhost:5173" -ForegroundColor White
Write-Host "   AI Demo:      http://localhost:5173/ai-demo" -ForegroundColor White
Write-Host "   API Health:   http://localhost:5173/api/ai/health" -ForegroundColor White
Write-Host "   Chat API:     http://localhost:5173/api/ai/chat" -ForegroundColor White
Write-Host ""
Write-Host "🛠️  Available Commands:" -ForegroundColor Yellow
Write-Host "   npm run ollama:health    - Check Ollama status" -ForegroundColor White
Write-Host "   npm run ollama:models    - List available models" -ForegroundColor White
Write-Host "   npm run ai:test          - Test AI integration" -ForegroundColor White
Write-Host "   npm run docker:status    - Check Docker containers" -ForegroundColor White
Write-Host ""

# Start development server
npm run dev
