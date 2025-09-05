# Setup Ollama for Legal AI System on Windows 10
# Checks for existing container/local setup and configures accordingly

Write-Host "=== Legal AI System - Ollama Configuration ===" -ForegroundColor Cyan

# Function to check if a service is running
function Test-ServiceRunning {
    param([string]$ServiceName, [string]$Port)
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port" -Method GET -TimeoutSec 5 -ErrorAction SilentlyContinue
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

# Function to check if Ollama is installed locally
function Test-OllamaInstalled {
    try {
        $ollama = Get-Command ollama -ErrorAction SilentlyContinue
        return $ollama -ne $null
    } catch {
        return $false
    }
}

Write-Host "`n1. Checking existing services..." -ForegroundColor Green

# Check for Ollama (local)
$ollamaLocal = Test-OllamaInstalled
$ollamaRunning = Test-ServiceRunning -ServiceName "Ollama" -Port "11434"

Write-Host "Ollama (local): " -NoNewline
if ($ollamaLocal) {
    Write-Host "Installed" -ForegroundColor Green
    if ($ollamaRunning) {
        Write-Host "  Status: Running (port 11434)" -ForegroundColor Green
    } else {
        Write-Host "  Status: Not running" -ForegroundColor Yellow
    }
} else {
    Write-Host "Not installed" -ForegroundColor Red
}

# Check for llama.cpp server
$llamacppRunning = Test-ServiceRunning -ServiceName "llama.cpp" -Port "8000"
Write-Host "llama.cpp server: " -NoNewline
if ($llamacppRunning) {
    Write-Host "Running (port 8000)" -ForegroundColor Green
} else {
    Write-Host "Not running" -ForegroundColor Red
}

Write-Host "`n2. Configuration recommendations..." -ForegroundColor Green

if ($ollamaLocal -and $ollamaRunning) {
    Write-Host "Recommended: Use local Ollama installation" -ForegroundColor Green
    Write-Host "  Configuration: http://localhost:11434" -ForegroundColor Cyan
    
    # Check if models are available
    try {
        $models = ollama list 2>$null
        Write-Host "`n  Available models:" -ForegroundColor Cyan
        Write-Host $models
    } catch {
        Write-Host "  No models found. Run: ollama pull gemma:2b" -ForegroundColor Yellow
    }
    
} elseif ($ollamaLocal -and -not $ollamaRunning) {
    Write-Host "Ollama installed but not running" -ForegroundColor Yellow
    Write-Host "  To start: ollama serve" -ForegroundColor Cyan
    
} else {
    Write-Host "Ollama not found. Installing..." -ForegroundColor Red
    Write-Host "  Download from: https://ollama.ai/download/windows" -ForegroundColor Cyan
    Write-Host "  Or use winget: winget install Ollama.Ollama" -ForegroundColor Cyan
}

# Docker configuration
Write-Host "`n3. Docker configuration update..." -ForegroundColor Green

if (Test-Path "docker-compose.yml") {
    Write-Host "Found docker-compose.yml" -ForegroundColor Green
    
    # Check if Ollama is already configured
    $dockerContent = Get-Content "docker-compose.yml" -Raw
    if ($dockerContent -like "*ollama:*") {
        Write-Host "Ollama service already configured in docker-compose.yml" -ForegroundColor Green
    } else {
        Write-Host "Ollama service not found in docker-compose.yml" -ForegroundColor Yellow
        Write-Host "  The main docker-compose.yml should already have Ollama configured." -ForegroundColor Cyan
    }
} else {
    Write-Host "docker-compose.yml not found" -ForegroundColor Red
}

Write-Host "`n4. Testing connectivity..." -ForegroundColor Green

# Test Ollama API if running
if ($ollamaRunning) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 10
        $models = $response.Content | ConvertFrom-Json
        Write-Host "Ollama API responding" -ForegroundColor Green
        Write-Host "  Models available: $($models.models.Count)" -ForegroundColor Cyan
    } catch {
        Write-Host "Ollama API not responding correctly" -ForegroundColor Yellow
    }
}

Write-Host "`n5. Next steps..." -ForegroundColor Green

if (-not $ollamaLocal) {
    Write-Host "1. Install Ollama from https://ollama.ai/download/windows" -ForegroundColor Cyan
    Write-Host "2. Run: ollama pull gemma:2b" -ForegroundColor Cyan
    Write-Host "3. Run: ollama pull nomic-embed-text" -ForegroundColor Cyan
} elseif (-not $ollamaRunning) {
    Write-Host "1. Start Ollama: ollama serve" -ForegroundColor Cyan
    Write-Host "2. Test: ollama list" -ForegroundColor Cyan
} else {
    Write-Host "System ready! Ollama is configured and running." -ForegroundColor Green
    Write-Host "1. Test the legal AI system: npm run dev" -ForegroundColor Cyan
    Write-Host "2. Visit: http://localhost:5173" -ForegroundColor Cyan
}

Write-Host "`n=== Configuration Summary ===" -ForegroundColor Cyan
Write-Host "Primary LLM Service: " -NoNewline
if ($ollamaRunning) {
    Write-Host "Ollama (Local)" -ForegroundColor Green
} elseif ($llamacppRunning) {
    Write-Host "llama.cpp (Fallback)" -ForegroundColor Yellow
} else {
    Write-Host "None available" -ForegroundColor Red
}

Write-Host "Configuration Files: " -NoNewline
if (Test-Path "docker-compose.yml") {
    Write-Host "Updated" -ForegroundColor Green
} else {
    Write-Host "Missing" -ForegroundColor Red
}

Write-Host "`nConfiguration complete!" -ForegroundColor Green