# Enhanced Legal AI Complete Setup & Launch Script
# Run this from your sveltekit-frontend directory

param(
    [switch]$FullSetup,
    [switch]$QuickStart,
    [switch]$TestOnly,
    [switch]$Help
)

$ErrorActionPreference = "Continue"

function Write-Status {
    param($Message, $Type = "Info")
    switch ($Type) {
        "Success" { Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
        "Warning" { Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
        "Error"   { Write-Host "[ERROR] $Message" -ForegroundColor Red }
        "Info"    { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
        "Title"   { Write-Host "`n[TITLE] $Message" -ForegroundColor Magenta }
    }
}

function Show-Help {
    Write-Host @"
Enhanced Legal AI Setup and Launch Script
==========================================

Usage:
  .\launch-enhanced-legal-ai.ps1 [options]

Options:
  -FullSetup     Complete setup including database, AI models, and dependencies
  -QuickStart    Quick launch for development (assumes setup is done)
  -TestOnly      Run all tests without starting the application
  -Help          Show this help message

Examples:
  .\launch-enhanced-legal-ai.ps1 -FullSetup     # First time setup
  .\launch-enhanced-legal-ai.ps1 -QuickStart    # Quick development start
  .\launch-enhanced-legal-ai.ps1 -TestOnly      # Run tests only

"@ -ForegroundColor White
    exit 0
}

if ($Help) { Show-Help }

Write-Status "Enhanced Legal AI Setup and Launch System" "Title"
Write-Host "================================================`n" -ForegroundColor Magenta

# Check if we're in the right directory
if (-not (Test-Path "package.json")) {
    Write-Status "Please run this script from the sveltekit-frontend directory" "Error"
    exit 1
}

# Verify current setup
Write-Status "Verifying current setup..." "Info"

$requiredFiles = @(
    "src/routes/api/analyze/+server.ts",
    "src/lib/components/ai/ThinkingStyleToggle.svelte",
    "src/lib/ai/thinking-processor.ts",
    "scripts/setup-thinking-ai.js",
    "scripts/test-thinking-analysis.js",
    "scripts/process-docs.js",
    "scripts/fetch-docs.js"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Status "Missing required files:" "Error"
    foreach ($file in $missingFiles) {
        Write-Host "  - $file" -ForegroundColor Red
    }
    exit 1
} else {
    Write-Status "All required files present" "Success"
}

# Check package.json
try {
    $packageJson = Get-Content "package.json" | ConvertFrom-Json
    $requiredDeps = @("jsdom", "sharp", "tesseract.js", "mammoth", "ollama")
    $missingDeps = @()

    foreach ($dep in $requiredDeps) {
        if (-not $packageJson.dependencies.$dep) {
            $missingDeps += $dep
        }
    }

    if ($missingDeps.Count -gt 0) {
        Write-Status "Missing dependencies: $($missingDeps -join ', ')" "Warning"
        Write-Status "Run 'npm install' to install missing dependencies" "Info"
    } else {
        Write-Status "All required dependencies present" "Success"
    }
} catch {
    Write-Status "Could not read package.json" "Warning"
}

# Full Setup Mode
if ($FullSetup) {
    Write-Status "Starting Full Setup..." "Title"
    
    # Install/update dependencies
    Write-Status "Installing/updating dependencies..." "Info"
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Failed to install dependencies" "Error"
        exit 1
    }
    Write-Status "Dependencies installed successfully" "Success"
    
    # Setup database
    Write-Status "Setting up database..." "Info"
    npm run db:push
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Database setup failed - continuing anyway" "Warning"
    } else {
        Write-Status "Database setup completed" "Success"
    }
    
    # Setup AI features
    Write-Status "Setting up AI features..." "Info"
    npm run thinking:setup
    if ($LASTEXITCODE -ne 0) {
        Write-Status "AI setup failed - continuing anyway" "Warning"
    } else {
        Write-Status "AI features setup completed" "Success"
    }
    
    # Process legal documents
    Write-Status "Processing legal documents..." "Info"
    npm run docs:process
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Document processing failed - continuing anyway" "Warning"
    } else {
        Write-Status "Legal documents processed" "Success"
    }
    
    # Setup vector search
    Write-Status "Setting up vector search..." "Info"
    npm run vector:init
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Vector search setup failed - continuing anyway" "Warning"
    } else {
        Write-Status "Vector search setup completed" "Success"
    }
}

# Test Mode
if ($TestOnly -or $FullSetup) {
    Write-Status "Running Tests..." "Title"
    
    # Run thinking analysis tests
    Write-Status "Testing thinking analysis..." "Info"
    npm run thinking:test
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Thinking analysis tests failed" "Warning"
    } else {
        Write-Status "Thinking analysis tests passed" "Success"
    }
    
    # Run AI integration tests
    Write-Status "Testing AI integration..." "Info"
    npm run ai:test
    if ($LASTEXITCODE -ne 0) {
        Write-Status "AI integration tests failed" "Warning"
    } else {
        Write-Status "AI integration tests passed" "Success"
    }
    
    # Run system health check
    Write-Status "Running system health check..." "Info"
    npm run system:health
    if ($LASTEXITCODE -ne 0) {
        Write-Status "System health check failed" "Warning"
    } else {
        Write-Status "System health check passed" "Success"
    }
    
    if ($TestOnly) {
        Write-Status "Tests completed" "Success"
        exit 0
    }
}

# Quick Start or after Full Setup
Write-Status "Starting Enhanced Legal AI Application..." "Title"

# Check if Ollama is running
Write-Status "Checking Ollama service..." "Info"
$ollamaRunning = $false
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    $ollamaRunning = $true
    Write-Status "Ollama is running" "Success"
} catch {
    Write-Status "Ollama is not running - starting it..." "Warning"
    try {
        Start-Process "ollama" -ArgumentList "serve" -NoNewWindow
        Start-Sleep 5
        Write-Status "Ollama started" "Success"
    } catch {
        Write-Status "Could not start Ollama automatically" "Warning"
        Write-Status "Please start Ollama manually: ollama serve" "Info"
    }
}

# Check for Gemma3 model
if ($ollamaRunning) {
    try {
        $models = Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
        $hasGemma = $models.models | Where-Object { $_.name -like "*gemma3*" }
        if ($hasGemma) {
            Write-Status "Gemma3 model available" "Success"
        } else {
            Write-Status "Gemma3 model not found - you may need to run 'ollama pull gemma3:7b'" "Warning"
        }
    } catch {
        Write-Status "Could not check available models" "Warning"
    }
}

# Start the application
Write-Status "Starting development server..." "Info"

# Create environment-specific launch
$env:NODE_ENV = "development"

# Check if we should start with enhanced features
if (Test-Path ".env.development") {
    Write-Status "Using development environment configuration" "Info"
} else {
    Write-Status "Using default configuration" "Info"
}

Write-Host @"

Enhanced Legal AI Application Started!
========================================

Application Features:
* Thinking Style AI Analysis
* Document Processing and OCR
* Vector Search and Embeddings
* Evidence Analysis and Classification
* Legal Compliance Checking
* Interactive Evidence Canvas
* Chain of Custody Verification

Access Points:
* Main App: http://localhost:5173
* Evidence Analysis: http://localhost:5173/evidence
* Interactive Canvas: http://localhost:5173/interactive-canvas
* AI Assistant: http://localhost:5173/ai-assistant
* Cases Management: http://localhost:5173/cases

API Endpoints:
* Document Analysis: http://localhost:5173/api/analyze
* AI Chat: http://localhost:5173/api/ai/chat
* Vector Search: http://localhost:5173/api/search/vector

Troubleshooting:
* If Ollama models are missing: ollama pull gemma3:7b
* If database issues: npm run db:push
* If AI features fail: npm run thinking:setup
* For full diagnostics: npm run system:health

Happy Legal AI Analysis!

"@ -ForegroundColor Green

# Start the development server
npm run dev
