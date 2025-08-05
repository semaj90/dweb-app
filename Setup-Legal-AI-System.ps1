# Setup-Legal-AI-System.ps1
# Complete setup script that creates all necessary files with error checking

param(
    [switch]$Force,
    [switch]$SkipDependencyCheck
)

$ErrorActionPreference = "Stop"
$script:ErrorCount = 0
$script:WarningCount = 0
$script:CreatedFiles = @()

# Color output functions with simple characters
function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red; $script:ErrorCount++ }
function Write-Warning { Write-Host "[WARN] $args" -ForegroundColor Yellow; $script:WarningCount++ }
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Header { 
    Write-Host "`n==========================================================" -ForegroundColor Cyan
    Write-Host " $args" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
}

# Get project root
$ProjectRoot = Get-Location
Write-Header "LEGAL AI SYSTEM SETUP"
Write-Info "Project Root: $ProjectRoot"

# Check if we're in the right directory
if (-not (Test-Path "package.json")) {
    Write-Error "package.json not found. Are you in the deeds-web-app directory?"
    Write-Info "Expected location: C:\Users\james\Desktop\deeds-web\deeds-web-app"
    exit 1
}

# Function to safely create directories
function New-SafeDirectory {
    param([string]$Path)
    
    try {
        if (-not (Test-Path $Path)) {
            New-Item -ItemType Directory -Path $Path -Force | Out-Null
            Write-Success "Created directory: $Path"
        } else {
            Write-Info "Directory exists: $Path"
        }
        return $true
    } catch {
        Write-Error "Failed to create directory: $Path - $_"
        return $false
    }
}

# Function to safely create files
function New-SafeFile {
    param(
        [string]$Path,
        [string]$Content,
        [switch]$Append
    )
    
    try {
        $directory = Split-Path -Parent $Path
        if ($directory -and -not (Test-Path $directory)) {
            New-SafeDirectory $directory | Out-Null
        }
        
        if ((Test-Path $Path) -and -not $Force) {
            Write-Warning "File exists: $Path (use -Force to overwrite)"
            return $false
        }
        
        if ($Append) {
            Add-Content -Path $Path -Value $Content -Encoding UTF8
        } else {
            Set-Content -Path $Path -Value $Content -Encoding UTF8
        }
        
        $script:CreatedFiles += $Path
        Write-Success "Created file: $Path"
        return $true
    } catch {
        Write-Error "Failed to create file: $Path - $_"
        return $false
    }
}

Write-Header "STEP 1: CREATING DIRECTORY STRUCTURE"

$directories = @(
    "workers",
    "database",
    "uploads",
    "logs",
    "src\routes\api\documents\upload",
    "src\routes\api\documents\[id]",
    "src\routes\api\jobs\[id]\status",
    "src\routes\api\search\vector",
    "src\routes\api\insights\[documentId]",
    "src\routes\api\notifications",
    "src\routes\api\tags\auto-suggest",
    "src\routes\api\health",
    "src\lib\workers",
    "src\lib\db",
    "src\lib\gpu"
)

foreach ($dir in $directories) {
    New-SafeDirectory (Join-Path $ProjectRoot $dir) | Out-Null
}

Write-Header "STEP 2: CHECKING DEPENDENCIES"

if (-not $SkipDependencyCheck) {
    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Success "Node.js installed: $nodeVersion"
    } catch {
        Write-Error "Node.js not found. Please install Node.js 18+"
    }

    # Check Go
    try {
        $goVersion = go version
        Write-Success "Go installed: $goVersion"
    } catch {
        Write-Error "Go not found. Please install Go 1.21+"
    }

    # Check PostgreSQL
    try {
        $pgVersion = psql --version 2>$null
        Write-Success "PostgreSQL installed: $pgVersion"
    } catch {
        Write-Warning "PostgreSQL psql not found in PATH"
    }

    # Check Redis
    if (Test-Path "redis-windows\redis-cli.exe") {
        Write-Success "Redis for Windows found"
    } else {
        Write-Error "Redis for Windows not found in redis-windows directory"
    }

    # Check Ollama
    try {
        $ollamaVersion = ollama --version 2>$null
        Write-Success "Ollama installed: $ollamaVersion"
        
        # Check for gemma3-legal model
        $models = ollama list 2>$null
        if ($models -match "gemma3-legal") {
            Write-Success "gemma3-legal model found"
        } else {
            Write-Warning "gemma3-legal model not found - please install your local model"
        }
    } catch {
        Write-Error "Ollama not found. Please install Ollama"
    }

    # Check CUDA
    if (Test-Path "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe") {
        Write-Success "NVIDIA CUDA found"
    } else {
        Write-Warning "NVIDIA CUDA not found - GPU acceleration disabled"
    }
}

Write-Header "STEP 3: CREATING CONFIGURATION FILES"

# Create .env file
$envContent = @"
# Database
DATABASE_URL=postgres://postgres:password@localhost/legal_ai
REDIS_URL=redis://localhost:6379

# Services
GO_SERVER_URL=http://localhost:8081
OLLAMA_URL=http://localhost:11434

# Models
LEGAL_MODEL=gemma3-legal
EMBEDDING_MODEL=nomic-embed-text

# Storage
UPLOAD_DIR=./uploads

# Environment
NODE_ENV=development
"@

New-SafeFile -Path ".env" -Content $envContent

# Create SET-LEGAL-AI-ENV.bat
$envBatContent = @'
@echo off
rem =============================================================================
rem SET-LEGAL-AI-ENV.bat
rem Configure environment variables for local AI models
rem =============================================================================

echo Setting Legal AI environment variables...

rem Model Configuration
set LEGAL_MODEL=gemma3-legal
set EMBEDDING_MODEL=nomic-embed-text
set VISION_MODEL=llava:7b

rem Service URLs
set OLLAMA_URL=http://localhost:11434
set GO_SERVER_URL=http://localhost:8081
set POSTGRES_URL=postgres://postgres:password@localhost/legal_ai?sslmode=disable
set REDIS_URL=localhost:6379

rem GPU Configuration
set CUDA_AVAILABLE=true
set GPU_MEMORY_LIMIT=12GB

rem Ollama Configuration
set OLLAMA_HOST=127.0.0.1:11434
set OLLAMA_KEEP_ALIVE=5m
set OLLAMA_NUM_PARALLEL=4

rem Processing Options
set ENABLE_GPU_ACCELERATION=true
set ENABLE_LEGAL_MODEL=true
set MODEL_TEMPERATURE=0.3
set MODEL_CONTEXT_SIZE=4096

echo.
echo Environment configured for local models:
echo   - Legal Model: %LEGAL_MODEL%
echo   - Embedding Model: %EMBEDDING_MODEL%
echo   - Ollama URL: %OLLAMA_URL%
echo   - GPU Acceleration: %ENABLE_GPU_ACCELERATION%
echo.
'@

New-SafeFile -Path "SET-LEGAL-AI-ENV.bat" -Content $envBatContent

Write-Header "STEP 4: CREATING VERIFICATION SCRIPTS"

# Create VERIFY-LOCAL-MODELS.bat
$verifyModelsContent = @'
@echo off
rem =============================================================================
rem VERIFY LOCAL OLLAMA MODELS
rem Quick verification that local models are ready
rem =============================================================================

echo.
echo Checking Ollama status...
echo.

rem Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Ollama not running. Starting...
    start "Ollama" ollama serve
    timeout /t 5 /nobreak >nul
)

rem List available models
echo Available local models:
echo ====================================
ollama list
echo ====================================
echo.

rem Quick test of gemma3-legal
echo Testing gemma3-legal model...
echo.
echo "Summarize: This is a test legal document." | ollama run gemma3-legal --verbose

echo.
echo ====================================
echo.
echo If gemma3-legal is listed above and responded correctly,
echo your system is ready to process legal documents!
echo.
echo The Go server expects:
echo   - Model name: gemma3-legal
echo   - Endpoint: http://localhost:11434
echo.
pause
'@

New-SafeFile -Path "VERIFY-LOCAL-MODELS.bat" -Content $verifyModelsContent

Write-Header "STEP 5: CREATING GO SERVER FILES"

# Check if go-microservice directory exists
if (-not (Test-Path "go-microservice")) {
    Write-Error "go-microservice directory not found"
    Write-Info "Creating go-microservice directory..."
    New-SafeDirectory "go-microservice"
}

# Create model-config.go
$modelConfigContent = @'
// model-config.go
// Add this file to your go-microservice directory

package main

import "os"

// ModelConfig defines the AI models to use
var ModelConfig = struct {
	// Primary legal analysis model (your local model)
	LegalModel string
	
	// Embedding model (if you have it locally)
	EmbeddingModel string
	
	// Vision model for image analysis (optional)
	VisionModel string
	
	// Model parameters
	Temperature float32
	MaxTokens   int
	NumCtx      int
}{
	LegalModel:     getEnv("LEGAL_MODEL", "gemma3-legal"),
	EmbeddingModel: getEnv("EMBEDDING_MODEL", "nomic-embed-text"),
	VisionModel:    getEnv("VISION_MODEL", "llava:7b"),
	Temperature:    0.3, // Lower temperature for more consistent legal analysis
	MaxTokens:      4096,
	NumCtx:         4096,
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
'@

New-SafeFile -Path "go-microservice\model-config.go" -Content $modelConfigContent

Write-Header "STEP 6: CREATING WORKER PACKAGE"

# Create workers package.json
$workersPackageJson = @'
{
  "name": "legal-ai-workers",
  "version": "1.0.0",
  "description": "BullMQ workers for Legal AI document processing",
  "main": "start-workers.js",
  "scripts": {
    "start": "node start-workers.js",
    "dev": "nodemon start-workers.js"
  },
  "dependencies": {
    "bullmq": "^5.1.0",
    "ioredis": "^5.3.2",
    "pdf-parse": "^1.1.1",
    "node-fetch": "^3.3.2",
    "form-data": "^4.0.0",
    "ws": "^8.16.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.2",
    "@types/node": "^20.10.5"
  }
}
'@

New-SafeFile -Path "workers\package.json" -Content $workersPackageJson

# Create start-workers.js
$startWorkersContent = @'
// workers/start-workers.js
const { spawn } = require('child_process');
const path = require('path');

const workers = [
  'document-processor.worker.js',
  'analysis.worker.js',
  'notification.worker.js'
];

console.log('Starting Legal AI Workers...');

workers.forEach(worker => {
  const workerPath = path.join(__dirname, worker);
  
  // Check if worker file exists
  if (!require('fs').existsSync(workerPath)) {
    console.error(`Worker file not found: ${workerPath}`);
    return;
  }
  
  const proc = spawn('node', [workerPath], {
    stdio: 'inherit',
    env: { ...process.env }
  });

  proc.on('error', (err) => {
    console.error(`Failed to start ${worker}:`, err);
  });

  console.log(`Started ${worker} with PID ${proc.pid}`);
});

process.on('SIGINT', () => {
  console.log('Shutting down workers...');
  process.exit(0);
});
'@

New-SafeFile -Path "workers\start-workers.js" -Content $startWorkersContent

Write-Header "STEP 7: CREATING DATABASE SCHEMA"

# Create PostgreSQL schema file
$schemaContent = @'
-- Legal AI Database Schema with pgvector
-- Run this after installing PostgreSQL and pgvector extension

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database if not exists (run as superuser)
-- CREATE DATABASE legal_ai;

-- Basic tables for initial setup
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(255) UNIQUE NOT NULL,
    case_id UUID,
    user_id UUID,
    file_name VARCHAR(500) NOT NULL,
    file_type VARCHAR(100),
    file_size BIGINT,
    file_path TEXT,
    content TEXT,
    summary TEXT,
    entities JSONB DEFAULT '[]',
    auto_tags JSONB DEFAULT '[]',
    user_tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(255) NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
    UNIQUE(document_id, chunk_index)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON document_embeddings 
    USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
'@

New-SafeFile -Path "database\schema.sql" -Content $schemaContent

Write-Header "STEP 8: CREATING STARTUP SCRIPTS"

# Create START-LEGAL-AI-COMPLETE.bat
$startCompleteContent = @'
@echo off
echo ====================================================================
echo   LEGAL AI COMPLETE SYSTEM STARTUP
echo   Using Local gemma3-legal Model
echo ====================================================================
echo.

REM Source environment configuration
call SET-LEGAL-AI-ENV.bat

REM Verify Ollama is running with local models
echo [CHECK] Verifying Ollama with gemma3-legal...
ollama list | findstr /i "gemma3-legal" >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] gemma3-legal model not found!
    echo Please ensure your local model is properly installed.
    pause
    exit /b 1
)

REM Start Ollama if not running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorLevel% neq 0 (
    echo Starting Ollama...
    start "Ollama" ollama serve
    timeout /t 5 /nobreak >nul
)

REM Check if main startup script exists
if not exist "START-LEGAL-AI.bat" (
    echo [ERROR] START-LEGAL-AI.bat not found!
    echo Please ensure you have the base startup script.
    pause
    exit /b 1
)

REM Start existing services
echo [START] Core services...
call START-LEGAL-AI.bat

REM Wait for services
timeout /t 10 /nobreak >nul

REM Start BullMQ workers
echo [START] BullMQ workers...
if exist "workers\start-workers.js" (
    cd workers
    call npm install --silent
    start "BullMQ Workers" cmd /c "node start-workers.js"
    cd ..
) else (
    echo [WARNING] Workers not found - skipping
)

echo.
echo ====================================================================
echo   ALL SERVICES STARTED WITH LOCAL MODELS
echo ====================================================================
echo.
echo   Using Models:
echo     - Legal Analysis: %LEGAL_MODEL%
echo     - Embeddings: %EMBEDDING_MODEL%
echo.
echo   Service URLs:
echo     - Frontend:    http://localhost:5173
echo     - Go Server:   http://localhost:8081
echo     - Ollama:      http://localhost:11434
echo     - Redis:       localhost:6379
echo     - PostgreSQL:  localhost:5432
echo.
echo   Upload a document at: http://localhost:5173/upload
echo.
pause
'@

New-SafeFile -Path "START-LEGAL-AI-COMPLETE.bat" -Content $startCompleteContent

Write-Header "STEP 9: CREATING PM2 CONFIGURATION"

$pm2ConfigContent = @'
module.exports = {
  apps: [
    {
      name: 'sveltekit',
      script: 'build/index.js',
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      }
    },
    {
      name: 'document-worker',
      script: './workers/document-processor.worker.js',
      instances: 2,
      exec_mode: 'cluster'
    },
    {
      name: 'analysis-worker',
      script: './workers/analysis.worker.js',
      instances: 1
    },
    {
      name: 'notification-worker',
      script: './workers/notification.worker.js',
      instances: 1
    }
  ]
};
'@

New-SafeFile -Path "ecosystem.config.js" -Content $pm2ConfigContent

Write-Header "STEP 10: VERIFYING SETUP"

# Check what was created
Write-Info "`nCreated $($script:CreatedFiles.Count) files"

# Run basic checks
$checks = @{
    "Directory Structure" = { Test-Path "workers" -and Test-Path "database" }
    "Go Model Config" = { Test-Path "go-microservice\model-config.go" }
    "Worker Package" = { Test-Path "workers\package.json" }
    "Environment Config" = { Test-Path ".env" -and Test-Path "SET-LEGAL-AI-ENV.bat" }
    "Startup Scripts" = { Test-Path "START-LEGAL-AI-COMPLETE.bat" }
    "Database Schema" = { Test-Path "database\schema.sql" }
}

$allPassed = $true
foreach ($check in $checks.Keys) {
    Write-Host -NoNewline "Checking $check... "
    if (& $checks[$check]) {
        Write-Success "OK"
    } else {
        Write-Error "FAILED"
        $allPassed = $false
    }
}

Write-Header "SETUP SUMMARY"

if ($script:ErrorCount -eq 0) {
    Write-Success "[SUCCESS] Setup completed successfully!"
    Write-Info "`nNext steps:"
    Write-Info "1. Install worker dependencies:"
    Write-Info "   cd workers && npm install"
    Write-Info ""
    Write-Info "2. Create PostgreSQL database:"
    Write-Info "   psql -U postgres -c 'CREATE DATABASE legal_ai;'"
    Write-Info "   psql -U postgres -d legal_ai -f database\schema.sql"
    Write-Info ""
    Write-Info "3. Update Go dependencies:"
    Write-Info "   cd go-microservice && go mod tidy"
    Write-Info ""
    Write-Info "4. Verify your gemma3-legal model:"
    Write-Info "   ollama list"
    Write-Info ""
    Write-Info "5. Start the complete system:"
    Write-Info "   START-LEGAL-AI-COMPLETE.bat"
} else {
    Write-Error "[FAILED] Setup completed with $($script:ErrorCount) errors"
    if ($script:WarningCount -gt 0) {
        Write-Warning "[WARNING] $($script:WarningCount) warnings"
    }
    Write-Info "`nPlease fix the errors above before proceeding."
}

# Save setup log
$logContent = @"
Legal AI System Setup Log
========================
Date: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Location: $ProjectRoot
Errors: $($script:ErrorCount)
Warnings: $($script:WarningCount)
Files Created: $($script:CreatedFiles.Count)

Created Files:
$($script:CreatedFiles -join "`n")
"@

New-SafeFile -Path "setup-log.txt" -Content $logContent

Write-Info "`nSetup log saved to: setup-log.txt"

# Create a quick test script
$testScriptContent = @'
@echo off
echo Running Legal AI System Quick Test...
echo.

echo [1/5] Checking PostgreSQL...
psql -U postgres -d legal_ai -c "SELECT version();" >nul 2>&1
if %errorLevel% equ 0 (echo OK) else (echo FAILED)

echo [2/5] Checking Redis...
redis-windows\redis-cli ping >nul 2>&1
if %errorLevel% equ 0 (echo OK) else (echo FAILED)

echo [3/5] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorLevel% equ 0 (echo OK) else (echo FAILED)

echo [4/5] Checking Go Server...
curl -s http://localhost:8081/health >nul 2>&1
if %errorLevel% equ 0 (echo OK) else (echo FAILED)

echo [5/5] Checking gemma3-legal model...
ollama list | findstr /i "gemma3-legal" >nul 2>&1
if %errorLevel% equ 0 (echo OK) else (echo FAILED)

echo.
pause
'@

New-SafeFile -Path "TEST-QUICK.bat" -Content $testScriptContent

Write-Success "`nCreated TEST-QUICK.bat for quick system verification"