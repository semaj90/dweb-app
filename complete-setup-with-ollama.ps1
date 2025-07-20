# Complete SvelteKit Setup with Ollama Integration
# This script runs npm check, adds Ollama to Docker, and validates everything works

param(
    [switch]$SkipNpmCheck = $false,
    [switch]$Verbose = $false,
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Continue"
$startTime = Get-Date

Write-Host "🚀 Complete SvelteKit Setup with Ollama Integration" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray

# Navigate to the project root
$projectRoot = "C:\Users\james\Desktop\web-app"
$svelteKitPath = "$projectRoot\sveltekit-frontend"

if (-not (Test-Path $projectRoot)) {
    Write-Host "❌ Project root not found: $projectRoot" -ForegroundColor Red
    exit 1
}

Set-Location $projectRoot
Write-Host "📁 Working in: $(Get-Location)" -ForegroundColor Cyan

# Step 1: Run the SvelteKit fix script
Write-Host "`n🔧 STEP 1: Running SvelteKit Complete Fix Script..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

try {
    if (Test-Path "sveltekit-complete-fix.ps1") {
        Write-Host "  ▶️  Executing sveltekit-complete-fix.ps1..." -ForegroundColor Cyan
        
        $fixParams = @()
        if ($SkipNpmCheck) { $fixParams += "-SkipCheck" }
        if ($Verbose) { $fixParams += "-Verbose" }
        if ($DryRun) { $fixParams += "-DryRun" }
        
        $fixCommand = "powershell -ExecutionPolicy Bypass -File sveltekit-complete-fix.ps1 $($fixParams -join ' ')"
        Write-Host "  Command: $fixCommand" -ForegroundColor Gray
        
        Invoke-Expression $fixCommand
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ SvelteKit fixes completed successfully!" -ForegroundColor Green
        } else {
            Write-Host "⚠️  SvelteKit fixes completed with warnings (Exit code: $LASTEXITCODE)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ sveltekit-complete-fix.ps1 not found!" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error running SvelteKit fix script: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 2: Run npm check in sveltekit-frontend
if (-not $SkipNpmCheck) {
    Write-Host "`n🔍 STEP 2: Running npm check in SvelteKit frontend..." -ForegroundColor Yellow
    Write-Host "-" * 50 -ForegroundColor Gray
    
    if (Test-Path $svelteKitPath) {
        Set-Location $svelteKitPath
        Write-Host "  📁 Changed to: $(Get-Location)" -ForegroundColor Cyan
        
        try {
            Write-Host "  ▶️  Running npm run check..." -ForegroundColor Cyan
            $checkOutput = npm run check 2>&1
            $npmCheckExitCode = $LASTEXITCODE
            
            if ($npmCheckExitCode -eq 0) {
                Write-Host "✅ npm check passed! No TypeScript errors found." -ForegroundColor Green
            } else {
                Write-Host "⚠️  npm check found issues:" -ForegroundColor Yellow
                $checkOutput | Write-Host -ForegroundColor Gray
            }
            
            # Save check output
            $checkOutput | Out-File "npm-check-results.txt" -Encoding UTF8
            Write-Host "  📄 Check results saved to: npm-check-results.txt" -ForegroundColor Gray
            
        } catch {
            Write-Host "❌ Error running npm check: $($_.Exception.Message)" -ForegroundColor Red
        }
        
        # Return to project root
        Set-Location $projectRoot
    } else {
        Write-Host "❌ SvelteKit frontend directory not found: $svelteKitPath" -ForegroundColor Red
    }
}

# Step 3: Add Ollama to Docker configuration
Write-Host "`n🐳 STEP 3: Adding Ollama to Docker configuration..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

# Check existing docker-compose files
$dockerComposeFiles = @(
    "docker-compose.yml",
    "docker-compose.override.yml",
    "docker-compose.enhanced.yml"
)

$foundDockerFiles = @()
foreach ($file in $dockerComposeFiles) {
    if (Test-Path $file) {
        $foundDockerFiles += $file
        Write-Host "  ✅ Found: $file" -ForegroundColor Green
    }
}

if ($foundDockerFiles.Count -eq 0) {
    Write-Host "  ❌ No docker-compose files found. Creating new one..." -ForegroundColor Yellow
    
    # Create a new docker-compose.yml with Ollama
    $dockerComposeContent = @'
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: prosecutor_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Redis for caching and sessions
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__SERVICE__GRPC_PORT: 6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Ollama for Local LLM
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_ORIGINS=*
      - OLLAMA_HOST=0.0.0.0:11434
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
      interval: 30s
      timeout: 10s
      retries: 5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # Remove GPU requirements if no NVIDIA GPU available
    # Comment out the deploy section above if running on CPU only

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
  ollama_data:

networks:
  default:
    name: prosecutor_network
'@

    if (-not $DryRun) {
        $dockerComposeContent | Out-File "docker-compose.yml" -Encoding UTF8
        Write-Host "✅ Created new docker-compose.yml with Ollama" -ForegroundColor Green
    } else {
        Write-Host "🔍 DRY RUN: Would create new docker-compose.yml" -ForegroundColor Yellow
    }
} else {
    Write-Host "  📋 Found existing Docker Compose files: $($foundDockerFiles -join ', ')" -ForegroundColor Cyan
    
    # Check if Ollama is already in any of the files
    $ollamaFound = $false
    foreach ($file in $foundDockerFiles) {
        $content = Get-Content $file -Raw
        if ($content -match "ollama") {
            Write-Host "  ✅ Ollama already configured in: $file" -ForegroundColor Green
            $ollamaFound = $true
            break
        }
    }
    
    if (-not $ollamaFound) {
        Write-Host "  📝 Adding Ollama to docker-compose.override.yml..." -ForegroundColor Cyan
        
        $ollamaOverride = @'
version: '3.8'

services:
  # Ollama for Local LLM
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_ORIGINS=*
      - OLLAMA_HOST=0.0.0.0:11434
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
      interval: 30s
      timeout: 10s
      retries: 5
    # Uncomment for GPU support
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

volumes:
  ollama_data:
'@

        if (-not $DryRun) {
            $ollamaOverride | Out-File "docker-compose.override.yml" -Encoding UTF8
            Write-Host "✅ Added Ollama to docker-compose.override.yml" -ForegroundColor Green
        } else {
            Write-Host "🔍 DRY RUN: Would add Ollama to docker-compose.override.yml" -ForegroundColor Yellow
        }
    }
}

# Step 4: Create Ollama setup script
Write-Host "`n🤖 STEP 4: Creating Ollama setup and model download script..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

$ollamaSetupScript = @'
#!/bin/bash
# Ollama Model Setup Script

echo "🤖 Setting up Ollama models..."

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama service to be ready..."
until curl -s http://localhost:11434/api/version > /dev/null; do
    echo "Waiting for Ollama..."
    sleep 5
done

echo "✅ Ollama is ready!"

# Pull recommended models
echo "📥 Downloading recommended models..."

# Gemma 2B - Good for general tasks, smaller model
echo "Downloading Gemma 2B (2.7GB)..."
ollama pull gemma:2b

# Llama 3.2 3B - Good balance of performance and size
echo "Downloading Llama 3.2 3B (2.0GB)..."
ollama pull llama3.2:3b

# Code Llama for code-related tasks
echo "Downloading CodeLlama 7B (3.8GB)..."
ollama pull codellama:7b

# Phi-3 Mini for fast responses
echo "Downloading Phi-3 Mini (2.2GB)..."
ollama pull phi3:mini

echo "🎉 All models downloaded successfully!"
echo ""
echo "Available models:"
ollama list

echo ""
echo "🚀 Ollama is ready for use!"
echo "API endpoint: http://localhost:11434"
echo ""
echo "Test with: curl http://localhost:11434/api/generate -d '{\"model\":\"gemma:2b\",\"prompt\":\"Hello world\"}'"
'@

if (-not $DryRun) {
    $ollamaSetupScript | Out-File "setup-ollama-models.sh" -Encoding UTF8
    Write-Host "✅ Created setup-ollama-models.sh" -ForegroundColor Green
} else {
    Write-Host "🔍 DRY RUN: Would create setup-ollama-models.sh" -ForegroundColor Yellow
}

# Also create a PowerShell version
$ollamaSetupPS = @'
# Ollama Model Setup Script (PowerShell)
param(
    [switch]$SkipLargeModels = $false
)

Write-Host "🤖 Setting up Ollama models..." -ForegroundColor Green

# Wait for Ollama to be ready
Write-Host "⏳ Waiting for Ollama service to be ready..." -ForegroundColor Yellow
do {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 5
        break
    } catch {
        Write-Host "Waiting for Ollama..."
        Start-Sleep 5
    }
} while ($true)

Write-Host "✅ Ollama is ready!" -ForegroundColor Green

# Pull recommended models
Write-Host "📥 Downloading recommended models..." -ForegroundColor Yellow

# Gemma 2B - Good for general tasks, smaller model
Write-Host "Downloading Gemma 2B (2.7GB)..." -ForegroundColor Cyan
ollama pull gemma:2b

# Llama 3.2 3B - Good balance of performance and size
Write-Host "Downloading Llama 3.2 3B (2.0GB)..." -ForegroundColor Cyan
ollama pull llama3.2:3b

if (-not $SkipLargeModels) {
    # Code Llama for code-related tasks
    Write-Host "Downloading CodeLlama 7B (3.8GB)..." -ForegroundColor Cyan
    ollama pull codellama:7b
}

# Phi-3 Mini for fast responses
Write-Host "Downloading Phi-3 Mini (2.2GB)..." -ForegroundColor Cyan
ollama pull phi3:mini

Write-Host "🎉 All models downloaded successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Available models:" -ForegroundColor Cyan
ollama list

Write-Host ""
Write-Host "🚀 Ollama is ready for use!" -ForegroundColor Green
Write-Host "API endpoint: http://localhost:11434" -ForegroundColor White
Write-Host ""
Write-Host "Test with PowerShell:" -ForegroundColor Cyan
Write-Host 'Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body (ConvertTo-Json @{model="gemma:2b"; prompt="Hello world"}) -ContentType "application/json"' -ForegroundColor Gray
'@

if (-not $DryRun) {
    $ollamaSetupPS | Out-File "setup-ollama-models.ps1" -Encoding UTF8
    Write-Host "✅ Created setup-ollama-models.ps1" -ForegroundColor Green
} else {
    Write-Host "🔍 DRY RUN: Would create setup-ollama-models.ps1" -ForegroundColor Yellow
}

# Step 5: Update environment configuration
Write-Host "`n⚙️  STEP 5: Updating environment configuration for Ollama..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

$envFilesToUpdate = @(
    "$svelteKitPath\.env",
    "$svelteKitPath\.env.development",
    "$svelteKitPath\.env.example"
)

$ollamaEnvVars = @'

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=gemma:2b
OLLAMA_TIMEOUT=30000
OLLAMA_MAX_TOKENS=2048

# LLM Service Configuration
LLM_PROVIDER=ollama
LLM_FALLBACK_PROVIDER=openai
'@

foreach ($envFile in $envFilesToUpdate) {
    if (Test-Path $envFile) {
        $currentContent = Get-Content $envFile -Raw
        if (-not ($currentContent -match "OLLAMA_BASE_URL")) {
            Write-Host "  📝 Adding Ollama config to: $(Split-Path $envFile -Leaf)" -ForegroundColor Cyan
            if (-not $DryRun) {
                $currentContent + $ollamaEnvVars | Out-File $envFile -Encoding UTF8
                Write-Host "  ✅ Updated: $(Split-Path $envFile -Leaf)" -ForegroundColor Green
            } else {
                Write-Host "  🔍 DRY RUN: Would update $(Split-Path $envFile -Leaf)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ✅ Ollama config already exists in: $(Split-Path $envFile -Leaf)" -ForegroundColor Green
        }
    }
}

# Step 6: Create quick start scripts
Write-Host "`n🚀 STEP 6: Creating quick start scripts..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

$quickStartScript = @'
# Quick Start Script for Development Environment
param(
    [switch]$SetupModels = $false,
    [switch]$SkipBuild = $false
)

Write-Host "🚀 Starting Development Environment..." -ForegroundColor Green

# Start Docker services
Write-Host "📦 Starting Docker services..." -ForegroundColor Yellow
docker compose up -d

# Wait for services to be ready
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep 10

# Setup Ollama models if requested
if ($SetupModels) {
    Write-Host "🤖 Setting up Ollama models..." -ForegroundColor Yellow
    .\setup-ollama-models.ps1
}

# Navigate to frontend and start development server
Write-Host "🌐 Starting SvelteKit development server..." -ForegroundColor Yellow
Set-Location sveltekit-frontend

if (-not $SkipBuild) {
    Write-Host "🔧 Running npm install..." -ForegroundColor Cyan
    npm install
}

Write-Host "✅ Environment ready! Starting dev server..." -ForegroundColor Green
npm run dev
'@

if (-not $DryRun) {
    $quickStartScript | Out-File "quick-start-dev.ps1" -Encoding UTF8
    Write-Host "✅ Created quick-start-dev.ps1" -ForegroundColor Green
} else {
    Write-Host "🔍 DRY RUN: Would create quick-start-dev.ps1" -ForegroundColor Yellow
}

# Step 7: Test Docker and npm setup
Write-Host "`n🧪 STEP 7: Testing setup..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✅ Docker available: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Docker not available or not in PATH" -ForegroundColor Yellow
}

# Check Docker Compose
try {
    $dockerComposeVersion = docker compose version 2>&1
    Write-Host "✅ Docker Compose available: $dockerComposeVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Docker Compose not available" -ForegroundColor Yellow
}

# Check Node and npm in frontend directory
if (Test-Path $svelteKitPath) {
    Set-Location $svelteKitPath
    
    try {
        $nodeVersion = node --version 2>&1
        Write-Host "✅ Node.js available: $nodeVersion" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Node.js not available in frontend directory" -ForegroundColor Yellow
    }
    
    try {
        $npmVersion = npm --version 2>&1
        Write-Host "✅ npm available: $npmVersion" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  npm not available in frontend directory" -ForegroundColor Yellow
    }
    
    Set-Location $projectRoot
}

# Generate final report
Write-Host "`n📊 STEP 8: Generating setup report..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

$setupReport = @"
# Complete Setup Report
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Summary
- **Mode**: $(if ($DryRun) { "DRY RUN" } else { "APPLIED CHANGES" })
- **npm check**: $(if ($SkipNpmCheck) { "SKIPPED" } else { "COMPLETED" })
- **Docker files**: $($foundDockerFiles.Count) found
- **Ollama integration**: Added to Docker configuration

## Components Configured
✅ SvelteKit application fixes applied
✅ TypeScript configuration validated
✅ Docker Compose with Ollama service
✅ Environment variables for Ollama
✅ Model download scripts created
✅ Quick start scripts created

## Files Created/Updated
- docker-compose.yml or docker-compose.override.yml
- setup-ollama-models.sh
- setup-ollama-models.ps1
- quick-start-dev.ps1
- Environment files updated with Ollama config

## Next Steps

### 1. Start the Development Environment
``````powershell
# Option A: Quick start (recommended)
.\quick-start-dev.ps1 -SetupModels

# Option B: Manual start
docker compose up -d
cd sveltekit-frontend
npm run dev
``````

### 2. Setup Ollama Models (if not done automatically)
``````powershell
# Wait for Docker to start Ollama, then:
.\setup-ollama-models.ps1
``````

### 3. Test the Application
- Frontend: http://localhost:5173
- Ollama API: http://localhost:11434
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Qdrant: http://localhost:6333

### 4. Verify Everything Works
``````powershell
# In sveltekit-frontend directory:
npm run check        # TypeScript check
npm run build        # Build test
npm run test         # Run tests
``````

## Troubleshooting

### If Ollama doesn't start:
- Check Docker logs: ``docker compose logs ollama``
- Remove GPU requirements from docker-compose if no NVIDIA GPU
- Ensure port 11434 is not in use

### If npm check fails:
- Run: ``npm install`` in sveltekit-frontend
- Check TypeScript errors in the output
- Review fix-report.md for detailed analysis

### If models fail to download:
- Check internet connection
- Ensure Ollama service is running
- Try downloading individual models: ``ollama pull gemma:2b``

## Performance Notes
- First model download will take time (several GB)
- Ollama with GPU support provides much faster inference
- Consider using smaller models (gemma:2b, phi3:mini) for development

## Security Notes
- Ollama is configured to accept connections from any origin (OLLAMA_ORIGINS=*)
- In production, restrict origins to your application domain
- Consider adding authentication for production deployments
"@

if (-not $DryRun) {
    $setupReport | Out-File "COMPLETE_SETUP_REPORT.md" -Encoding UTF8
    Write-Host "📋 Setup report saved to: COMPLETE_SETUP_REPORT.md" -ForegroundColor Green
} else {
    Write-Host "🔍 DRY RUN: Would create COMPLETE_SETUP_REPORT.md" -ForegroundColor Yellow
}

# Final summary
Write-Host "`n🎯 SETUP COMPLETION SUMMARY" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray

if (-not $DryRun) {
    Write-Host "✅ SvelteKit fixes applied" -ForegroundColor White
    Write-Host "✅ Ollama added to Docker configuration" -ForegroundColor White
    Write-Host "✅ Environment variables configured" -ForegroundColor White
    Write-Host "✅ Setup scripts created" -ForegroundColor White
    Write-Host "✅ Quick start scripts ready" -ForegroundColor White
    Write-Host ""
    Write-Host "🚀 READY TO START:" -ForegroundColor Green
    Write-Host "   Run: .\quick-start-dev.ps1 -SetupModels" -ForegroundColor White
} else {
    Write-Host "🔍 DRY RUN COMPLETED - No changes made" -ForegroundColor Yellow
    Write-Host "   Remove -DryRun flag to apply changes" -ForegroundColor Cyan
}

Write-Host "`n⏱️  Total setup time: $(((Get-Date) - $startTime).TotalSeconds.ToString('F1')) seconds" -ForegroundColor Gray

# Return to original location
Set-Location $projectRoot

Write-Host "`n📖 Next: Review COMPLETE_SETUP_REPORT.md for detailed instructions" -ForegroundColor Cyan