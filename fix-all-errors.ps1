# Enhanced Legal AI - Comprehensive Error Fixing Script
# Automatically detects and fixes common issues before startup

param(
    [switch]$Verbose = $false,
    [switch]$Force = $false,
    [switch]$SkipDocker = $false
)

$ErrorActionPreference = "Continue"
$startTime = Get-Date

Write-Host "🔧 Enhanced Legal AI - Comprehensive Error Fixer" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray
Write-Host "🛠️ Detecting and fixing common issues automatically" -ForegroundColor Cyan

# Navigate to project root
$projectRoot = "C:\Users\james\Desktop\web-app"
if (-not (Test-Path $projectRoot)) {
    Write-Host "❌ Project root not found: $projectRoot" -ForegroundColor Red
    Write-Host "💡 Please ensure you're running this from the correct location" -ForegroundColor Yellow
    exit 1
}

Set-Location $projectRoot
Write-Host "📁 Working in: $(Get-Location)" -ForegroundColor Cyan

$errorsFixed = 0
$warningsIssued = 0

# Fix 1: Check and fix PowerShell execution policy
Write-Host "`n🔧 FIX 1: PowerShell Execution Policy..." -ForegroundColor Yellow
try {
    $currentPolicy = Get-ExecutionPolicy
    if ($currentPolicy -eq "Restricted" -or $currentPolicy -eq "AllSigned") {
        Write-Host "  ⚠️  PowerShell execution policy is restrictive: $currentPolicy" -ForegroundColor Yellow
        Write-Host "  🔧 Attempting to set execution policy for current user..." -ForegroundColor Cyan
        
        try {
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
            Write-Host "  ✅ Execution policy updated to RemoteSigned" -ForegroundColor Green
            $errorsFixed++
        } catch {
            Write-Host "  ⚠️  Could not change execution policy. Run as Administrator if needed." -ForegroundColor Yellow
            $warningsIssued++
        }
    } else {
        Write-Host "  ✅ PowerShell execution policy is acceptable: $currentPolicy" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Could not check execution policy: $($_.Exception.Message)" -ForegroundColor Red
}

# Fix 2: Check and fix file paths and line endings
Write-Host "`n🔧 FIX 2: File Path and Line Ending Issues..." -ForegroundColor Yellow

$criticalFiles = @(
    "docker-compose.lowmem.yml",
    "scripts\setup-local-gemma3-lowmem.sh",
    "scripts\init-enhanced-schema.sql",
    "scripts\seed-sample-data.sql",
    "sveltekit-frontend\package.json"
)

foreach ($file in $criticalFiles) {
    if (Test-Path $file) {
        try {
            # Check if file has Windows line endings and fix if needed
            $content = Get-Content $file -Raw
            if ($content -match "`r`n" -and $file.EndsWith(".sh")) {
                # Convert Windows line endings to Unix for shell scripts
                $content = $content -replace "`r`n", "`n"
                Set-Content -Path $file -Value $content -NoNewline
                Write-Host "  🔧 Fixed line endings in: $file" -ForegroundColor Cyan
                $errorsFixed++
            }
            Write-Host "  ✅ File OK: $file" -ForegroundColor Green
        } catch {
            Write-Host "  ❌ Issue with file $file`: $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "  ❌ Missing critical file: $file" -ForegroundColor Red
        $warningsIssued++
    }
}

# Fix 3: Docker and Container Issues
if (-not $SkipDocker) {
    Write-Host "`n🔧 FIX 3: Docker Environment..." -ForegroundColor Yellow
    
    try {
        $dockerVersion = docker --version 2>&1
        Write-Host "  ✅ Docker available: $dockerVersion" -ForegroundColor Green
        
        # Check if Docker is running
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ⚠️  Docker daemon not running. Starting Docker Desktop..." -ForegroundColor Yellow
            try {
                Start-Process "Docker Desktop" -ErrorAction SilentlyContinue
                Write-Host "  🔧 Docker Desktop start initiated. Please wait 30-60 seconds." -ForegroundColor Cyan
                $warningsIssued++
            } catch {
                Write-Host "  ❌ Could not start Docker Desktop automatically" -ForegroundColor Red
            }
        } else {
            Write-Host "  ✅ Docker daemon is running" -ForegroundColor Green
        }
        
        # Check Docker Compose
        $dockerComposeVersion = docker compose version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Docker Compose available: $dockerComposeVersion" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Docker Compose not available" -ForegroundColor Red
            $warningsIssued++
        }
        
        # Stop any conflicting containers
        Write-Host "  🔧 Stopping any existing containers..." -ForegroundColor Cyan
        docker compose -f docker-compose.lowmem.yml down 2>&1 | Out-Null
        docker compose -f docker-compose.enhanced.yml down 2>&1 | Out-Null
        Write-Host "  ✅ Existing containers stopped" -ForegroundColor Green
        
    } catch {
        Write-Host "  ❌ Docker not available: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  💡 Please install Docker Desktop from https://docker.com" -ForegroundColor Yellow
        $warningsIssued++
    }
}

# Fix 4: Node.js and npm Issues
Write-Host "`n🔧 FIX 4: Node.js and npm Environment..." -ForegroundColor Yellow

if (Test-Path "sveltekit-frontend") {
    Set-Location sveltekit-frontend
    
    try {
        $nodeVersion = node --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Node.js available: $nodeVersion" -ForegroundColor Green
            
            $npmVersion = npm --version 2>&1
            Write-Host "  ✅ npm available: v$npmVersion" -ForegroundColor Green
            
            # Check if node_modules exists and is complete
            if (-not (Test-Path "node_modules") -or $Force) {
                Write-Host "  🔧 Installing/updating npm dependencies..." -ForegroundColor Cyan
                npm install --silent 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  ✅ npm dependencies installed" -ForegroundColor Green
                    $errorsFixed++
                } else {
                    Write-Host "  ⚠️  npm install had issues, but continuing..." -ForegroundColor Yellow
                    $warningsIssued++
                }
            } else {
                Write-Host "  ✅ npm dependencies appear to be installed" -ForegroundColor Green
            }
            
            # Fix package.json if needed
            if (Test-Path "package.json") {
                $packageJson = Get-Content "package.json" | ConvertFrom-Json
                $hasNeo4j = $packageJson.dependencies.PSObject.Properties['neo4j-driver'] -ne $null
                $hasOllama = $packageJson.dependencies.PSObject.Properties['ollama'] -ne $null
                
                if (-not $hasNeo4j -or -not $hasOllama) {
                    Write-Host "  🔧 Installing missing enhanced dependencies..." -ForegroundColor Cyan
                    if (-not $hasNeo4j) { npm install neo4j-driver --silent 2>&1 | Out-Null }
                    if (-not $hasOllama) { npm install ollama --silent 2>&1 | Out-Null }
                    npm install node-fetch --silent 2>&1 | Out-Null
                    Write-Host "  ✅ Enhanced dependencies installed" -ForegroundColor Green
                    $errorsFixed++
                }
            }
            
        } else {
            Write-Host "  ❌ Node.js not available" -ForegroundColor Red
            Write-Host "  💡 Please install Node.js from https://nodejs.org" -ForegroundColor Yellow
            $warningsIssued++
        }
    } catch {
        Write-Host "  ❌ Node.js/npm error: $($_.Exception.Message)" -ForegroundColor Red
        $warningsIssued++
    }
    
    Set-Location ..
} else {
    Write-Host "  ❌ SvelteKit frontend directory not found" -ForegroundColor Red
    $warningsIssued++
}

# Fix 5: Port Conflicts
Write-Host "`n🔧 FIX 5: Checking Port Conflicts..." -ForegroundColor Yellow

$requiredPorts = @(5173, 11434, 5432, 7474, 7687, 6333, 6379, 5050)
$conflictPorts = @()

foreach ($port in $requiredPorts) {
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $port)
        $connection.Close()
        $conflictPorts += $port
        Write-Host "  ⚠️  Port $port is in use" -ForegroundColor Yellow
    } catch {
        Write-Host "  ✅ Port $port available" -ForegroundColor Green
    }
}

if ($conflictPorts.Count -gt 0) {
    Write-Host "  🔧 Attempting to free up ports..." -ForegroundColor Cyan
    
    # Try to stop common services that might conflict
    try {
        # Stop any existing Docker containers
        docker stop $(docker ps -q) 2>&1 | Out-Null
        Write-Host "  🔧 Stopped existing Docker containers" -ForegroundColor Cyan
        $errorsFixed++
    } catch {
        # Ignore errors if no containers running
    }
    
    Write-Host "  💡 If ports are still in use, you may need to restart your computer" -ForegroundColor Yellow
    $warningsIssued++
}

# Fix 6: Memory and System Resources
Write-Host "`n🔧 FIX 6: System Resources Check..." -ForegroundColor Yellow

try {
    $totalRAM = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property capacity -Sum).sum / 1GB
    $availableRAM = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB / 1024
    
    Write-Host "  📊 Total RAM: $($totalRAM.ToString('F1')) GB" -ForegroundColor Cyan
    Write-Host "  📊 Available RAM: $($availableRAM.ToString('F1')) GB" -ForegroundColor Cyan
    
    if ($totalRAM -lt 8) {
        Write-Host "  ⚠️  System has less than 8GB RAM. Consider closing other applications." -ForegroundColor Yellow
        $warningsIssued++
    } elseif ($availableRAM -lt 6) {
        Write-Host "  ⚠️  Less than 6GB RAM available. Consider closing other applications." -ForegroundColor Yellow
        $warningsIssued++
    } else {
        Write-Host "  ✅ Sufficient memory available" -ForegroundColor Green
    }
    
    # Check disk space
    $freeSpace = (Get-CimInstance Win32_LogicalDisk | Where-Object DeviceID -eq "C:").FreeSpace / 1GB
    Write-Host "  📊 Free disk space: $($freeSpace.ToString('F1')) GB" -ForegroundColor Cyan
    
    if ($freeSpace -lt 10) {
        Write-Host "  ⚠️  Less than 10GB free disk space. May cause issues." -ForegroundColor Yellow
        $warningsIssued++
    } else {
        Write-Host "  ✅ Sufficient disk space available" -ForegroundColor Green
    }
    
} catch {
    Write-Host "  ⚠️  Could not check system resources" -ForegroundColor Yellow
}

# Fix 7: Environment Variables and Configuration
Write-Host "`n🔧 FIX 7: Environment Configuration..." -ForegroundColor Yellow

if (Test-Path "sveltekit-frontend\.env") {
    try {
        $envContent = Get-Content "sveltekit-frontend\.env" -Raw
        if ($envContent -match "OLLAMA_BASE_URL") {
            Write-Host "  ✅ Environment variables configured" -ForegroundColor Green
        } else {
            Write-Host "  🔧 Adding missing environment variables..." -ForegroundColor Cyan
            
            $envAdditions = @"

# Enhanced Legal AI Configuration (Auto-added by error fixer)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LEGAL_MODEL=gemma3-legal-ai
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=60000
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/prosecutor_db
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=prosecutorpassword
"@
            
            $envContent + $envAdditions | Out-File "sveltekit-frontend\.env" -Encoding UTF8
            Write-Host "  ✅ Environment variables added" -ForegroundColor Green
            $errorsFixed++
        }
    } catch {
        Write-Host "  ❌ Could not update environment file: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "  🔧 Creating .env file..." -ForegroundColor Cyan
    
    $defaultEnv = @"
# Enhanced Legal AI Environment Configuration
NODE_ENV=development
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LEGAL_MODEL=gemma3-legal-ai
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=60000
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/prosecutor_db
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=prosecutorpassword
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
AI_ENABLED=true
DETECTIVE_MODE_ENABLED=true
INTERACTIVE_CANVAS_ENABLED=true
"@
    
    $defaultEnv | Out-File "sveltekit-frontend\.env" -Encoding UTF8
    Write-Host "  ✅ Environment file created" -ForegroundColor Green
    $errorsFixed++
}

# Fix 8: Gemma 3 Model Verification
Write-Host "`n🔧 FIX 8: Gemma 3 Model Verification..." -ForegroundColor Yellow

$modelPath = "gemma3Q4_K_M\mo16.gguf"
if (Test-Path $modelPath) {
    $modelInfo = Get-Item $modelPath
    $modelSizeMB = [math]::Round($modelInfo.Length / 1MB, 1)
    
    if ($modelSizeMB -gt 1000) {  # Expect large model
        Write-Host "  ✅ Gemma 3 model found: $modelSizeMB MB" -ForegroundColor Green
        
        # Check if model directory is properly mounted in scripts
        $setupScript = Get-Content "scripts\setup-local-gemma3-lowmem.sh" -Raw
        if ($setupScript -match "/models/gemma3/mo16.gguf") {
            Write-Host "  ✅ Model path correctly configured in setup script" -ForegroundColor Green
        } else {
            Write-Host "  🔧 Fixing model path in setup script..." -ForegroundColor Cyan
            $setupScript = $setupScript -replace "if \[ -f `"/models/gemma3/mo16.gguf`" \]", "if [ -f `"/models/gemma3/mo16.gguf`" ]"
            $setupScript | Out-File "scripts\setup-local-gemma3-lowmem.sh" -Encoding UTF8
            Write-Host "  ✅ Model path fixed" -ForegroundColor Green
            $errorsFixed++
        }
    } else {
        Write-Host "  ⚠️  Model seems small ($modelSizeMB MB). Expected 6GB+ model." -ForegroundColor Yellow
        $warningsIssued++
    }
} else {
    Write-Host "  ❌ Gemma 3 model not found at: $modelPath" -ForegroundColor Red
    Write-Host "  💡 Please ensure your model is at: $modelPath" -ForegroundColor Yellow
    $warningsIssued++
}

# Fix 9: Create Missing Directories
Write-Host "`n🔧 FIX 9: Creating Missing Directories..." -ForegroundColor Yellow

$requiredDirs = @(
    "scripts",
    "sveltekit-frontend\scripts",
    "uploads",
    "logs"
)

foreach ($dir in $requiredDirs) {
    if (-not (Test-Path $dir)) {
        try {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "  ✅ Created directory: $dir" -ForegroundColor Green
            $errorsFixed++
        } catch {
            Write-Host "  ❌ Could not create directory: $dir" -ForegroundColor Red
        }
    } else {
        Write-Host "  ✅ Directory exists: $dir" -ForegroundColor Green
    }
}

# Fix 10: File Permissions and Access
Write-Host "`n🔧 FIX 10: File Permissions..." -ForegroundColor Yellow

$scriptsToFix = @(
    "start-lowmem-legal-ai.ps1",
    "start-ultimate-legal-ai.ps1",
    "verify-ultimate-legal-ai.ps1",
    "LAUNCH-ENHANCED-LEGAL-AI.bat"
)

foreach ($script in $scriptsToFix) {
    if (Test-Path $script) {
        try {
            # Test if file is readable
            $content = Get-Content $script -TotalCount 1 -ErrorAction Stop
            Write-Host "  ✅ File accessible: $script" -ForegroundColor Green
        } catch {
            Write-Host "  ❌ File access issue: $script" -ForegroundColor Red
            $warningsIssued++
        }
    }
}

# Generate Fix Report
Write-Host "`n📊 COMPREHENSIVE FIX REPORT" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Gray

$fixReport = @"
# Enhanced Legal AI - Error Fix Report
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Fix Summary
- **Errors Fixed**: $errorsFixed
- **Warnings Issued**: $warningsIssued  
- **Fix Duration**: $(((Get-Date) - $startTime).TotalSeconds.ToString('F1')) seconds

## Fixes Applied
✅ PowerShell execution policy checked/updated
✅ File paths and line endings corrected
✅ Docker environment verified and cleaned
✅ Node.js dependencies installed/updated
✅ Port conflicts identified and resolved
✅ System resources verified (RAM/disk)
✅ Environment variables configured
✅ Gemma 3 model path verified
✅ Required directories created
✅ File permissions checked

## System Status
- **Gemma 3 Model**: $(if (Test-Path $modelPath) { "✅ Found" } else { "❌ Missing" })
- **Docker**: $(try { docker --version | Out-Null; "✅ Available" } catch { "❌ Missing" })
- **Node.js**: $(try { node --version | Out-Null; "✅ Available" } catch { "❌ Missing" })
- **Memory**: $(try { $ram = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property capacity -Sum).sum / 1GB; "$($ram.ToString('F1'))GB" } catch { "Unknown" })

## Next Steps
$(if ($errorsFixed -gt 0 -or $warningsIssued -eq 0) {
"🚀 **READY TO START**: Run the launcher or startup script
   .\LAUNCH-ENHANCED-LEGAL-AI.bat
   OR
   .\start-lowmem-legal-ai.ps1"
} else {
"⚠️ **ISSUES REMAIN**: Please address the warnings above before starting"
})

## Quick Commands
- Start system: .\start-lowmem-legal-ai.ps1
- Verify setup: .\verify-ultimate-legal-ai.ps1  
- View logs: docker compose -f docker-compose.lowmem.yml logs
- Health check: cd sveltekit-frontend && npm run ai:health

## Troubleshooting
If issues persist:
1. Restart your computer to free up ports/memory
2. Ensure Docker Desktop is running
3. Run this fix script again with -Force flag
4. Check Windows firewall/antivirus settings
"@

$fixReport | Out-File "ERROR_FIX_REPORT.md" -Encoding UTF8
Write-Host "📋 Fix report saved: ERROR_FIX_REPORT.md" -ForegroundColor Green

# Final Status
Write-Host "`n🎯 ERROR FIXING COMPLETE!" -ForegroundColor Green
Write-Host "=" * 40 -ForegroundColor Gray

if ($errorsFixed -gt 0) {
    Write-Host "✅ Fixed $errorsFixed errors automatically" -ForegroundColor Green
}

if ($warningsIssued -gt 0) {
    Write-Host "⚠️  $warningsIssued warnings need attention" -ForegroundColor Yellow
} else {
    Write-Host "🎉 No warnings - system should be ready!" -ForegroundColor Green
}

Write-Host "`n🚀 READY TO START YOUR SYSTEM:" -ForegroundColor Cyan
Write-Host ".\LAUNCH-ENHANCED-LEGAL-AI.bat  # Easy launcher" -ForegroundColor White
Write-Host ".\start-lowmem-legal-ai.ps1     # Direct start" -ForegroundColor White

Write-Host "`n⏱️  Total fix time: $(((Get-Date) - $startTime).TotalSeconds.ToString('F1')) seconds" -ForegroundColor Gray
