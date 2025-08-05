# Setup-Instructions-Legal-AI.ps1
# Complete setup instructions and commands for Legal AI System

<#
.SYNOPSIS
    Complete setup instructions for Legal AI System with local gemma3-legal model
.DESCRIPTION
    This script contains all setup instructions and commands converted to PowerShell
.EXAMPLE
    .\Setup-Instructions-Legal-AI.ps1
.NOTES
    Save this file and run sections as needed
#>

# Color functions for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-Host "✅ $args" -ForegroundColor Green }
function Write-Error { Write-Host "❌ $args" -ForegroundColor Red }
function Write-Warning { Write-Host "⚠️  $args" -ForegroundColor Yellow }
function Write-Info { Write-Host "ℹ️  $args" -ForegroundColor Cyan }
function Write-Header { 
    Write-Host "`n════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host " $args" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════`n" -ForegroundColor Cyan
}

Write-Header "LEGAL AI SYSTEM SETUP INSTRUCTIONS"
Write-Info "PowerShell Version - Using Local gemma3-legal Model"

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: INITIAL SETUP
# ═══════════════════════════════════════════════════════════════════════

Write-Header "STEP 1: SAVE THE SETUP SCRIPT"

Write-Info @"
Save the main setup script as:
C:\Users\james\Desktop\deeds-web\deeds-web-app\Setup-Legal-AI-System.ps1

This script will create all necessary files and directories.
"@

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: RUN THE SETUP SCRIPT
# ═══════════════════════════════════════════════════════════════════════

Write-Header "STEP 2: RUN THE SETUP SCRIPT"

Write-Info "Open PowerShell as Administrator and run:"

$setupCommands = @'
# Navigate to project directory
Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app"

# Run setup script with default options
powershell -ExecutionPolicy Bypass -File Setup-Legal-AI-System.ps1

# Or with options:
# Force overwrite existing files
powershell -ExecutionPolicy Bypass -File Setup-Legal-AI-System.ps1 -Force

# Skip dependency checks
powershell -ExecutionPolicy Bypass -File Setup-Legal-AI-System.ps1 -SkipDependencyCheck

# Both options
powershell -ExecutionPolicy Bypass -File Setup-Legal-AI-System.ps1 -Force -SkipDependencyCheck
'@

Write-Host $setupCommands -ForegroundColor Yellow

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: WHAT THE SCRIPT CREATES
# ═══════════════════════════════════════════════════════════════════════

Write-Header "STEP 3: WHAT THE SCRIPT CREATES"

Write-Success "Creates Directory Structure:"
Write-Host @"
  • /workers           - BullMQ worker files
  • /database          - SQL schema files
  • /uploads           - Document upload storage
  • /src/routes/api/*  - SvelteKit API routes
  • /src/lib/*         - Client libraries
"@

Write-Success "`nChecks Dependencies:"
Write-Host @"
  • Node.js 18+
  • Go 1.21+
  • PostgreSQL
  • Redis for Windows
  • Ollama with gemma3-legal model
  • NVIDIA CUDA
"@

Write-Success "`nCreates Configuration Files:"
Write-Host @"
  • .env                       - Environment variables
  • SET-LEGAL-AI-ENV.bat       - Windows environment setup
  • model-config.go            - Go server model configuration
  • workers/package.json       - Worker dependencies
  • database/schema.sql        - PostgreSQL schema
  • ecosystem.config.js        - PM2 configuration
"@

Write-Success "`nCreates Scripts:"
Write-Host @"
  • VERIFY-LOCAL-MODELS.bat    - Verify Ollama models
  • START-LEGAL-AI-COMPLETE.bat - Start entire system
  • TEST-QUICK.bat             - Quick system test
"@

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: POST-SETUP COMMANDS
# ═══════════════════════════════════════════════════════════════════════

Write-Header "STEP 4: AFTER RUNNING THE SETUP SCRIPT"

Write-Info "Execute these PowerShell commands in order:"

# Function to run commands with error checking
function Invoke-SetupCommand {
    param(
        [string]$Description,
        [scriptblock]$Command,
        [switch]$ContinueOnError
    )
    
    Write-Host "`n$Description" -ForegroundColor Cyan
    try {
        & $Command
        Write-Success "Completed successfully"
    } catch {
        Write-Error "Failed: $_"
        if (-not $ContinueOnError) {
            Write-Warning "Fix this error before continuing"
            return $false
        }
    }
    return $true
}

# 1. Install worker dependencies
Write-Info "`n1. Install Worker Dependencies:"
$workerCommands = @'
Set-Location workers
npm install
Set-Location ..
'@
Write-Host $workerCommands -ForegroundColor Yellow

# 2. Create PostgreSQL database
Write-Info "`n2. Create PostgreSQL Database:"
$postgresCommands = @'
# Using psql directly
psql -U postgres -c "CREATE DATABASE legal_ai;"
psql -U postgres -d legal_ai -f "database\schema.sql"

# Or using full path if psql not in PATH
& "C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres -c "CREATE DATABASE legal_ai;"
& "C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres -d legal_ai -f "database\schema.sql"
'@
Write-Host $postgresCommands -ForegroundColor Yellow

# 3. Update Go dependencies
Write-Info "`n3. Update Go Dependencies:"
$goCommands = @'
Set-Location go-microservice
go mod tidy
go mod download
Set-Location ..
'@
Write-Host $goCommands -ForegroundColor Yellow

# 4. Verify Ollama model
Write-Info "`n4. Verify Your Model:"
$ollamaCommands = @'
# List all models
ollama list

# Test gemma3-legal model
ollama run gemma3-legal "Test legal document summary"
'@
Write-Host $ollamaCommands -ForegroundColor Yellow

# 5. Run quick test
Write-Info "`n5. Run Quick Test:"
Write-Host ".\TEST-QUICK.bat" -ForegroundColor Yellow

# 6. Start the system
Write-Info "`n6. Start the System:"
Write-Host ".\START-LEGAL-AI-COMPLETE.bat" -ForegroundColor Yellow

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════

Write-Header "STEP 5: COMMON ERRORS AND FIXES"

Write-Info "The script uses color coding:"
Write-Success "Green = Success"
Write-Error "Red = Error (must fix)"
Write-Warning "Yellow = Warning (should fix)"
Write-Info "Cyan = Information"

Write-Host "`nCommon errors and PowerShell fixes:" -ForegroundColor White

# Error 1: Model not found
Write-Warning "`nError: 'gemma3-legal model not found'"
$modelFixCommands = @'
# Check installed models
ollama list

# If not listed, ensure your model is properly created
ollama show gemma3-legal

# Test the model
ollama run gemma3-legal "What is a legal contract?"
'@
Write-Host $modelFixCommands -ForegroundColor Gray

# Error 2: PostgreSQL not found
Write-Warning "`nError: 'PostgreSQL psql not found'"
$psqlFixCommands = @'
# Add PostgreSQL to PATH (adjust version as needed)
$env:Path += ";C:\Program Files\PostgreSQL\15\bin"

# Or use full path
$psqlPath = "C:\Program Files\PostgreSQL\15\bin\psql.exe"
& $psqlPath -U postgres -c "CREATE DATABASE legal_ai;"
'@
Write-Host $psqlFixCommands -ForegroundColor Gray

# Error 3: File exists
Write-Warning "`nError: 'File exists (use -Force to overwrite)'"
$forceFixCommand = @'
# Run with -Force flag
powershell -ExecutionPolicy Bypass -File Setup-Legal-AI-System.ps1 -Force
'@
Write-Host $forceFixCommand -ForegroundColor Gray

# Error 4: Redis not found
Write-Warning "`nError: 'Redis for Windows not found'"
$redisFixCommands = @'
# Check Redis location
Test-Path "redis-windows\redis-server.exe"

# If missing, verify you have Redis for Windows in the redis-windows folder
Get-ChildItem "redis-windows" -Filter "*.exe"
'@
Write-Host $redisFixCommands -ForegroundColor Gray

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

Write-Header "STEP 6: VERIFY EVERYTHING WORKS"

Write-Info "Run the comprehensive verification script:"
$verifyCommand = @'
powershell -ExecutionPolicy Bypass -File Verify-Legal-AI-Integration.ps1 -Detailed
'@
Write-Host $verifyCommand -ForegroundColor Yellow

Write-Info "`nThis will test:"
Write-Success "  • PostgreSQL connection"
Write-Success "  • Redis connection"
Write-Success "  • Go GPU server"
Write-Success "  • Ollama service"
Write-Success "  • gemma3-legal model"
Write-Success "  • SvelteKit frontend"
Write-Success "  • BullMQ workers"

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: QUICK START
# ═══════════════════════════════════════════════════════════════════════

Write-Header "STEP 7: QUICK START AFTER SETUP"

Write-Info "Once everything is set up, start the entire system with one command:"
Write-Host "`n.\START-LEGAL-AI-COMPLETE.bat" -ForegroundColor Green -BackgroundColor Black

Write-Info "`nThis will:"
Write-Host @"
  1. Set environment variables
  2. Verify gemma3-legal model
  3. Start Redis
  4. Start Go GPU server
  5. Start Ollama
  6. Start BullMQ workers
  7. Start SvelteKit
"@

Write-Success "`nYour Legal AI system will be available at:"
Write-Host @"
  • Frontend:     http://localhost:5173
  • Go API:       http://localhost:8081
  • Health Check: http://localhost:8081/health
  • Ollama:       http://localhost:11434
"@

# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

Write-Header "HELPFUL POWERSHELL COMMANDS"

# Function to check service status
function Test-ServiceStatus {
    Write-Info "Checking all services..."
    
    # PostgreSQL
    $pgStatus = Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet
    Write-Host "PostgreSQL: $(if($pgStatus){'✅ Running'}else{'❌ Not running'})"
    
    # Redis
    $redisStatus = Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet
    Write-Host "Redis: $(if($redisStatus){'✅ Running'}else{'❌ Not running'})"
    
    # Go Server
    $goStatus = Test-NetConnection -ComputerName localhost -Port 8081 -InformationLevel Quiet
    Write-Host "Go Server: $(if($goStatus){'✅ Running'}else{'❌ Not running'})"
    
    # Ollama
    try {
        $ollamaResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 2
        Write-Host "Ollama: ✅ Running"
    } catch {
        Write-Host "Ollama: ❌ Not running"
    }
    
    # SvelteKit
    $svelteStatus = Test-NetConnection -ComputerName localhost -Port 5173 -InformationLevel Quiet
    Write-Host "SvelteKit: $(if($svelteStatus){'✅ Running'}else{'❌ Not running'})"
}

# Function to stop all services
function Stop-AllServices {
    Write-Warning "Stopping all Legal AI services..."
    
    # Stop Node processes
    Get-Process -Name "node" -ErrorAction SilentlyContinue | Stop-Process -Force
    
    # Stop Go server
    Get-Process -Name "legal-ai-gpu-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    
    # Stop Redis
    Get-Process -Name "redis-server" -ErrorAction SilentlyContinue | Stop-Process -Force
    
    # Stop Ollama
    Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
    
    Write-Success "All services stopped"
}

# Function to clean logs
function Clear-Logs {
    Write-Info "Cleaning log files..."
    Remove-Item "logs\*.log" -Force -ErrorAction SilentlyContinue
    Remove-Item "*.log" -Force -ErrorAction SilentlyContinue
    Write-Success "Logs cleaned"
}

Write-Info @"

Useful PowerShell commands:

# Check all service status
Test-ServiceStatus

# Stop all services
Stop-AllServices

# Clean logs
Clear-Logs

# Watch GPU usage
while(`$true) { nvidia-smi; Start-Sleep -Seconds 2; Clear-Host }

# Monitor Redis
redis-windows\redis-cli.exe monitor

# Check Ollama models
ollama list

# Test Go server health
Invoke-RestMethod -Uri "http://localhost:8081/health" | ConvertTo-Json
"@

# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════

Write-Header "TROUBLESHOOTING GUIDE"

Write-Info @"
If you encounter issues:

1. Check the setup log:
   Get-Content setup-log.txt

2. Verify environment variables:
   `$env:LEGAL_MODEL
   `$env:OLLAMA_URL
   `$env:GO_SERVER_URL

3. Test individual components:
   # Redis
   redis-windows\redis-cli.exe ping
   
   # Ollama
   curl http://localhost:11434/api/tags
   
   # PostgreSQL
   psql -U postgres -c "SELECT version();"

4. Check Windows Firewall:
   Ensure ports 5173, 6379, 8081, 11434, 5432 are allowed

5. Review error logs:
   Get-Content logs\*.log -Tail 50
"@

Write-Success "`n✅ Setup instructions complete! Your Legal AI system is configured to use your local gemma3-legal model."
Write-Info "No external models will be downloaded - everything runs locally."

# Save this script's last run time
$runInfo = @{
    LastRun = Get-Date
    User = $env:USERNAME
    ProjectPath = Get-Location
}
$runInfo | ConvertTo-Json | Set-Content "setup-instructions-lastrun.json"