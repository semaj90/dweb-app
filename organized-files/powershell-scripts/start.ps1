# Legal AI System - Quick Start Script
# One-command startup for the entire system

param(
    [switch]$SkipValidation = $false,
    [switch]$OpenBrowser = $false
)

$ErrorActionPreference = "Stop"

Write-Host @"
╔═══════════════════════════════════════════════════════╗
║          Legal AI System - Quick Start                 ║
║                  Version 1.0.0                         ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

function Start-SystemWithRetry {
    $maxAttempts = 3
    $attempt = 0
    
    while ($attempt -lt $maxAttempts) {
        $attempt++
        Write-Host "`n🚀 Starting Legal AI System (Attempt $attempt/$maxAttempts)..." -ForegroundColor Yellow
        
        try {
            # Step 1: Validate system
            if (-not $SkipValidation) {
                Write-Host "`n[Step 1/4] Running system validation..." -ForegroundColor Cyan
                $validation = & ".\validate.ps1" -Minimal
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "⚠️  Validation failed. Running quick fix..." -ForegroundColor Yellow
                    & ".\health-check.ps1" -Fix
                }
            } else {
                Write-Host "`n[Step 1/4] Skipping validation..." -ForegroundColor Gray
            }
            
            # Step 2: Ensure Docker is running
            Write-Host "`n[Step 2/4] Checking Docker..." -ForegroundColor Cyan
            $dockerRunning = docker info 2>$null
            if (-not $dockerRunning) {
                Write-Host "Starting Docker Desktop..." -ForegroundColor Yellow
                Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -ErrorAction SilentlyContinue
                Start-Sleep -Seconds 20
            }
            
            # Step 3: Start infrastructure
            Write-Host "`n[Step 3/4] Starting infrastructure services..." -ForegroundColor Cyan
            docker-compose up -d
            
            # Wait for services
            Write-Host "Waiting for services to initialize..." -ForegroundColor Gray
            $services = @("postgres", "redis", "qdrant", "ollama")
            $ready = $false
            $waitTime = 0
            
            while (-not $ready -and $waitTime -lt 60) {
                $ready = $true
                foreach ($service in $services) {
                    $health = docker inspect "legal_ai_$service" --format='{{.State.Health.Status}}' 2>$null
                    if ($health -ne "healthy" -and $health -ne $null) {
                        $ready = $false
                        break
                    }
                }
                
                if (-not $ready) {
                    Write-Host "." -NoNewline
                    Start-Sleep -Seconds 2
                    $waitTime += 2
                }
            }
            
            Write-Host " Ready!" -ForegroundColor Green
            
            # Step 4: Start development server
            Write-Host "`n[Step 4/4] Starting development server..." -ForegroundColor Cyan
            
            # Open browser if requested
            if ($OpenBrowser) {
                Write-Host "Opening browser in 5 seconds..." -ForegroundColor Gray
                Start-Job -ScriptBlock {
                    Start-Sleep -Seconds 5
                    Start-Process "http://localhost:5173"
                } | Out-Null
            }
            
            # Display startup message
            Write-Host "`n" -NoNewline
            Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Green
            Write-Host "✅ Legal AI System is starting up!" -ForegroundColor Green
            Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Green
            Write-Host ""
            Write-Host "📍 Access the application at:" -ForegroundColor Yellow
            Write-Host "   http://localhost:5173" -ForegroundColor White
            Write-Host ""
            Write-Host "📊 View system health:" -ForegroundColor Yellow
            Write-Host "   http://localhost:5173/api/health" -ForegroundColor White
            Write-Host ""
            Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Yellow
            Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Green
            Write-Host ""
            
            # Change to frontend directory and start server
            Push-Location sveltekit-frontend
            npm run dev
            
            # If we get here, the server was stopped
            break
            
        } catch {
            Write-Host "`n❌ Startup failed: $_" -ForegroundColor Red
            
            if ($attempt -lt $maxAttempts) {
                Write-Host "Retrying in 5 seconds..." -ForegroundColor Yellow
                Start-Sleep -Seconds 5
            } else {
                Write-Host "`n❌ Failed to start after $maxAttempts attempts" -ForegroundColor Red
                Write-Host "Run .\health-check.ps1 for diagnostics" -ForegroundColor Yellow
                exit 1
            }
        } finally {
            Pop-Location -ErrorAction SilentlyContinue
        }
    }
}

# Cleanup function
function Stop-System {
    Write-Host "`n🛑 Shutting down Legal AI System..." -ForegroundColor Yellow
    
    # Stop development server (handled by Ctrl+C)
    
    # Stop Docker services
    Write-Host "Stopping Docker services..." -ForegroundColor Gray
    docker-compose stop
    
    Write-Host "✅ System stopped successfully" -ForegroundColor Green
}

# Set up cleanup on exit
try {
    # Register cleanup
    Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
        Stop-System
    } | Out-Null
    
    # Start the system
    Start-SystemWithRetry
    
} finally {
    # Ensure cleanup runs
    Stop-System
}
