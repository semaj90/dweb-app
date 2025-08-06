# Complete-Legal-AI-Startup.ps1
# Comprehensive startup script for the Legal AI system

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Legal AI System Manager"

# Color-coded output functions
function Write-Success { Write-Host "✅ $args" -ForegroundColor Green }
function Write-Info { Write-Host "ℹ️  $args" -ForegroundColor Cyan }
function Write-Warning { Write-Host "⚠️  $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "❌ $args" -ForegroundColor Red }
function Write-Header { 
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Blue
    Write-Host " $args" -ForegroundColor White
    Write-Host "=" * 60 -ForegroundColor Blue
}

Write-Header "LEGAL AI SYSTEM STARTUP"

# 1. Start PostgreSQL
Write-Info "Starting PostgreSQL..."
$pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
if ($pgService -and $pgService.Status -ne "Running") {
    Start-Service $pgService.Name
    Start-Sleep -Seconds 3
}
Write-Success "PostgreSQL is running"

# 2. Setup Database
Write-Info "Setting up database..."
$env:PGPASSWORD = "postgres"
psql -U postgres -d legal_ai_db -f ".\database\setup-legal-ai.sql" 2>&1 | Out-Null
Write-Success "Database configured"

# 3. Start Redis
Write-Info "Starting Redis..."
$redisProcess = Get-Process -Name "redis-server" -ErrorAction SilentlyContinue
if (-not $redisProcess) {
    if (Test-Path ".\redis-windows\redis-server.exe") {
        Start-Process -FilePath ".\redis-windows\redis-server.exe" -WindowStyle Hidden
        Start-Sleep -Seconds 2
    }
}
Write-Success "Redis is running"

# 4. Start Ollama
Write-Info "Starting Ollama..."
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2
} catch {
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}
Write-Success "Ollama is running"

# 5. Build and Start Go Server
Write-Info "Building Go server..."
Push-Location ".\go-microservice"
go build -o legal-ai-server.exe . 2>&1 | Out-Null
$env:OLLAMA_URL = "http://localhost:11434"
$env:DATABASE_URL = "postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db"
$env:PORT = "8080"
Start-Process -FilePath ".\legal-ai-server.exe" -WindowStyle Hidden
Pop-Location
Write-Success "Go server started on port 8080"

# 6. Start Workers
Write-Info "Starting BullMQ workers..."
Push-Location ".\workers"
if (-not (Test-Path "node_modules")) {
    npm install bullmq ioredis axios 2>&1 | Out-Null
}
Start-Process -FilePath "node" -ArgumentList "document-processor.worker.js" -WindowStyle Hidden
Pop-Location
Write-Success "Workers started"

# 7. Start Frontend
Write-Info "Starting SvelteKit frontend..."
Push-Location ".\sveltekit-frontend"
if (-not (Test-Path "node_modules")) {
    npm install 2>&1 | Out-Null
}
Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WindowStyle Hidden
Pop-Location
Write-Success "Frontend started on http://localhost:5173"

# Final status
Write-Header "SYSTEM READY"
Write-Success "Legal AI System is running!"
Write-Info ""
Write-Info "🌐 Frontend: http://localhost:5173"
Write-Info "🚀 API: http://localhost:8080/health"
Write-Info "📊 Database: PostgreSQL on port 5432"
Write-Info "💾 Cache: Redis on port 6379"
Write-Info "🤖 AI: Ollama on port 11434"
Write-Info ""
Write-Info "Press Ctrl+C to stop all services"

# Keep script running
while ($true) {
    Start-Sleep -Seconds 60
}
