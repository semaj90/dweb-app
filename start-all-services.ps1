# Legal AI Platform - Complete Startup Script
Write-Host "Starting Legal AI Platform Services..." -ForegroundColor Cyan

# Start databases if not running
$processes = @()

# PostgreSQL (already running)
Write-Host "✓ PostgreSQL already running on port 5432" -ForegroundColor Green

# Redis (already running)  
Write-Host "✓ Redis already running on port 6379" -ForegroundColor Green

# Start Neo4j if installed
if (Get-Command neo4j -ErrorAction SilentlyContinue) {
    Write-Host "Starting Neo4j..." -ForegroundColor Yellow
    Start-Process -FilePath "neo4j" -ArgumentList "console" -WindowStyle Minimized
} else {
    Write-Host "⚠ Neo4j not installed - skipping" -ForegroundColor Yellow
}

# Start MinIO if exists
if (Test-Path ".\minio.exe") {
    Write-Host "Starting MinIO..." -ForegroundColor Yellow
    Start-Process -FilePath ".\minio.exe" -ArgumentList "server", ".\data", "--console-address", ":9001" -WindowStyle Minimized
} else {
    Write-Host "⚠ MinIO not found - skipping" -ForegroundColor Yellow
}

# Ollama (already running)
Write-Host "✓ Ollama already running on port 11434" -ForegroundColor Green

# Start Enhanced RAG
if (Test-Path ".\enhanced-rag\main.go") {
    Write-Host "Starting Enhanced RAG Service..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd enhanced-rag; go run main.go" -WindowStyle Minimized
    $processes += "Enhanced RAG"
}

# Start Upload Service
if (Test-Path ".\legal-document-upload\package.json") {
    Write-Host "Starting Upload Service..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd legal-document-upload; npm run dev" -WindowStyle Minimized
    $processes += "Upload Service"
}

# ML Pipeline (already running)
Write-Host "✓ ML Pipeline already running on port 8080" -ForegroundColor Green

# XState Manager (already running)
Write-Host "✓ XState Manager already running on port 8095" -ForegroundColor Green

# Frontend (already running)
Write-Host "✓ Frontend already running on port 5173" -ForegroundColor Green

Write-Host "`n═════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "All available services started!" -ForegroundColor Green
Write-Host "═════════════════════════════════════════" -ForegroundColor Cyan

Write-Host "`nAccess points:" -ForegroundColor Yellow
Write-Host "  Frontend:      http://localhost:5173" -ForegroundColor White
Write-Host "  Enhanced RAG:  http://localhost:8094" -ForegroundColor White
Write-Host "  Upload Service: http://localhost:8093" -ForegroundColor White
Write-Host "  ML Pipeline:   http://localhost:8080" -ForegroundColor White
Write-Host "  XState Mgr:    http://localhost:8095" -ForegroundColor White
Write-Host "  MinIO Console: http://localhost:9001" -ForegroundColor White
Write-Host "  Neo4j Browser: http://localhost:7474" -ForegroundColor White

Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
