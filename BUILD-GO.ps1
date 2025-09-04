# Quick Go Build Script
Write-Host "`n=== BUILDING ENHANCED RAG V2 ===" -ForegroundColor Cyan

cd go-microservice
$env:CGO_ENABLED = "0"

# Build the simple server
Write-Host "Building simple RAG server..." -ForegroundColor Yellow
& cmd /c "go build -o rag-server.exe simple-rag-server.go 2>&1"

if (Test-Path "rag-server.exe") {
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host "Starting server on port 8084..." -ForegroundColor Yellow
    
    # Kill any existing instance
    Get-Process rag-server -ErrorAction SilentlyContinue | Stop-Process -Force
    
    # Start the server
    Start-Process -FilePath ".\rag-server.exe" -WindowStyle Hidden
    Start-Sleep -Seconds 2
    
    Write-Host "`nServer started!" -ForegroundColor Green
    Write-Host "Check: http://localhost:8084/api/health" -ForegroundColor Cyan
} else {
    Write-Host "Build failed. Check if Go is in PATH." -ForegroundColor Red
    Write-Host "Try running in Command Prompt instead." -ForegroundColor Yellow
}

cd ..
Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")