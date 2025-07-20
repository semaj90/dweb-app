# WSL2 Docker Setup Script
# Run this in PowerShell as Administrator

Write-Host "🚀 Setting up WSL2 + Docker for safe development" -ForegroundColor Green

# Check if WSL2 is installed
$wslVersion = wsl --status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing WSL2..." -ForegroundColor Yellow
    wsl --install
    Write-Host "⚠️ Please restart your computer after WSL2 installation completes" -ForegroundColor Red
    exit
}

# Install Ubuntu if not present
$ubuntuInstalled = wsl --list --quiet | Select-String "Ubuntu"
if (-not $ubuntuInstalled) {
    Write-Host "Installing Ubuntu..." -ForegroundColor Yellow
    wsl --install -d Ubuntu
}

Write-Host "✅ WSL2 setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Open Ubuntu from Start Menu" -ForegroundColor White
Write-Host "2. Create your username/password" -ForegroundColor White
Write-Host "3. Run: sudo apt update && sudo apt install docker.io docker-compose -y" -ForegroundColor White
Write-Host "4. Run: sudo usermod -aG docker `$USER" -ForegroundColor White
Write-Host "5. Restart WSL: wsl --shutdown (then reopen Ubuntu)" -ForegroundColor White
Write-Host ""
Write-Host "For your project:" -ForegroundColor Cyan
Write-Host "- Development: Use SQLite (npm run dev)" -ForegroundColor White
Write-Host "- Testing: Use Docker in WSL2 (docker-compose up -d)" -ForegroundColor White
