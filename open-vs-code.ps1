# Open Legal AI Platform in VS Code
# Quick launcher script for the development environment

Write-Host "🚀 Opening Legal AI Platform in VS Code..." -ForegroundColor Cyan

# Change to project directory
Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app"

# Open VS Code with the workspace
& code deeds-web-app.code-workspace

Write-Host "✅ VS Code opened with workspace!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Quick Start Guide:" -ForegroundColor Yellow
Write-Host "1. Press Ctrl+Shift+P to open Command Palette" -ForegroundColor White
Write-Host "2. Type 'Tasks: Run Task'" -ForegroundColor White  
Write-Host "3. Select '🚀 Start Legal AI Platform (Quick)'" -ForegroundColor White
Write-Host ""
Write-Host "🌐 Access Points (after starting):" -ForegroundColor Yellow
Write-Host "• Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "• Health Dashboard: http://localhost:5173/system/health" -ForegroundColor White
Write-Host ""
Write-Host "📖 Full guide: See VS_CODE_USAGE.md in the project root" -ForegroundColor Magenta