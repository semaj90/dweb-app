Write-Host "🔍 Checking SvelteKit Frontend Errors..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

Write-Host "📁 Checking file structure..." -ForegroundColor Yellow
Write-Host "- package.json exists: $(if (Test-Path 'package.json') { '✅' } else { '❌' })"
Write-Host "- src/lib exists: $(if (Test-Path 'src/lib') { '✅' } else { '❌' })"
Write-Host "- node_modules exists: $(if (Test-Path 'node_modules') { '✅' } else { '❌' })"

Write-Host ""
Write-Host "📦 Attempting TypeScript check..." -ForegroundColor Yellow
try {
    $checkResult = & npm run check 2>&1 | Select-Object -First 10
    Write-Host $checkResult
} catch {
    Write-Host "TypeScript check failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "✅ Quick check completed!" -ForegroundColor Green
