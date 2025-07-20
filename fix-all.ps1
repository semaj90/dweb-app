# Comprehensive error fixing script for Windows PowerShell

Write-Host "🔧 Starting comprehensive error fix process..." -ForegroundColor Cyan

# Create logs directory if it doesn't exist
if (!(Test-Path -Path "logs")) {
    New-Item -ItemType Directory -Path "logs"
}

# Step 1: Install missing dependencies
Write-Host "`n📦 Installing missing dependencies..." -ForegroundColor Yellow
Push-Location sveltekit-frontend
npm install fuse.js
Pop-Location

# Step 2: Install glob for the fix scripts
Write-Host "`n📦 Installing script dependencies..." -ForegroundColor Yellow
npm install glob

# Step 3: Run the specific error fixes
Write-Host "`n🔨 Running specific error fixes..." -ForegroundColor Yellow
node fix-specific-errors.mjs

# Step 4: Fix TypeScript imports
Write-Host "`n📝 Fixing TypeScript imports..." -ForegroundColor Yellow
node fix-all-typescript-imports.mjs

# Step 5: Run comprehensive fix (if glob is installed)
Write-Host "`n🛠️ Running comprehensive fixes..." -ForegroundColor Yellow
try {
    node fix-all-errors.mjs
} catch {
    Write-Host "⚠️ Comprehensive fix failed, continuing..." -ForegroundColor Magenta
}

# Step 6: Format code
Write-Host "`n✨ Formatting code..." -ForegroundColor Yellow
Push-Location sveltekit-frontend
try {
    npm run format
} catch {
    Write-Host "⚠️ Formatting failed, continuing..." -ForegroundColor Magenta
}
Pop-Location

# Step 7: Final check and log
Write-Host "`n📋 Running final check..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = "logs\final-check-$timestamp.log"

Push-Location sveltekit-frontend
$checkOutput = npm run check 2>&1
$checkOutput | Out-File -FilePath "..\$logFile" -Encoding UTF8
$checkOutput | Write-Host
Pop-Location

# Count remaining errors
$errorCount = ($checkOutput | Select-String -Pattern "Error:" -AllMatches).Matches.Count
$warningCount = ($checkOutput | Select-String -Pattern "Warn:" -AllMatches).Matches.Count

Write-Host "`n📊 Summary:" -ForegroundColor Cyan
Write-Host "  Errors: $errorCount" -ForegroundColor $(if ($errorCount -eq 0) { "Green" } else { "Red" })
Write-Host "  Warnings: $warningCount" -ForegroundColor $(if ($warningCount -eq 0) { "Green" } else { "Yellow" })
Write-Host "  Log saved to: $logFile" -ForegroundColor Gray

if ($errorCount -eq 0) {
    Write-Host "`n✅ All errors fixed! 🎉" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ Some errors remain. Check the log file for details." -ForegroundColor Yellow
    Write-Host "Common solutions:" -ForegroundColor Yellow
    Write-Host "  1. Run: npm install" -ForegroundColor White
    Write-Host "  2. Check for circular dependencies" -ForegroundColor White
    Write-Host "  3. Verify all imports are correct" -ForegroundColor White
}

Write-Host "`n✅ Fix process complete!" -ForegroundColor Green
