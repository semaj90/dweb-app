Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Running Svelte Check Error Fixes" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Applying comprehensive fixes..." -ForegroundColor Yellow
node fix-svelte-check-errors.mjs

Write-Host ""
Write-Host "Step 2: Running svelte-check to verify..." -ForegroundColor Yellow
Write-Host ""
npm run check

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "Fix process complete!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Check the output above for any remaining issues." -ForegroundColor White
Write-Host "Most warnings about unused CSS selectors (dark mode) can be ignored." -ForegroundColor Gray
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
