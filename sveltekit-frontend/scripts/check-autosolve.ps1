# PowerShell script to run check followed by autosolve without debugger
# Clear all Node.js debugger environment variables that cause debugger attachment
Remove-Item Env:NODE_OPTIONS -ErrorAction SilentlyContinue
Remove-Item Env:NODE_INSPECT -ErrorAction SilentlyContinue  
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
Remove-Item Env:VSCODE_NODE_CACHED_DATA_DIR -ErrorAction SilentlyContinue
Remove-Item Env:ELECTRON_ENABLE_LOGGING -ErrorAction SilentlyContinue

# Set clean Node options
$env:NODE_OPTIONS = "--max-old-space-size=8192"
$env:NODE_INSPECT = ""

Write-Host "Running TypeScript check..." -ForegroundColor Cyan
& npm run check:ultra-fast

Write-Host "Check completed. Running autosolve to fix remaining errors..." -ForegroundColor Yellow
& npm run auto:solve

if ($LASTEXITCODE -eq 0) {
    Write-Host "Autosolve completed successfully" -ForegroundColor Green
} else {
    Write-Host "Autosolve completed with some remaining issues" -ForegroundColor Yellow
}