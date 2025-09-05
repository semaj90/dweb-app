# Fix svelte-check memory and Melt UI compatibility issues
param(
    [int]$MaxMemory = 8192,
    [switch]$UpdatePackages,
    [switch]$OnlyCount
)

Write-Host "Fixing svelte-check issues..." -ForegroundColor Cyan

# Clear all Node.js debugger environment variables
$env:NODE_OPTIONS = "--max-old-space-size=$MaxMemory"
$env:NODE_INSPECT = ""
$env:NODE_DEBUG = ""
[Environment]::SetEnvironmentVariable('NODE_INSPECT', '', 'Process')
[Environment]::SetEnvironmentVariable('NODE_DEBUG', '', 'Process')

Write-Host "Set Node.js memory limit to $MaxMemory MB" -ForegroundColor Green

if ($UpdatePackages) {
    Write-Host "Updating svelte-check for Svelte 5 + Melt compatibility..." -ForegroundColor Yellow
    npm update svelte-check "@sveltejs/kit" svelte "@melt-ui/svelte"
    Write-Host "Packages updated for Svelte 5 compatibility" -ForegroundColor Green
}

if ($OnlyCount) {
    Write-Host "Running quick error count..." -ForegroundColor Yellow
    $output = & './node_modules/.bin/svelte-check' --tsconfig './tsconfig.json' --threshold error --fail-on-warnings false --no-watch 2>&1
    
    # Extract error count from output
    $errorLines = $output | Where-Object { $_ -match "found \d+ error" }
    if ($errorLines) {
        Write-Host "Error count: $errorLines" -ForegroundColor Red
    } else {
        Write-Host "No clear error count found - check output above" -ForegroundColor Green
    }
} else {
    Write-Host "Running full svelte-check with increased memory..." -ForegroundColor Yellow
    & './node_modules/.bin/svelte-check' --tsconfig './tsconfig.json' --threshold error --fail-on-warnings false --no-watch
}

Write-Host "svelte-check process completed" -ForegroundColor Cyan