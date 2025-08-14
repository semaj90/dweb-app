# Targeted fix for MCP servers that need the Windows wrapper

$configPath = "$env:USERPROFILE\.claude.json"

Write-Host "MCP Server Windows Fixer - Targeted Approach" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Read the config file
Write-Host "Reading configuration file..." -ForegroundColor Yellow
$content = Get-Content $configPath -Raw

# Servers that need fixing based on analysis
$serversToFix = @(
    @{name = "postgres"; pattern = '("postgres"\s*:\s*\{[^}]*"command"\s*:\s*")npx'; replacement = '${1}cmd /c npx'},
    @{name = "context7"; pattern = '("context7"\s*:\s*\{[^}]*"command"\s*:\s*")npx'; replacement = '${1}cmd /c npx'}
)

$changed = $false
$originalContent = $content

foreach ($server in $serversToFix) {
    Write-Host "Fixing $($server.name) server..." -ForegroundColor Yellow
    
    if ($content -match $server.pattern) {
        $content = $content -replace $server.pattern, $server.replacement
        $changed = $true
        Write-Host "  ✓ Fixed $($server.name) command" -ForegroundColor Green
    } else {
        Write-Host "  ! Pattern not found for $($server.name)" -ForegroundColor Red
    }
}

if ($changed) {
    # Create backup
    $backupPath = "$configPath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Set-Content -Path $backupPath -Value $originalContent -Encoding UTF8
    Write-Host ""
    Write-Host "✓ Created backup: $backupPath" -ForegroundColor Green
    
    # Save the fixed content
    Set-Content -Path $configPath -Value $content -Encoding UTF8
    Write-Host "✓ Applied fixes to: $configPath" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Restart VS Code" -ForegroundColor White
    Write-Host "2. Run /doctor command to verify" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "No changes made - patterns not found or already fixed." -ForegroundColor Yellow
}