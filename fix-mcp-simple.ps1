# Simple MCP fixer for Windows

$configPath = "$env:USERPROFILE\.claude.json"

Write-Host "Fixing MCP servers for Windows..." -ForegroundColor Cyan

# Read config
$content = Get-Content $configPath -Raw

# Create backup
$backupPath = "$configPath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item $configPath $backupPath
Write-Host "Backup created: $backupPath" -ForegroundColor Green

# Fix postgres and context7 servers
$originalContent = $content

# Fix postgres server - find "postgres" section with npx command
$content = $content -replace '("postgres"[^}]*"command"\s*:\s*")npx', '${1}cmd /c npx'

# Fix context7 server - find "context7" section with npx command  
$content = $content -replace '("context7"[^}]*"command"\s*:\s*")npx', '${1}cmd /c npx'

# Save changes
Set-Content -Path $configPath -Value $content -Encoding UTF8

Write-Host "Fixed MCP server configurations" -ForegroundColor Green
Write-Host "Please restart VS Code and run /doctor to verify" -ForegroundColor Yellow