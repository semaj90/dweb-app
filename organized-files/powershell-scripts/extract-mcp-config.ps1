# Extract and display MCP server configurations from .claude.json

$configPath = "$env:USERPROFILE\.claude.json"

Write-Host "Extracting MCP server configurations..." -ForegroundColor Cyan
Write-Host "File size: $((Get-Item $configPath).Length / 1MB) MB" -ForegroundColor Yellow
Write-Host ""

# Read the file and find mcpServers section
$content = Get-Content $configPath -Raw

# Use simpler approach to find servers
$servers = @('puppeteer', 'postgres', 'filesystem', 'context7')

Write-Host "Current MCP Server Configurations:" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green

foreach ($server in $servers) {
    # Build regex pattern for each server
    $pattern = "`"$server`"\s*:\s*\{[^}]*`"command`"\s*:\s*`"([^`"]+)`""
    
    if ($content -match $pattern) {
        $command = $matches[1]
        
        Write-Host ""
        Write-Host "${server}:" -ForegroundColor Yellow
        Write-Host "  Current: $command" -ForegroundColor Gray
        
        if ($command -like "npx*" -and $command -notlike "cmd /c*") {
            Write-Host "  Fixed:   cmd /c $command" -ForegroundColor Green
            Write-Host "  Status:  NEEDS FIXING" -ForegroundColor Red
        } else {
            Write-Host "  Status:  OK" -ForegroundColor Green
        }
    } else {
        Write-Host ""
        Write-Host "${server}:" -ForegroundColor Yellow
        Write-Host "  Status:  NOT FOUND" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "===================================" -ForegroundColor Green
Write-Host ""
Write-Host "To fix the configurations, you need to:" -ForegroundColor Cyan
Write-Host "1. Edit $configPath" -ForegroundColor White
Write-Host "2. Find each server configuration listed above" -ForegroundColor White
Write-Host "3. Add 'cmd /c ' before any 'npx' commands" -ForegroundColor White
Write-Host "4. Save the file and restart VS Code" -ForegroundColor White