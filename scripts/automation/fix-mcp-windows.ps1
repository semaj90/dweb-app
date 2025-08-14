# PowerShell script to fix MCP server configurations for Windows
# This adds the required 'cmd /c' wrapper for Windows execution

$configPath = "$env:USERPROFILE\.claude.json"

# Read the current configuration
$config = Get-Content $configPath -Raw | ConvertFrom-Json

# Function to fix Windows command execution
function Fix-MCPServer {
    param(
        [Parameter(Mandatory=$true)]
        $server,
        [Parameter(Mandatory=$true)]
        [string]$serverName
    )
    
    if ($server -and $server.command) {
        # Check if command starts with npx and doesn't have cmd /c wrapper
        if ($server.command -match "^npx" -and $server.command -notmatch "^cmd") {
            Write-Host "Fixing $serverName server command..." -ForegroundColor Yellow
            $server.command = "cmd /c $($server.command)"
            return $true
        }
    }
    return $false
}

# Fix each MCP server configuration
$serversToFix = @('puppeteer', 'postgres', 'filesystem', 'context7')
$fixed = $false

foreach ($serverName in $serversToFix) {
    if ($config.mcpServers.$serverName) {
        if (Fix-MCPServer -server $config.mcpServers.$serverName -serverName $serverName) {
            $fixed = $true
        }
    }
}

if ($fixed) {
    # Backup the original configuration
    $backupPath = "$configPath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item $configPath $backupPath
    Write-Host "Created backup at: $backupPath" -ForegroundColor Green
    
    # Save the fixed configuration
    $config | ConvertTo-Json -Depth 100 | Set-Content $configPath -Encoding UTF8
    Write-Host "Fixed MCP server configurations in $configPath" -ForegroundColor Green
    Write-Host "Please restart VS Code for changes to take effect." -ForegroundColor Cyan
} else {
    Write-Host "No fixes needed - configurations appear to be correct." -ForegroundColor Green
}

# Display current configuration for verification
Write-Host "`nCurrent MCP server commands:" -ForegroundColor Cyan
foreach ($serverName in $serversToFix) {
    if ($config.mcpServers.$serverName) {
        Write-Host "  $serverName: $($config.mcpServers.$serverName.command)" -ForegroundColor Gray
    }
}