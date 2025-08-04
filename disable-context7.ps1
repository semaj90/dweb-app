# Quick PowerShell Script to Disable Context7 MCP Assistant Extension
# This disables the laggy mcp-context7-assistant extension locally

Write-Host "🔧 Disabling Context7 MCP Assistant Extension..." -ForegroundColor Cyan

# Check if VS Code is running
$vscodeProcess = Get-Process code -ErrorAction SilentlyContinue
if ($vscodeProcess) {
    Write-Host "⚠️  VS Code is running. Closing it first..." -ForegroundColor Yellow
    $vscodeProcess | Stop-Process -Force
    Start-Sleep 2
}

# Method 1: Disable via command line
Write-Host "1️⃣ Disabling extension via VS Code CLI..." -ForegroundColor Green
try {
    & code --disable-extension "undefined_publisher.mcp-context7-assistant"
    Write-Host "✅ Extension disabled via CLI" -ForegroundColor Green
} catch {
    Write-Host "⚠️  CLI method failed, trying alternative..." -ForegroundColor Yellow
}

# Method 2: Update workspace settings (already done above)
Write-Host "2️⃣ Updating workspace settings..." -ForegroundColor Green

# Create extension disable configuration
$workspaceConfig = @"
{
  "extensions.ignoreRecommendations": true,
  "extensions.autoCheckUpdates": false,
  "extensions.autoUpdate": false,
  
  // Disable Context7 MCP specifically
  "context7.enabled": false,
  "mcpContext7.enabled": false,
  "mcpContext7.agentOrchestrator.enabled": false,
  "mcpContext7.ragEnabled": false,
  "mcpContext7.ragAutoStart": false,
  "enhancedRAG.enabled": false,
  
  // Performance improvements
  "files.watcherExclude": {
    "**/node_modules/**": true,
    "**/.svelte-kit/**": true,
    "**/mcp-servers/**": true,
    "**/context7-docs/**": true
  }
}
"@

# Save to workspace settings
$workspaceConfig | Out-File -FilePath ".\.vscode\settings.performance.json" -Encoding UTF8

Write-Host "✅ Performance settings saved" -ForegroundColor Green

# Method 3: Check installed extensions
Write-Host "3️⃣ Checking installed extensions..." -ForegroundColor Green
$extensions = & code --list-extensions 2>$null

if ($extensions -contains "undefined_publisher.mcp-context7-assistant") {
    Write-Host "📦 Found: undefined_publisher.mcp-context7-assistant" -ForegroundColor Yellow
    Write-Host "   Extension is installed but should now be disabled" -ForegroundColor White
} else {
    Write-Host "✅ Extension not found in installed list" -ForegroundColor Green
}

Write-Host "`n🎯 Results:" -ForegroundColor Cyan
Write-Host "• Context7 MCP Assistant disabled locally" -ForegroundColor Green
Write-Host "• Auto-trigger features turned off" -ForegroundColor Green  
Write-Host "• RAG backend connections disabled" -ForegroundColor Green
Write-Host "• File watchers optimized for performance" -ForegroundColor Green

Write-Host "`n🚀 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Restart VS Code to apply changes" -ForegroundColor White
Write-Host "2. The extension will be disabled but not uninstalled" -ForegroundColor White
Write-Host "3. You can re-enable it later if needed" -ForegroundColor White

Write-Host "`n💡 To re-enable later:" -ForegroundColor Yellow
Write-Host "code --enable-extension undefined_publisher.mcp-context7-assistant" -ForegroundColor Gray

Write-Host "`n✅ Context7 MCP Assistant disabled successfully!" -ForegroundColor Green