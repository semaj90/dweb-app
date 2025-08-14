# Fix Claude Code Extension Issues
Write-Host "Fixing Claude Code VS Code Extension..." -ForegroundColor Green

# 1. Ensure Claude Code is enabled
Write-Host "`n1. Checking and enabling Claude Code extension..." -ForegroundColor Yellow
& 'C:\Users\james\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd' --enable-extension anthropic.claude-code

# 2. Add Claude Code specific settings to VS Code
Write-Host "`n2. Adding Claude Code settings to VS Code..." -ForegroundColor Yellow

$vscodeSettings = @"
{
  "claude-code.enabled": true,
  "claude-code.autoComplete": true,
  "claude-code.chat.enabled": true,
  "workbench.commandPalette.experimental.suggestCommands": true,
  "extensions.ignoreRecommendations": false
}
"@

$settingsPath = "C:\Users\james\AppData\Roaming\Code\User\settings.json"

# Backup existing settings
if (Test-Path $settingsPath) {
    Copy-Item $settingsPath "$settingsPath.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Write-Host "✅ Backed up existing settings" -ForegroundColor Green
}

# Read existing settings and merge with Claude settings
if (Test-Path $settingsPath) {
    $existingSettings = Get-Content $settingsPath -Raw | ConvertFrom-Json -AsHashtable
} else {
    $existingSettings = @{}
}

# Add Claude Code specific settings
$existingSettings["claude-code.enabled"] = $true
$existingSettings["claude-code.autoComplete"] = $true
$existingSettings["claude-code.chat.enabled"] = $true
$existingSettings["workbench.commandPalette.experimental.suggestCommands"] = $true

# Write back to settings file
$existingSettings | ConvertTo-Json -Depth 10 | Set-Content $settingsPath -Encoding UTF8
Write-Host "✅ Added Claude Code settings to VS Code" -ForegroundColor Green

# 3. Restart VS Code processes to ensure clean state
Write-Host "`n3. Restarting VS Code processes for clean state..." -ForegroundColor Yellow
Get-Process -Name "Code" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "✅ VS Code processes restarted" -ForegroundColor Green

# 4. Launch VS Code with a clean workspace
Write-Host "`n4. Launching VS Code with your project..." -ForegroundColor Yellow
$projectPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app"
& 'C:\Users\james\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd' $projectPath

Write-Host "`n✅ VS Code launched with your project" -ForegroundColor Green

Write-Host "`n🎉 Claude Code should now be working!" -ForegroundColor Cyan
Write-Host "To test:" -ForegroundColor White
Write-Host "  1. Open VS Code (should be launching now)" -ForegroundColor White
Write-Host "  2. Press Ctrl+Shift+P to open Command Palette" -ForegroundColor White
Write-Host "  3. Type 'Claude' - you should see Claude commands" -ForegroundColor White
Write-Host "  4. Try typing code and see Claude suggestions" -ForegroundColor White
Write-Host "  5. Look for Claude icon in the activity bar (left sidebar)" -ForegroundColor White

Write-Host "`nIf it still does not work:" -ForegroundColor Red
Write-Host "  - Check VS Code Output panel for Claude Code logs" -ForegroundColor White
Write-Host "  - Sign in to your Claude account if prompted" -ForegroundColor White
Write-Host "  - Reload VS Code window (Ctrl+Shift+P > Developer: Reload Window)" -ForegroundColor White