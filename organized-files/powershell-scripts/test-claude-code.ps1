# Test Claude Code Extension Functionality
Write-Host "Testing Claude Code VS Code Extension..." -ForegroundColor Green

# 1. Check if extension is installed and enabled
Write-Host "`n1. Checking Claude Code extension status..." -ForegroundColor Yellow
$claudeExtension = & 'C:\Users\james\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd' --list-extensions --show-versions | Select-String "anthropic.claude-code"
if ($claudeExtension) {
    Write-Host "✅ Claude Code extension installed: $claudeExtension" -ForegroundColor Green
} else {
    Write-Host "❌ Claude Code extension not found" -ForegroundColor Red
    exit 1
}

# 2. Check VS Code version compatibility
Write-Host "`n2. Checking VS Code version..." -ForegroundColor Yellow
$vscodeVersion = & 'C:\Users\james\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd' --version | Select-Object -First 1
Write-Host "✅ VS Code version: $vscodeVersion" -ForegroundColor Green

# 3. Check if Claude configuration exists
Write-Host "`n3. Checking Claude configuration..." -ForegroundColor Yellow
if (Test-Path "C:\Users\james\.claude\settings.json") {
    Write-Host "✅ Claude settings found" -ForegroundColor Green
    Get-Content "C:\Users\james\.claude\settings.json" | Write-Host
} else {
    Write-Host "❌ Claude settings not found" -ForegroundColor Red
}

# 4. Test opening VS Code with a file to trigger Claude
Write-Host "`n4. Testing VS Code launch with a test file..." -ForegroundColor Yellow
$testFile = "C:\Users\james\Desktop\deeds-web\deeds-web-app\test-file.js"
"// Test file for Claude Code extension`nconsole.log('Hello Claude!');" | Out-File -FilePath $testFile -Encoding UTF8

Write-Host "✅ Created test file: $testFile" -ForegroundColor Green
Write-Host "Opening VS Code with test file..." -ForegroundColor Yellow

# Launch VS Code with the test file
& 'C:\Users\james\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd' $testFile

Write-Host "`n✅ VS Code should now be open with Claude Code extension loaded" -ForegroundColor Green
Write-Host "Try pressing Ctrl+Shift+P and search for 'Claude' to see if the extension is working" -ForegroundColor Cyan

# 5. Check for common VS Code extension issues
Write-Host "`n5. Checking for common issues..." -ForegroundColor Yellow

# Check if there are any conflicting extensions
$extensions = & 'C:\Users\james\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd' --list-extensions
$potentialConflicts = $extensions | Select-String -Pattern "(copilot|gpt|ai|assistant)" | Where-Object { $_ -notmatch "claude" }

if ($potentialConflicts) {
    Write-Host "⚠️  Potential conflicting AI extensions found:" -ForegroundColor Orange
    $potentialConflicts | ForEach-Object { Write-Host "  - $_" -ForegroundColor Orange }
    Write-Host "These might interfere with Claude Code. Consider disabling them temporarily." -ForegroundColor Orange
} else {
    Write-Host "✅ No obvious conflicting extensions found" -ForegroundColor Green
}

Write-Host ""
Write-Host "If Claude Code still is not working, try:" -ForegroundColor Cyan
Write-Host "   1. Reload VS Code window (Developer: Reload Window)" -ForegroundColor White
Write-Host "   2. Check the Output panel for Claude Code logs" -ForegroundColor White
Write-Host "   3. Sign in to Claude if prompted" -ForegroundColor White
Write-Host "   4. Try the Command Palette and search for Claude" -ForegroundColor White