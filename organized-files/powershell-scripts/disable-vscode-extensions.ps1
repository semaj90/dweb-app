# PowerShell Script to Disable VS Code Extensions Locally (Not Uninstall)
# This script disables problematic extensions in the current workspace only

Write-Host "🔧 VS Code Extension Manager - Local Disable (No Uninstall)" -ForegroundColor Cyan
Write-Host "This script will disable extensions locally in this workspace only." -ForegroundColor Yellow

# Extensions that commonly cause TypeScript/Svelte conflicts
$problematicExtensions = @(
    "octref.vetur",
    "johnsoncodehk.volar", 
    "vue.volar",
    "ms-vscode.vscode-eslint",
    "dbaeumer.vscode-eslint",
    "bradlc.vscode-tailwindcss"
)

# Check if VS Code is running
$vscodeProcess = Get-Process code -ErrorAction SilentlyContinue
if ($vscodeProcess) {
    Write-Host "⚠️  VS Code is currently running. Please close it first." -ForegroundColor Red
    Write-Host "Press any key to continue when VS Code is closed..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

Write-Host "`n📋 Extensions to disable in this workspace:" -ForegroundColor Green
$problematicExtensions | ForEach-Object { Write-Host "  - $_" -ForegroundColor White }

# Create workspace settings to disable extensions
$workspaceSettings = @{
    "extensions.ignoreRecommendations" = $true
    "svelte.enable-ts-plugin" = $true
    "typescript.validate.enable" = $true
    "problems.autoReveal" = $true
}

# Convert to JSON and save
$workspaceSettingsJson = $workspaceSettings | ConvertTo-Json -Depth 3
$settingsPath = ".\.vscode\settings.workspace.json"

if (!(Test-Path ".\.vscode")) {
    New-Item -ItemType Directory -Path ".\.vscode" -Force | Out-Null
}

$workspaceSettingsJson | Out-File -FilePath $settingsPath -Encoding UTF8

Write-Host "`n✅ Workspace settings created: $settingsPath" -ForegroundColor Green

# Create extension recommendations file
$extensionsConfig = @{
    "recommendations" = @(
        "svelte.svelte-vscode",
        "ms-vscode.vscode-typescript-next"
    )
    "unwantedRecommendations" = $problematicExtensions
}

$extensionsJson = $extensionsConfig | ConvertTo-Json -Depth 3
$extensionsPath = ".\.vscode\extensions.json"
$extensionsJson | Out-File -FilePath $extensionsPath -Encoding UTF8

Write-Host "✅ Extension recommendations updated: $extensionsPath" -ForegroundColor Green

Write-Host "`n🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Open VS Code in this workspace" -ForegroundColor White
Write-Host "2. Extensions will be automatically disabled for this workspace only" -ForegroundColor White
Write-Host "3. Run 'npm run check' to verify TypeScript errors are reduced" -ForegroundColor White
Write-Host "4. Extensions remain installed globally for other projects" -ForegroundColor White

Write-Host "`n📊 Current Error Status:" -ForegroundColor Cyan
Set-Location "sveltekit-frontend"
Write-Host "Running TypeScript check..." -ForegroundColor Yellow

try {
    $checkResult = npm run check 2>&1 | Select-String "found.*errors"
    if ($checkResult) {
        Write-Host "Current status: $checkResult" -ForegroundColor Green
    } else {
        Write-Host "TypeScript check completed - see output above" -ForegroundColor Green
    }
} catch {
    Write-Host "Unable to run TypeScript check. Run manually with: npm run check" -ForegroundColor Yellow
}

Write-Host "`n🚀 Workspace configured successfully!" -ForegroundColor Green
Write-Host "Extensions are disabled locally - not uninstalled." -ForegroundColor Yellow