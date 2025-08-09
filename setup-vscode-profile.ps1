# VS Code PowerShell Profile Setup
cd 'C:\Users\james\Desktop\deeds-web\deeds-web-app'

$src = Join-Path (Get-Location) '.vscode/Microsoft.VSCode_profile.ps1'
$destPS7 = Join-Path $env:USERPROFILE 'Documents/PowerShell/Microsoft.VSCode_profile.ps1'
$destWinPS = Join-Path $env:USERPROFILE 'Documents/WindowsPowerShell/Microsoft.VSCode_profile.ps1'

if (Test-Path $src) {
    New-Item -ItemType Directory -Force -Path (Split-Path $destPS7) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path $destWinPS) | Out-Null
    Copy-Item -Force $src $destPS7
    Copy-Item -Force $src $destWinPS
    Write-Host 'VS Code PowerShell profile copied to:'
    Write-Host $destPS7
    Write-Host $destWinPS
} else {
    Write-Host 'Source profile not found at:'
    Write-Host $src
}