param(
  [string]$Destination
)

# Backup Go module cache (Windows/PowerShell)
# Usage:
#   pwsh -File scripts/backup-go-mod-cache.ps1
#   pwsh -File scripts/backup-go-mod-cache.ps1 -Destination C:\backups\go-mod-cache.zip

$ErrorActionPreference = 'Stop'

$goPath = (& go env GOPATH).Trim()
if (-not $goPath) {
  Write-Error "GOPATH not found. Ensure Go is installed and on PATH."
}

$modCache = Join-Path $goPath 'pkg/mod'
if (-not (Test-Path $modCache)) {
  Write-Error "Module cache directory not found: $modCache"
}

if (-not $Destination -or [string]::IsNullOrWhiteSpace($Destination)) {
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $Destination = Join-Path (Get-Location) "go-mod-cache-$stamp.zip"
}

Write-Host "Backing up module cache from: $modCache"
Write-Host "Destination: $Destination"

# Ensure destination directory exists
$destDir = Split-Path $Destination -Parent
if (-not (Test-Path $destDir)) {
  New-Item -ItemType Directory -Path $destDir | Out-Null
}

Compress-Archive -Path (Join-Path $modCache '*') -DestinationPath $Destination -Force

Write-Host "✅ Backup complete: $Destination"
