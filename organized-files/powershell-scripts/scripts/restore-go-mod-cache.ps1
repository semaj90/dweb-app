param(
  [Parameter(Mandatory=$true)][string]$Archive
)

# Restore Go module cache (Windows/PowerShell)
# Usage:
#   pwsh -File scripts/restore-go-mod-cache.ps1 -Archive path\to\go-mod-cache.zip

$ErrorActionPreference = 'Stop'

$goPath = (& go env GOPATH).Trim()
if (-not $goPath) {
  Write-Error "GOPATH not found. Ensure Go is installed and on PATH."
}

$modCache = Join-Path $goPath 'pkg/mod'
if (-not (Test-Path $Archive)) {
  Write-Error "Archive not found: $Archive"
}

Write-Host "Restoring module cache to: $modCache"

# Ensure mod cache dir exists
if (-not (Test-Path $modCache)) {
  New-Item -ItemType Directory -Path $modCache | Out-Null
}

# Extract archive
Expand-Archive -Path $Archive -DestinationPath $modCache -Force

Write-Host "✅ Restore complete."
