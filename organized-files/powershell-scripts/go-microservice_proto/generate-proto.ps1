Param(
    [switch]$Verbose
)

Write-Host "🔧 Proto Generation Starting" -ForegroundColor Cyan

# Resolve GOPATH/bin buf first (new install via go install), fallback to PATH
$goPath = (go env GOPATH) 2>$null
$bufCandidates = @()
if ($goPath) { $bufCandidates += (Join-Path $goPath 'bin/buf.exe') }
$bufCandidates += (Get-Command buf -ErrorAction SilentlyContinue | ForEach-Object { $_.Source })
$buf = $bufCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $buf) {
    Write-Host "❌ Could not locate buf binary. Install with: go install github.com/bufbuild/buf/cmd/buf@latest" -ForegroundColor Red
    exit 1
}

if ($Verbose) { Write-Host ("Using buf binary: " + $buf) -ForegroundColor Yellow }

# Ensure we are in the script directory (proto root)
Set-Location -Path $PSScriptRoot

if (-not (Test-Path './buf.yaml')) { Write-Host '❌ buf.yaml not found here; aborting.' -ForegroundColor Red; exit 1 }
if (-not (Test-Path './buf.gen.yaml')) { Write-Host '❌ buf.gen.yaml not found here; aborting.' -ForegroundColor Red; exit 1 }

# Run lint first (non-fatal)
& $buf lint
if ($LASTEXITCODE -ne 0) { Write-Host '⚠️ buf lint reported issues (continuing).' -ForegroundColor Yellow }

# Generate
& $buf generate
if ($LASTEXITCODE -ne 0) { Write-Host '❌ buf generate failed.' -ForegroundColor Red; exit $LASTEXITCODE }

# Post-check
if (Test-Path './gen/go') { Write-Host '✅ Go stubs generated: gen/go' -ForegroundColor Green } else { Write-Host '⚠️ gen/go missing' -ForegroundColor Yellow }
if (Test-Path './gen/openapiv2') { Write-Host '✅ OpenAPI spec generated: gen/openapiv2' -ForegroundColor Green } else { Write-Host '⚠️ gen/openapiv2 missing' -ForegroundColor Yellow }

Write-Host '🏁 Proto Generation Complete' -ForegroundColor Cyan
