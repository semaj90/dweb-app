param(
    [switch]$InstallIfMissing
)

Write-Host "=== Buf Version Check ===" -ForegroundColor Cyan
try {
    $gopath = go env GOPATH 2>$null
} catch {
    Write-Host "Go toolchain not available. Install Go first." -ForegroundColor Red
    exit 1
}
$gobin = go env GOBIN 2>$null
if (-not $gobin) { $gobin = Join-Path $gopath 'bin' }
$bufPath = Join-Path $gobin 'buf.exe'
if (-not (Test-Path $bufPath)) {
    Write-Host "buf.exe not found at $bufPath" -ForegroundColor Yellow
    if ($InstallIfMissing) {
        Write-Host "Attempting: go install github.com/bufbuild/buf/cmd/buf@latest" -ForegroundColor Cyan
        go install github.com/bufbuild/buf/cmd/buf@latest
    } else {
        Write-Host "Run again with -InstallIfMissing to install." -ForegroundColor Yellow
        exit 2
    }
}
if (-not (Test-Path $bufPath)) {
    Write-Host "Installation failed or buf still missing." -ForegroundColor Red
    exit 3
}
& $bufPath --version
