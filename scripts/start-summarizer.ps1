param(
  [int]$Port = 8091,
  [string]$OllamaBaseUrl = 'http://localhost:11434',
  [string]$Model = 'llama3.1:8b',
  [int]$MaxConcurrency = 2,
  [int]$AcquireTimeoutMs = 1500,
  [int]$L1TtlSec = 60,
  [int]$L1Max = 512,
  [int]$CacheTtlSec = 600,
  [string]$RedisAddr
)

$ErrorActionPreference = 'Stop'

Write-Host "[summarizer] configuring environment..." -ForegroundColor Cyan
$env:SUMMARIZER_HTTP_PORT = "$Port"
$env:OLLAMA_BASE_URL = "$OllamaBaseUrl"
$env:OLLAMA_MODEL = "$Model"
$env:SUMMARIZER_MAX_CONCURRENCY = "$MaxConcurrency"
$env:SUMMARIZER_ACQUIRE_TIMEOUT_MS = "$AcquireTimeoutMs"
$env:SUMMARIZER_L1_TTL_SEC = "$L1TtlSec"
$env:SUMMARIZER_L1_MAX = "$L1Max"
$env:SUMMARIZER_CACHE_TTL_SEC = "$CacheTtlSec"
if ($RedisAddr) { $env:REDIS_ADDR = "$RedisAddr" }

# Resolve executable path
$repoRoot = Split-Path -Parent $PSScriptRoot
$exePath = Join-Path $repoRoot 'go-microservice\bin\summarizer-http.exe'
if (-not (Test-Path $exePath)) {
  Write-Host "[summarizer] executable not found at $exePath, attempting build..." -ForegroundColor Yellow
  Push-Location (Join-Path $repoRoot 'go-microservice')
  try {
    if (-not (Test-Path 'bin')) { New-Item -ItemType Directory -Path 'bin' | Out-Null }
    go mod tidy | Out-Null
    go build -o ./bin/summarizer-http.exe ./cmd/summarizer-service
  }
  finally { Pop-Location }
}

if (-not (Test-Path $exePath)) {
  Write-Error "[summarizer] build failed or binary missing at $exePath"
  exit 1
}

Write-Host "[summarizer] starting service on port $Port ..." -ForegroundColor Green
Start-Process -NoNewWindow -WindowStyle Hidden -FilePath $exePath -WorkingDirectory (Split-Path -Parent $exePath)

Start-Sleep -Seconds 1

# Quick health probe
try {
  $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$Port/health" -TimeoutSec 3
  if ($resp.StatusCode -eq 200) {
    Write-Host "[summarizer] up: http://localhost:$Port" -ForegroundColor Green
    exit 0
  } else {
    Write-Warning "[summarizer] health returned status $($resp.StatusCode)"
    exit 0
  }
}
catch {
  Write-Warning "[summarizer] health check failed. It may still be starting."
  exit 0
}
