param(
  [int]$IntervalSeconds = 20
)

Write-Host "Event loop started: TS check + Go SIMD build every $IntervalSeconds seconds" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray

while ($true) {
  $tsJob = Start-Job -Name "ts-check" -ScriptBlock {
    param($frontendPath)
    try {
      Push-Location $frontendPath
  Write-Host "[TS] Running ultra-fast check..." -ForegroundColor Yellow
      npm run check:ultra-fast
    } finally {
      Pop-Location
    }
  } -ArgumentList (Join-Path $PSScriptRoot "..\sveltekit-frontend")

  $goJob = Start-Job -Name "go-simd" -ScriptBlock {
    param($servicePath)
    try {
      Push-Location $servicePath
  Write-Host "[Go] Building SIMD parser (legacy tag)..." -ForegroundColor Yellow
      if (-not (Test-Path bin)) { New-Item -ItemType Directory -Path bin | Out-Null }
      go mod tidy
      go build -tags legacy -o ./bin/simd-parser.exe ./simd_parser.go
      if (Test-Path './bin/simd-parser.exe') {
  Write-Host "[Go] simd-parser.exe built" -ForegroundColor Green
      } else {
  Write-Host "[Go] simd-parser.exe missing after build" -ForegroundColor Red
      }
    } finally {
      Pop-Location
    }
  } -ArgumentList (Join-Path $PSScriptRoot "..\go-microservice")

  Wait-Job -Job $tsJob, $goJob | Out-Null

  Write-Host "--- TS output ---" -ForegroundColor DarkCyan
  Receive-Job $tsJob | ForEach-Object { $_ }
  Write-Host "--- Go output ---" -ForegroundColor DarkCyan
  Receive-Job $goJob | ForEach-Object { $_ }

  Remove-Job $tsJob, $goJob -Force -ErrorAction SilentlyContinue

  Write-Host "Sleeping $IntervalSeconds seconds..." -ForegroundColor DarkGray
  Start-Sleep -Seconds $IntervalSeconds
}
