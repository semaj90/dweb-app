# Start vector proxy and ai stream in background, then run health checks
Set-Location 'C:\Users\james\Desktop\deeds-web\deeds-web-app'
if (-not (Test-Path .\logs)) { New-Item -ItemType Directory -Path .\logs | Out-Null }

# Start QUIC Vector Proxy
$vectorPath = Join-Path (Get-Location) 'go-microservice\bin\quic-vector-proxy.exe'
if (Test-Path $vectorPath) {
  $p1 = Start-Process -FilePath $vectorPath -RedirectStandardOutput (Join-Path (Get-Location) 'logs\quic-vector-proxy.log') -RedirectStandardError (Join-Path (Get-Location) 'logs\quic-vector-proxy.err.log') -PassThru
  Write-Host "Started quic-vector-proxy PID $($p1.Id)"
} else {
  Write-Host 'quic-vector-proxy.exe not found'
}

Start-Sleep -Milliseconds 300

# Start QUIC AI Stream
$aiPath = Join-Path (Get-Location) 'go-microservice\bin\quic-ai-stream.exe'
if (Test-Path $aiPath) {
  $p2 = Start-Process -FilePath $aiPath -RedirectStandardOutput (Join-Path (Get-Location) 'logs\quic-ai-stream.log') -RedirectStandardError (Join-Path (Get-Location) 'logs\quic-ai-stream.err.log') -PassThru
  Write-Host "Started quic-ai-stream PID $($p2.Id)"
} else {
  Write-Host 'quic-ai-stream.exe not found'
}

Start-Sleep -Seconds 1

Write-Host "Running health checks (curl.exe -k)..."
& curl.exe -k -sS https://localhost:8443/health | Write-Host "gateway: $(Get-Content -Raw -Encoding UTF8 -)" 2>$null
# For vector
try { $v = & curl.exe -k -sS https://localhost:8445/health } catch { $v = "(curl failed) $($_)" }
Write-Host "vector: $v"
try { $a = & curl.exe -k -sS https://localhost:8447/health } catch { $a = "(curl failed) $($_)" }
Write-Host "ai-stream: $a"

Write-Host 'Done.'
