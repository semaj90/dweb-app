# Start QUIC binaries (if present) and check /health endpoints
Set-Location 'C:\Users\james\Desktop\deeds-web\deeds-web-app'
Write-Host '▶ Starting QUIC executables (if present)'
$bins = @(
  'go-microservice\\bin\\quic-gateway.exe',
  'go-microservice\\bin\\quic-vector-proxy.exe',
  'go-microservice\\bin\\quic-ai-stream.exe',
  'go-microservice\\bin\\rag-quic-proxy.exe'
)
foreach ($b in $bins) {
  if (Test-Path $b) {
    try {
      $p = Start-Process -FilePath $b -PassThru -WindowStyle Hidden -ErrorAction Stop
      Write-Host "  Started: $b (PID $($p.Id))"
    } catch {
      Write-Host ("  Failed to start {0}: {1}" -f $b, $_.Exception.Message)
    }
  } else {
    Write-Host "  Not found: $b"
  }
}

Start-Sleep -Seconds 3
[System.Net.ServicePointManager]::ServerCertificateValidationCallback = { $true }
$urls = @(
  'https://localhost:8443/health',
  'https://localhost:8444/health',
  'https://localhost:8445/health',
  'https://localhost:8446/health',
  'https://localhost:8447/health',
  'https://localhost:8448/health'
)
foreach ($u in $urls) {
  Write-Host "▶ Checking $u"
  try {
    $r = Invoke-RestMethod -Uri $u -TimeoutSec 5
    Write-Host "  OK: $($r | ConvertTo-Json -Compress)"
  } catch {
    Write-Host "  FAIL: $($_.Exception.Message)"
  }
}
