# Stop owners of UDP ports 8445 and 8447, restart vector + ai services, run health checks
Set-Location 'C:\Users\james\Desktop\deeds-web\deeds-web-app'
if (-not (Test-Path .\logs)) { New-Item -ItemType Directory -Path .\logs | Out-Null }
$ports = @(8445,8447)
foreach ($port in $ports) {
  Write-Host "Checking UDP :$port..."
  $lines = netstat -ano -p UDP | Select-String (":" + $port)
  if (-not $lines) { Write-Host "  No UDP listener on :$port"; continue }
  foreach ($m in $lines) {
    $line = $m.Line.Trim()
    Write-Host "  NETSTAT: $line"
    $cols = -split $line
    $foundPid = $cols[-1]
    Write-Host ("  Found PID {0} for port {1}" -f $foundPid, $port)
    try {
      $proc = Get-Process -Id $foundPid -ErrorAction Stop
      Write-Host ("  Process: {0} (Id={1})" -f $proc.ProcessName, $proc.Id)
    } catch {
      Write-Host ("  No process found with PID {0}" -f $foundPid)
    }
    Write-Host ("  Stopping PID {0}..." -f $foundPid)
    try {
      Stop-Process -Id $foundPid -Force -ErrorAction Stop
      Write-Host ("  Stopped PID {0}" -f $foundPid)
    } catch {
      Write-Host ("  Failed to stop PID {0}: {1}" -f $foundPid, $_.Exception.Message)
    }
  }
}
Start-Sleep -Seconds 1

# Start services
$started = @{}
$vectorPath = Join-Path (Get-Location) 'go-microservice\bin\quic-vector-proxy.exe'
$aiPath = Join-Path (Get-Location) 'go-microservice\bin\quic-ai-stream.exe'
if (Test-Path $vectorPath) {
  $p = Start-Process -FilePath $vectorPath -RedirectStandardOutput (Join-Path (Get-Location) 'logs\quic-vector-proxy.log') -RedirectStandardError (Join-Path (Get-Location) 'logs\quic-vector-proxy.err.log') -PassThru
  Write-Host ("Started quic-vector-proxy PID {0}" -f $p.Id)
  $started['vector'] = $p.Id
} else { Write-Host 'quic-vector-proxy.exe not found' }
Start-Sleep -Milliseconds 300
if (Test-Path $aiPath) {
  $p2 = Start-Process -FilePath $aiPath -RedirectStandardOutput (Join-Path (Get-Location) 'logs\quic-ai-stream.log') -RedirectStandardError (Join-Path (Get-Location) 'logs\quic-ai-stream.err.log') -PassThru
  Write-Host ("Started quic-ai-stream PID {0}" -f $p2.Id)
  $started['ai'] = $p2.Id
} else { Write-Host 'quic-ai-stream.exe not found' }

Start-Sleep -Seconds 1

# Health checks
Write-Host "Health checks (curl.exe -k):"
function DoCurl($url, $label) {
  try {
    $out = & curl.exe -k -sS $url 2>&1
    if ($LASTEXITCODE -eq 0) { Write-Host ("{0}: {1}" -f $label, $out) } else { Write-Host ("{0}: curl failed (exit {1}) - {2}" -f $label, $LASTEXITCODE, $out) }
  } catch {
    Write-Host ("{0}: curl exception - {1}" -f $label, $_.Exception.Message)
  }
}

DoCurl 'https://localhost:8443/health' 'gateway'
DoCurl 'https://localhost:8445/health' 'vector'
DoCurl 'https://localhost:8447/health' 'ai-stream'

Write-Host 'Done.'
