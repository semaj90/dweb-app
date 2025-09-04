# QUIC diagnostics: tail logs and show listeners
Set-Location 'C:\Users\james\Desktop\deeds-web\deeds-web-app'
$logs = @(
  'logs\\quic-gateway.log', 'logs\\quic-gateway.err.log',
  'logs\\quic-vector-proxy.log','logs\\quic-vector-proxy.err.log',
  'logs\\quic-ai-stream.log','logs\\quic-ai-stream.err.log',
  'logs\\rag-quic-proxy.log','logs\\rag-quic-proxy.err.log'
)
foreach ($f in $logs) {
  if (Test-Path $f) {
    Write-Host "=== $f ==="
    Get-Content $f -Tail 200 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
  } else {
    Write-Host "--- missing: $f"
  }
}

# Show UDP listeners for 8443-8448
Write-Host '\n=== UDP listeners (ports 8443-8448) ==='
$udp = netstat -ano -p UDP | Select-String ':8443|:8444|:8445|:8446|:8447|:8448'
if ($udp) { $udp | ForEach-Object { Write-Host $_.Line } } else { Write-Host 'No UDP listeners in range' }

# Show TCP listeners for 8443-8448
Write-Host '\n=== TCP listeners (ports 8443-8448) ==='
$tcp = netstat -ano -p TCP | Select-String ':8443|:8444|:8445|:8446|:8447|:8448'
if ($tcp) { $tcp | ForEach-Object { Write-Host $_.Line } } else { Write-Host 'No TCP listeners in range' }

# Map PIDs found to process names
$foundPIDs = @()
($udp + $tcp) | ForEach-Object {
  $line = $_.Line.Trim()
  $cols = -split $line
  $pid = $cols[-1]
  if ($pid -and ($foundPIDs -notcontains $pid)) { $foundPIDs += $pid }
}

if ($foundPIDs.Count -gt 0) {
  Write-Host '\n=== PID -> Process mapping ==='
  foreach ($pid in $foundPIDs) {
    try {
      $proc = Get-Process -Id $pid -ErrorAction Stop
      Write-Host ("PID {0} -> {1} (Id={2})" -f $pid, $proc.ProcessName, $proc.Id)
    } catch {
      Write-Host ("PID {0} -> (no running process)" -f $pid)
    }
  }
} else { Write-Host '\nNo PIDs found for those ports' }
