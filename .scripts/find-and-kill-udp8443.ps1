# Find and stop any process owning UDP :8443
$ports = ':8443'
$udp = netstat -ano -p UDP | Select-String $ports
if (-not $udp) {
  Write-Host "No UDP listener found on :8443"
  exit 0
}
foreach ($match in $udp) {
  $line = $match.Line.Trim()
  Write-Host "NETSTAT: $line"
  $cols = -split $line
  $foundPid = $cols[-1]
  Write-Host ("Found PID {0}" -f $foundPid)
  try {
    $proc = Get-Process -Id $foundPid -ErrorAction Stop
    Write-Host ("Process: {0} (Id={1})" -f $proc.ProcessName, $proc.Id)
  } catch {
    Write-Host ("No process found with PID {0}" -f $foundPid)
  }
  Write-Host ("Stopping PID {0}..." -f $foundPid)
  try {
    Stop-Process -Id $foundPid -Force -ErrorAction Stop
    Write-Host ("Stopped PID {0}" -f $foundPid)
  } catch {
    Write-Host ("Failed to stop PID {0}: {1}" -f $foundPid, $_.Exception.Message)
  }
}
Start-Sleep -Seconds 1
Write-Host "Verify UDP :8443 listeners after stop:"
$after = netstat -ano -p UDP | Select-String ':8443'
if (-not $after) { Write-Host "No UDP listener found on :8443" } else { $after | ForEach-Object { Write-Host $_.Line } }
