Param(
  [string]$Discovery = "scripts/orchestration/service-discovery.json",
  [string]$Out = ".vscode/orchestration-health.json"
)

$ErrorActionPreference = "SilentlyContinue"
$services = Get-Content $Discovery | ConvertFrom-Json
$result = @()
foreach ($s in $services.services) {
  $entry = [ordered]@{ name=$s.name; port=$s.port; category=$s.category; status="unknown" }
  if ($s.protocol -eq 'http' -and $s.health) {
    try {
      $url = "http://$($s.host):$($s.port)$($s.health)"
      $resp = Invoke-WebRequest -Uri $url -Method GET -TimeoutSec 2
      if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) { $entry.status = 'healthy' } else { $entry.status = 'unhealthy' }
    } catch { $entry.status = 'down' }
  } else {
    $r = Test-NetConnection -ComputerName $s.host -Port $s.port
    $entry.status = if ($r.TcpTestSucceeded) { 'open' } else { 'closed' }
  }
  $result += $entry
}

$summary = [ordered]@{ ts = (Get-Date).ToString('s'); environment = $services.environment; results = $result }
$summary | ConvertTo-Json -Depth 5 | Out-File -FilePath $Out -Encoding utf8
Write-Output "Health written to $Out"
