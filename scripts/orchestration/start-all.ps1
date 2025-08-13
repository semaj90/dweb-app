Param(
  [switch]$DryRun = $true,
  [string]$Discovery = "scripts/orchestration/service-discovery.json",
  [string]$Deps = "scripts/orchestration/dependencies.json"
)

$services = Get-Content $Discovery | ConvertFrom-Json
$deps = Get-Content $Deps | ConvertFrom-Json

function Start-Info($msg) { Write-Host "[orchestrator] $msg" }

# Simple topological-order-ish plan based on dependencies
$ordered = New-Object System.Collections.Generic.List[System.Object]
$remaining = New-Object 'System.Collections.Generic.List[pscustomobject]'
$remaining.AddRange($services.services)

while ($remaining.Count -gt 0) {
  $progress = $false
  foreach ($svc in @($remaining)) {
    $dn = $deps.dependencies[$svc.name]
    if (-not $dn -or @($dn | Where-Object { $_ -in $ordered.name }).Count -eq $dn.Count) {
      $ordered.Add($svc)
      $remaining.Remove($svc)
      $progress = $true
    }
  }
  if (-not $progress) { break }
}

Start-Info "Planned order: $($ordered | ForEach-Object { $_.name } -join ', ')"

if ($DryRun) { Start-Info "Dry-Run specified. Not starting any processes."; exit 0 }

# NOTE: Real start commands are intentionally omitted to keep placeholders safe.
# You can map each service name here to the proper VS Code task or command.
foreach ($svc in $ordered) {
  Start-Info "Would start: $($svc.name) on $($svc.host):$($svc.port)"
}
