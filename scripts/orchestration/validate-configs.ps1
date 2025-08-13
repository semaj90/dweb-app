$files = @(
  'scripts/orchestration/service-discovery.json',
  'scripts/orchestration/dependencies.json',
  'scripts/orchestration/message-routing.json',
  'scripts/orchestration/logging.json'
)
$ok = $true
foreach ($f in $files) {
  try { Get-Content $f | ConvertFrom-Json | Out-Null; Write-Output "✅ $f valid" }
  catch { Write-Output "❌ $f invalid: $($_.Exception.Message)"; $ok = $false }
}
if (-not $ok) { exit 1 }
