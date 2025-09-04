param(
  [int]$Interval = 5,
  [int]$Port = $(if($env:GPU_ORCHESTRATOR_PORT){$env:GPU_ORCHESTRATOR_PORT}else{8095})
)
Write-Host "📊 Tailing /metrics every $Interval s (Ctrl+C to stop)" -ForegroundColor Cyan
while($true){
  try {
    $content = (Invoke-WebRequest -Uri "http://localhost:$Port/metrics" -TimeoutSec 3).Content
    $lines = $content -split "`n" | Where-Object { $_ -match 'gpu_(active_tasks|queue_depth|completed_tasks|avg_task_latency_ms)' }
    $ts = (Get-Date).ToString('HH:mm:ss')
    Write-Host "[$ts] $($lines -join ' | ')"
  } catch { Write-Host "metrics fetch failed" -ForegroundColor Red }
  Start-Sleep -Seconds $Interval
}
