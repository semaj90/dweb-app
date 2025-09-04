param(
  [int]$DelayMs = 1500,
  [int]$Port = $(if($env:GPU_ORCHESTRATOR_PORT){$env:GPU_ORCHESTRATOR_PORT}else{8095})
)
Write-Host "🧪 Submitting long-running task (will cancel after $DelayMs ms)" -ForegroundColor Cyan
$payload = @{ type='embedding'; priority=3; input='long_task'; metadata=@{ delay_ms = 4000 } } | ConvertTo-Json -Depth 5
$resp = Invoke-RestMethod -Uri "http://localhost:$Port/gpu/task" -Method Post -Body $payload -ContentType 'application/json'
$id = $resp.task_id
Write-Host "➡️ Submitted $id" -ForegroundColor Green
Start-Sleep -Milliseconds $DelayMs
Write-Host "🛑 Cancelling $id" -ForegroundColor Yellow
try {
  $cancel = Invoke-RestMethod -Uri "http://localhost:$Port/gpu/task/cancel?id=$id"
  Write-Host ("Result: " + $cancel.status) -ForegroundColor Magenta
} catch {
  Write-Host "Cancel failed: $($_.Exception.Message)" -ForegroundColor Red
}
