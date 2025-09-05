param(
  [int]$Count = 15,
  [int]$MaxPriority = 10,
  [int]$DelayBetweenMs = 80,
  [int]$Port = $(if($env:GPU_ORCHESTRATOR_PORT){$env:GPU_ORCHESTRATOR_PORT}else{8095})
)
Write-Host "🚀 Submitting $Count tasks (random priority 1..$MaxPriority) to orchestrator on :$Port" -ForegroundColor Cyan
$submitted = @()
for($i=1; $i -le $Count; $i++) {
  $priority = Get-Random -Minimum 1 -Maximum ($MaxPriority+1)
  $type = (Get-Random -InputObject @('embedding','inference','tensor_op'))
  $payload = @{ type = $type; priority = $priority; input = "payload_$i"; metadata = @{ source = 'burst'; idx = $i } } | ConvertTo-Json -Depth 5
  try {
    $resp = Invoke-RestMethod -Uri "http://localhost:$Port/gpu/task" -Method Post -Body $payload -ContentType 'application/json'
    $submitted += $resp.task_id
    Write-Host ("  #$i -> $($resp.task_id) ($type p=$priority)") -ForegroundColor Green
  } catch {
    Write-Host ("  #$i -> submit failed: $($_.Exception.Message)") -ForegroundColor Red
  }
  Start-Sleep -Milliseconds $DelayBetweenMs
}
Write-Host "✅ Submitted $($submitted.Count) tasks" -ForegroundColor Yellow
Write-Host "⏳ Polling active list..." -ForegroundColor Cyan
try {
  $list = Invoke-RestMethod -Uri "http://localhost:$Port/gpu/tasks"
  Write-Host ("Active: " + $list.active.Count + " | QueueDepth: " + $list.queue_depth + " | Completed: " + $list.completed)
} catch { Write-Host "Failed to get task list" -ForegroundColor Red }
