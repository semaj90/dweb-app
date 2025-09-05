param(
  [int]$Iterations = 50,
  [int]$Concurrency = 4,
  [int]$Port = $(if($env:GPU_ORCHESTRATOR_PORT){$env:GPU_ORCHESTRATOR_PORT}else{8095}),
  [int]$SleepMs = 120
)
Write-Host "⚙️ Load generator: iterations=$Iterations concurrency=$Concurrency port=$Port" -ForegroundColor Cyan
$scriptBlock = {
  param($i,$port,$sleep)
  $prio = Get-Random -Minimum 1 -Maximum 11
  $type = (Get-Random -InputObject @('embedding','inference','tensor_op','cuda_kernel'))
  $payload = @{ type=$type; priority=$prio; input="item_$i"; metadata=@{ batch='lg'; delay_ms = (Get-Random -Minimum 0 -Maximum 150) } } | ConvertTo-Json -Depth 5
  try { Invoke-RestMethod -Uri "http://localhost:$port/gpu/task" -Method Post -Body $payload -ContentType 'application/json' | Out-Null } catch {}
  Start-Sleep -Milliseconds $sleep
}
for($i=1; $i -le $Iterations; $i+=$Concurrency){
  $jobs = @()
  for($j=0;$j -lt $Concurrency -and ($i+$j) -le $Iterations;$j++){
    $idx = $i + $j
    $jobs += Start-Job -ScriptBlock $scriptBlock -ArgumentList $idx,$Port,$SleepMs
  }
  Receive-Job -Job $jobs -Wait -AutoRemoveJob | Out-Null
}
Write-Host "✅ Load generation complete" -ForegroundColor Green
