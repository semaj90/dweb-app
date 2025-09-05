param(
  [string]$ProcessName = 'gpu-orchestrator'
)
Write-Host "🛑 Attempting graceful stop of $ProcessName.exe" -ForegroundColor Yellow
$procs = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
if(!$procs){ Write-Host "No running process found." -ForegroundColor DarkGray; exit 0 }
foreach($p in $procs){
  try { Stop-Process -Id $p.Id -Force; Write-Host "Stopped PID $($p.Id)" -ForegroundColor Green } catch { Write-Host "Failed stopping PID $($p.Id)" -ForegroundColor Red }
}
Write-Host "✅ Done" -ForegroundColor Cyan
