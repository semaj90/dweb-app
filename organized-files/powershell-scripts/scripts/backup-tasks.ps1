Param(
  [string]$TasksPath = ".vscode/tasks.json",
  [string]$KratosPath = "go-services/cmd/kratos-server/main.go",
  [string]$DestDir = ".vscode/backups"
)

$ErrorActionPreference = "Stop"
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not (Test-Path $DestDir)) { New-Item -ItemType Directory -Path $DestDir | Out-Null }

$tasksDest = Join-Path $DestDir ("tasks.json.$ts.bak")
Copy-Item -Path $TasksPath -Destination $tasksDest -Force

if (Test-Path $KratosPath) {
  $kratosDest = Join-Path $DestDir ("kratos-main.$ts.go.bak")
  Copy-Item -Path $KratosPath -Destination $kratosDest -Force
}

Write-Output "Backups written to $DestDir (timestamp=$ts)"
