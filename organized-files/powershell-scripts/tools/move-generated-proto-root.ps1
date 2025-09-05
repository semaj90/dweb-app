$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $root '..\go-microservice\proto')
$files = Get-ChildItem -File -Filter '*.pb.go'
if ($files.Count -eq 0) { Write-Host 'No pb.go files found in proto root' ; exit 0 }
$backupDir = Join-Path '..\tools_backup_generated_proto_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
$pairs = @{ 'ai_dimensional.pb.go'='aidimensional'; 'aiserver.pb.go'='aiserver'; 'aiserver_grpc.pb.go'='aiserver'; 'recommendations.pb.go'='recommendations'; 'recommendations_grpc.pb.go'='recommendations'; 'vector-service.pb.go'='vectorservice'; 'vector-service_grpc.pb.go'='vectorservice' }
foreach ($f in $files) {
  $name = $f.Name
  Write-Host "Processing top-level file: $name"
  if ($pairs.ContainsKey($name)) {
    $pkg = $pairs[$name]
    $destDir = Join-Path '.' $pkg
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
    Write-Host "Backing up $name to $backupDir"
    Move-Item -Path $name -Destination (Join-Path $backupDir $name) -Force
    Write-Host "Moving $name -> $destDir"
    Move-Item -Path (Join-Path $backupDir $name) -Destination (Join-Path $destDir $name) -Force
  } else {
    Write-Host "No mapping for $name, leaving in place"
  }
}
Write-Host 'Move complete. Current proto tree:'
Get-ChildItem -Recurse | Select-Object FullName | Format-Table -AutoSize
