$protoRoot = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..\go-microservice\proto' | Resolve-Path
Set-Location $protoRoot
Write-Host 'Working in' (Get-Location)
$files = Get-ChildItem -File -Filter '*.pb.go'
if ($files.Count -eq 0) { Write-Host 'No top-level pb.go files found' ; exit 0 }
$mapping = @{
  'aiserver.pb.go' = 'aiserver'
  'aiserver_grpc.pb.go' = 'aiserver'
  'ai_dimensional.pb.go' = 'aidimensional'
  'recommendations.pb.go' = 'recommendations'
  'recommendations_grpc.pb.go' = 'recommendations'
  'vector-service.pb.go' = 'vectorservice'
  'vector-service_grpc.pb.go' = 'vectorservice'
}
foreach ($f in $files) {
  $name = $f.Name
  Write-Host "Found: $name"
  if ($mapping.ContainsKey($name)) {
    $dest = Join-Path $protoRoot $mapping[$name]
    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Path $dest | Out-Null }
    Write-Host "Moving $name -> $dest"
    Move-Item -Path $f.FullName -Destination (Join-Path $dest $name) -Force
  } else {
    Write-Host "No mapping for $name; leaving in place"
  }
}
Write-Host 'Move complete. Running go mod tidy and go build'
Set-Location (Join-Path $protoRoot '..')
go mod tidy
go build ./...
Write-Host 'Done'
