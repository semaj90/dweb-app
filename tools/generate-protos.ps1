$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$protoc = Join-Path $root 'protoc-23.4\bin\protoc.exe'
$gobin = Join-Path $root 'bin'
$env:Path = $gobin + ';' + (Split-Path $protoc) + ';' + $env:Path
Set-Location (Join-Path $root '..\go-microservice')
$pairs = @{
  'aiserver.proto' = 'aiserver'
  'ai_dimensional.proto' = 'aidimensional'
  'recommendations.proto' = 'recommendations'
  'vector-service.proto' = 'vectorservice'
}
foreach ($k in $pairs.Keys) {
  $pkg = $pairs[$k]
  $out = Join-Path 'proto' $pkg
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  Write-Host "Generating $k -> proto/$pkg"
  & $protoc --proto_path=proto --go_out=$out --go-grpc_out=$out (Join-Path 'proto' $k)
  if ($LASTEXITCODE -ne 0) { throw "protoc failed for $k with exit $LASTEXITCODE" }
}
Write-Host 'Running go mod tidy'
go mod tidy
Write-Host 'Running go build ./...'
go build ./...
Write-Host 'Done'
