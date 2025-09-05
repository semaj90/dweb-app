$toolsBin = Join-Path $PSScriptRoot 'bin'
if (-not (Test-Path $toolsBin)) { New-Item -ItemType Directory -Path $toolsBin | Out-Null }
$env:GOBIN = $toolsBin
Write-Host "GOBIN => $env:GOBIN"
Write-Host 'Installing protoc-gen-go@v1.30.0'
go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.30.0
Write-Host 'Installing protoc-gen-go-grpc@v1.3.0'
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.3.0
Write-Host 'Plugins installed into:'; Get-ChildItem $env:GOBIN | Select-Object Name | ForEach-Object { Write-Host $_.Name }
