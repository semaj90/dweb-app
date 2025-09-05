# Repo-local proto generation script
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\gen-protos.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

$protoDir = Join-Path $root "go-microservice\proto"
$toolsProtocCandidates = @("$root\tools\protoc-23.4\bin\protoc.exe", "$root\sveltekit-frontend\tools\protoc-23.4\bin\protoc.exe", "$root\tools\protoc\bin\protoc.exe")
$protoc = $null
foreach ($p in $toolsProtocCandidates) {
    if (Test-Path $p) { $protoc = $p; break }
}
if (-not $protoc) {
    Write-Host "protoc not found in repo-local tools. Please install protoc or place it under tools/protoc-23.4/bin/protoc.exe" -ForegroundColor Yellow
    exit 1
}
Write-Host "Using protoc: $protoc"

# Ensure tools/bin exists for Go plugin installation
$toolsBin = Join-Path $root "tools\bin"
if (-not (Test-Path $toolsBin)) { New-Item -ItemType Directory -Path $toolsBin | Out-Null }

# Install protoc-gen-go and protoc-gen-go-grpc into tools/bin (GOBIN)
$env:GOBIN = (Resolve-Path $toolsBin).Path
Write-Host "Installing protoc-gen-go into $env:GOBIN"
& go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.30.0
& go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.3.0

# Ensure plugins are discoverable on PATH for protoc
$env:PATH = "$env:GOBIN;$(Split-Path -Parent $protoc);$env:PATH"

# Helper to extract go_package name
function Get-GoPackageName($protoFile) {
    $text = Get-Content $protoFile -Raw
    # Use single-quoted regex to avoid escaping issues
    if ($text -match 'option\s+go_package\s*=\s*"([^\"]+)"\s*;') {
        $val = $matches[1]
        # go_package format: import/path;package
        if ($val -match ';([a-zA-Z0-9_]+)$') { return $matches[1] }
        # fallback to last path segment
        return (Split-Path $val -Leaf)
    }
    return $null
}

$protos = Get-ChildItem -Path $protoDir -Filter *.proto -Recurse
if ($protos.Count -eq 0) { Write-Host "No proto files found under $protoDir"; exit 0 }

foreach ($f in $protos) {
    Write-Host "Processing proto: $($f.FullName)"
    $pkgName = Get-GoPackageName $f.FullName
    if (-not $pkgName) { Write-Host "  WARNING: no go_package option found; generating into proto root" -ForegroundColor Yellow; $outDir = Join-Path $protoDir "gen" }
    else { $outDir = Join-Path $protoDir $pkgName }

    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

    & $protoc --proto_path=$protoDir --go_out=paths=source_relative:$outDir --go-grpc_out=paths=source_relative:$outDir $f.FullName
    if ($LASTEXITCODE -ne 0) { Write-Host "protoc failed for $($f.Name) with exit code $LASTEXITCODE" -ForegroundColor Red; exit $LASTEXITCODE }
    Write-Host "  Generated into $outDir"
}

# Tidy and try to build go-microservice
Push-Location (Join-Path $root "go-microservice")
Write-Host "Running go mod tidy..."
& go mod tidy
if ($LASTEXITCODE -ne 0) { Write-Host "go mod tidy failed" -ForegroundColor Red }

Write-Host "Building go-microservice ./..."
& go build ./...
$buildExit = $LASTEXITCODE
if ($buildExit -ne 0) { Write-Host "go build failed with exit code $buildExit" -ForegroundColor Red }
else { Write-Host "go build succeeded" -ForegroundColor Green }
Pop-Location

Write-Host "Done. Generated protos and attempted build."
