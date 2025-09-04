# Regenerate all .pb.go files using repo-local protoc and plugins
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\protoc-regenerate.ps1

# $scriptDir is scripts directory; repo root is its parent
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repo = Split-Path -Parent $scriptDir
# Candidate protoc locations (repo-relative)
$candidates = @(
    (Join-Path $repo 'sveltekit-frontend\tools\protoc-23.4\bin\protoc.exe'),
    (Join-Path $repo 'sveltekit-frontend\tools\protoc\protoc.exe'),
    (Join-Path $repo 'tools\protoc.exe')
)

$protoc = $null
foreach ($c in $candidates) {
    if (Test-Path $c) { $protoc = $c; break }
}

$gobin = Join-Path $repo 'tools\bin'

if (-not $protoc) {
    Write-Error "protoc not found. Checked: $($candidates -join ';') . Update path and try again."
    exit 1
}

# Ensure our repo-local plugin bin dir is on PATH so protoc-gen-go/grpc are found
$env:PATH = $gobin + ';' + $env:PATH

# Make logs dir
$logs = Join-Path $repo 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

Write-Host "Using protoc: $protoc"
Write-Host "PATH includes: $gobin"

# Collect proto files
$protoFiles = Get-ChildItem -Path (Join-Path $repo 'go-microservice\proto') -Filter *.proto -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
$topProtoFiles = Get-ChildItem -Path (Join-Path $repo 'proto') -Filter *.proto -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
$protoFiles = $protoFiles + $topProtoFiles

if ($protoFiles.Count -eq 0) {
    Write-Host "No .proto files found."
    exit 0
}

foreach ($p in $protoFiles) {
    Write-Host "Generating: $p"
    $args = @(
        "--proto_path=$repo",
        "--proto_path=$(Join-Path $repo 'go-microservice\proto')",
        "--go_out=paths=source_relative:$(Join-Path $repo 'go-microservice\proto')",
        "--go-grpc_out=paths=source_relative:$(Join-Path $repo 'go-microservice\proto')",
        "$p"
    )
    # Call protoc directly to avoid Start-Process quoting issues
    & $protoc @args 2>&1 | Tee-Object -FilePath (Join-Path $logs 'protoc-out.txt') -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "protoc failed for $p (exit $LASTEXITCODE)"
    }
}

Write-Host "Done. See logs\protoc-out.txt for details."
