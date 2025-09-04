# Move generated .pb.go files under go-microservice/proto into subfolders based on package name inside the file
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\reorg-pbgo.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# repo root is parent of scripts directory
$repo = Split-Path -Parent $scriptDir
Set-Location $repo
$protoDir = Join-Path $repo "go-microservice\proto"

Write-Host "Scanning for *.pb.go files in $protoDir"
$pbfiles = Get-ChildItem -Path $protoDir -Filter "*.pb.go" -File -Recurse

foreach ($f in $pbfiles) {
    # skip files already in subdirs (we only want those in proto root or direct)
    if ($f.DirectoryName -ne $protoDir) { continue }
    Write-Host "Processing" $f.Name
    $text = Get-Content $f.FullName -Raw
    # try to extract package name from file
    # Use inline (?m) to allow ^ to match line starts
    if ($text -match '(?m)^package\s+([a-zA-Z0-9_]+)') {
        $pkg = $matches[1]
    } else {
        Write-Host "  Could not find package declaration in $($f.Name); skipping" -ForegroundColor Yellow
        continue
    }

    $targetDir = Join-Path $protoDir $pkg
    if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir | Out-Null }

    $targetPath = Join-Path $targetDir $f.Name
    if (Test-Path $targetPath) {
        $bak = $targetPath + ".bak-$(Get-Date -Format yyyyMMddHHmmss)"
        Write-Host "  Target exists. Backing up existing to $bak"
        Move-Item -Path $targetPath -Destination $bak -Force
    }

    Write-Host "  Moving $($f.Name) -> $targetDir"
    Move-Item -Path $f.FullName -Destination $targetPath -Force
}

Write-Host "Done reorganizing .pb.go files. You should run 'go mod tidy' and 'go build ./...' in go-microservice to verify."
