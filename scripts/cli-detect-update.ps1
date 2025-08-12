param(
  [switch]$AutoUpdate = $false,
  [string]$OutFile = ".\logs\cli-detect.json"
)

function Ensure-Dir($path) {
  $dir = Split-Path -Parent $path
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
}

function Get-CliInfo($name, $pkg) {
  $exists = $false
  $version = $null
  try {
    $cmd = Get-Command $name -ErrorAction Stop
    $exists = $true
    $verOut = & $name --version 2>$null
    if ($LASTEXITCODE -eq 0 -and $verOut) { $version = $verOut.Trim() }
  } catch { $exists = $false }

  [PSCustomObject]@{ name = $name; package = $pkg; exists = $exists; version = $version }
}

$mapping = @(
  @{ name = 'claude'; pkg = 'claude' },
  @{ name = 'claude-cli'; pkg = 'claude-cli' },
  @{ name = 'gemini'; pkg = 'gemini-cli' }
)

$results = @()
foreach ($m in $mapping) {
  $results += Get-CliInfo -name $m.name -pkg $m.pkg
}

if ($AutoUpdate) {
  foreach ($r in $results | Where-Object { $_.exists }) {
    if ($r.package) {
      Write-Host "Updating $($r.name) via npm -g ($($r.package))..."
      npm install -g $r.package | Out-Null
    }
  }
}

Ensure-Dir $OutFile
$payload = @{ timestamp = (Get-Date).ToString('o'); clis = $results }
$payload | ConvertTo-Json -Depth 4 | Set-Content -Path $OutFile -Encoding UTF8
Write-Host "Saved CLI status to $OutFile"
