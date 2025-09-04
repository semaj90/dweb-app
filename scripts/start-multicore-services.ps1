<#!
.SYNOPSIS
  Build & start Enhanced Multicore Go Service and Context7 MCP Multi-Core Server in parallel.
.DESCRIPTION
  - Builds go-microservice/enhanced-multicore-service.go
  - Detects free ports (defaults: Go 8098, MCP 4100)
  - Starts processes with structured logging & health polling
  - Provides Stop-All function
.PARAMETER GoPort
  Port for Go enhanced multicore service (default 8098)
.PARAMETER MCPPort
  Port for MCP multi-core server (default 4100)
.PARAMETER Workers
  Max worker processes for MCP (default min(CPU,8))
.PARAMETER EnableGPU
  Enable GPU flag for Go service (default $true)
.PARAMETER Watch
  Rebuild Go service on source changes (basic poll) (default $false)
.EXAMPLE
  ./scripts/start-multicore-services.ps1 -GoPort 8099 -MCPPort 4200 -Workers 6
#>
param(
  [int]$GoPort = 8098,
  [int]$MCPPort = 4100,
  [int]$Workers = 0,
  [switch]$EnableGPU = $true,
  [switch]$Watch = $false
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Log {
  param([string]$Msg,[string]$Tag='SYS',[string]$Color='Gray')
  $ts = (Get-Date).ToString('HH:mm:ss')
  Write-Host "[$ts][$Tag] $Msg" -ForegroundColor $Color
}

function Test-PortFree { param([int]$Port) (Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded -eq $false }

function Find-FreePort { param([int]$Start,[int]$MaxAttempts=20)
  for($i=0;$i -lt $MaxAttempts;$i++){ $p=$Start+$i; if(Test-PortFree $p){ return $p } }
  throw "No free port near $Start"
}

if(-not (Test-PortFree $GoPort)){ $GoPort = Find-FreePort -Start $GoPort }
if(-not (Test-PortFree $MCPPort)){ $MCPPort = Find-FreePort -Start $MCPPort }

if($Workers -le 0){ $Workers = [Math]::Min([Environment]::ProcessorCount,8) }

$RepoRoot = Split-Path -Parent $PSScriptRoot
$GoSrc    = Join-Path $RepoRoot 'go-microservice/enhanced-multicore-service.go'
$GoBinDir = Join-Path $RepoRoot 'go-microservice/bin'
$GoExe    = Join-Path $GoBinDir 'multicore-service.exe'

if(-not (Test-Path $GoSrc)){ throw "Go source not found: $GoSrc" }
if(-not (Test-Path $GoBinDir)){ New-Item -ItemType Directory -Path $GoBinDir | Out-Null }

Write-Log "Building Go service..." 'GO' 'Cyan'
Push-Location (Split-Path $GoSrc -Parent)
try {
  go build -o $GoExe $(Split-Path $GoSrc -Leaf) 2>&1 | ForEach-Object { Write-Log $_ 'GO' 'DarkCyan' }
} finally { Pop-Location }
if(-not (Test-Path $GoExe)){ throw 'Build failed' }
Write-Log "Go build complete: $GoExe" 'GO' 'Green'

$LogsDir = Join-Path $RepoRoot 'logs'
if(-not (Test-Path $LogsDir)){ New-Item -ItemType Directory -Path $LogsDir | Out-Null }
$GoLog  = Join-Path $LogsDir 'multicore-go.log'
$MCPLog = Join-Path $LogsDir 'mcp-multicore.log'

$env:MULTICORE_PORT = "$GoPort"
$env:ENABLE_GPU = ($EnableGPU.IsPresent).ToString().ToLower()
$env:MCP_PORT = "$MCPPort"
$env:MCP_MULTICORE = 'true'
$env:MCP_DEBUG = 'false'

$GoProc = Start-Process -FilePath $GoExe -NoNewWindow -RedirectStandardOutput $GoLog -RedirectStandardError $GoLog -PassThru
Write-Log "Started Go Multicore PID=$($GoProc.Id) port=$GoPort" 'GO' 'Green'

$McpPath = Join-Path $RepoRoot 'context7-mcp-server-multicore.js'
if(-not (Test-Path $McpPath)){ Write-Log 'MCP server file missing' 'MCP' 'Red'; Stop-Process -Id $GoProc.Id; throw 'Missing MCP server JS' }

$Node = (Get-Command node -ErrorAction SilentlyContinue).Path
if(-not $Node){ throw 'node not found in PATH' }

$MCPProc = Start-Process -FilePath $Node -ArgumentList @($McpPath) -NoNewWindow -RedirectStandardOutput $MCPLog -RedirectStandardError $MCPLog -PassThru
Write-Log "Started MCP Multi-Core PID=$($MCPProc.Id) port=$MCPPort workers=$Workers" 'MCP' 'Green'

# Health polling
function Wait-Health {
  param([string]$Name,[string]$Url,[int]$Retries=25,[int]$DelayMs=400)
  for($i=1;$i -le $Retries;$i++){
    try {
      $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
      if($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300){ Write-Log "$Name healthy ($($resp.StatusCode))" $Name 'Green'; return $true }
    } catch { }
    Start-Sleep -Milliseconds $DelayMs
  }
  Write-Log "$Name NOT healthy after retries" $Name 'Red'
  return $false
}

$goOk  = Wait-Health -Name 'GO'  -Url "http://localhost:$GoPort/api/health" -Retries 30
$mcpOk = Wait-Health -Name 'MCP' -Url "http://localhost:$MCPPort/health" -Retries 30

function Show-LogTail { param([string]$Path,[int]$Lines=20) if(Test-Path $Path){ Write-Host "----- $Path (last $Lines) -----" -ForegroundColor DarkGray; Get-Content $Path -Tail $Lines } }

if(-not ($goOk -and $mcpOk)){
  Write-Log 'One or more services failed health check. Tail logs:' 'SYS' 'Yellow'
  Show-LogTail $GoLog
  Show-LogTail $MCPLog
  Write-Log 'Stopping processes...' 'SYS' 'Yellow'
  if($GoProc -and !$GoProc.HasExited){ Stop-Process -Id $GoProc.Id -Force }
  if($MCPProc -and !$MCPProc.HasExited){ Stop-Process -Id $MCPProc.Id -Force }
  exit 1
}

Write-Log 'Both services healthy. Streaming live tails (Ctrl+C to exit)...' 'SYS' 'Cyan'

$stop = $false
Register-EngineEvent PowerShell.Exiting -Action { $script:stop=$true } | Out-Null

Start-Job -ScriptBlock { param($GoLog) Get-Content -Path $GoLog -Wait } -ArgumentList $GoLog | Out-Null
Start-Job -ScriptBlock { param($MCPLog) Get-Content -Path $MCPLog -Wait } -ArgumentList $MCPLog | Out-Null

function Stop-All {
  Write-Log 'Stopping all services...' 'SYS' 'Yellow'
  Get-Job | Stop-Job -Force -ErrorAction SilentlyContinue
  if($GoProc -and !$GoProc.HasExited){ Stop-Process -Id $GoProc.Id -Force }
  if($MCPProc -and !$MCPProc.HasExited){ Stop-Process -Id $MCPProc.Id -Force }
  Write-Log 'Stopped.' 'SYS' 'Gray'
}

if($Watch){
  Write-Log 'Watch mode enabled (poll every 3s)...' 'WATCH' 'DarkYellow'
  $lastHash = ''
  while(-not $stop){
    $hash = Get-ChildItem (Split-Path $GoSrc -Parent) -Recurse -Include *.go | Get-FileHash -Algorithm SHA256 | ForEach-Object Hash | Sort-Object | Out-String
    if($hash -ne $lastHash){
      Write-Log 'Change detected → rebuilding...' 'WATCH' 'Yellow'
      try {
        Push-Location (Split-Path $GoSrc -Parent)
        go build -o $GoExe $(Split-Path $GoSrc -Leaf) 2>&1 | ForEach-Object { Write-Log $_ 'GO' 'DarkCyan' }
        Pop-Location
        if($GoProc -and !$GoProc.HasExited){ Stop-Process -Id $GoProc.Id -Force }
        $GoProc = Start-Process -FilePath $GoExe -NoNewWindow -RedirectStandardOutput $GoLog -RedirectStandardError $GoLog -PassThru
        Write-Log "Restarted Go service PID=$($GoProc.Id)" 'WATCH' 'Green'
      } catch { Write-Log "Rebuild failed: $($_.Exception.Message)" 'WATCH' 'Red' }
      $lastHash = $hash
    }
    Start-Sleep -Seconds 3
  }
} else {
  while(-not $stop){ Start-Sleep -Seconds 60 }
}

Stop-All
