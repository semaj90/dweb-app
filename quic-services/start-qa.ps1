#!/usr/bin/env pwsh
<#
Start QUIC binaries (background) and poll their /health endpoints until healthy.
Usage: powershell -NoProfile -ExecutionPolicy Bypass -File start-qa.ps1
#>
Set-StrictMode -Version Latest -ErrorAction Stop
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

$binDir = Join-Path $scriptDir "..\go-microservice\bin"
if (-not (Test-Path $binDir)) { Write-Host "Bin dir not found: $binDir"; exit 1 }

$services = @(
    @{ Name = 'quic-gateway'; Exe = 'quic-gateway.exe'; Port = 8444 },
    @{ Name = 'quic-vector-proxy'; Exe = 'quic-vector-proxy.exe'; Port = 8446 },
    @{ Name = 'quic-ai-stream'; Exe = 'quic-ai-stream.exe'; Port = 8448 },
    @{ Name = 'rag-quic-proxy'; Exe = 'rag-quic-proxy.exe'; Port = 8089 }
)

$logsDir = Join-Path $scriptDir "..\logs"
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

function Start-ServiceIfMissing($svc) {
    $exePath = Join-Path $binDir $svc.Exe
    if (-not (Test-Path $exePath)) { Write-Host "Skipping $($svc.Name): exe not found at $exePath" -ForegroundColor Yellow; return }
    $procName = [System.IO.Path]::GetFileNameWithoutExtension($svc.Exe)
    if (Get-Process -Name $procName -ErrorAction SilentlyContinue) { Write-Host "$($svc.Name) already running" -ForegroundColor Cyan; return }

    $outLog = Join-Path $logsDir "$($svc.Name).log"
    $errLog = Join-Path $logsDir "$($svc.Name).err.log"
    Write-Host "Starting $($svc.Name) -> $exePath" -ForegroundColor Green
    Start-Process -FilePath $exePath -ArgumentList "--port", "$($svc.Port)" -NoNewWindow -RedirectStandardOutput $outLog -RedirectStandardError $errLog
}

foreach ($s in $services) { Start-ServiceIfMissing $s }

function Wait-ForHealth($svc, $timeoutSec) {
    $url = "http://localhost:$($svc.Port)/health"
    $end = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $end) {
        try {
            $resp = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 3 -ErrorAction Stop
            if ($resp -ne $null) { Write-Host "✅ $($svc.Name) healthy at $url" -ForegroundColor Green; return $true }
        } catch {
            # ignore and retry
        }
        Start-Sleep -Seconds 1
    }
    Write-Host "❌ $($svc.Name) did not become healthy within $timeoutSec seconds (tried $url)" -ForegroundColor Red
    return $false
}

# wait for services
foreach ($s in $services) {
    Wait-ForHealth $s 12 | Out-Null
}

Write-Host "Start-qa finished." -ForegroundColor Green
Pop-Location
