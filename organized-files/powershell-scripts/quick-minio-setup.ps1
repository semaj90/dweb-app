# Quick MinIO Setup for Legal AI System
param(
    [string]$DataDir = "C:\minio-data",
    [int]$Port = 9000,
    [int]$ConsolePort = 9001
)

Write-Host "🗄️ Setting up MinIO for Legal AI System..." -ForegroundColor Green

# Create data directory
if (!(Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    Write-Host "✅ Created MinIO data directory: $DataDir" -ForegroundColor Green
}

# Ensure TLS 1.2 (harmless if already set)
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

# Resolve paths
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$MinioExe   = Join-Path $scriptRoot "minio.exe"

# Use local minio.exe only
if (!(Test-Path -LiteralPath $MinioExe)) {
    Write-Host "❌ Local MinIO executable not found at: $MinioExe" -ForegroundColor Red
    Write-Host "Place minio.exe in this directory and rerun." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "✅ Using local MinIO: $MinioExe" -ForegroundColor Green
    try { Unblock-File -LiteralPath $MinioExe -ErrorAction SilentlyContinue } catch {}

    # Ensure firewall ports are open (if available)
    try {
        if (-not (Get-NetFirewallRule -DisplayName "MinIO API ($Port)" -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName "MinIO API ($Port)" -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow | Out-Null
        }
        if (-not (Get-NetFirewallRule -DisplayName "MinIO Console ($ConsolePort)" -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName "MinIO Console ($ConsolePort)" -Direction Inbound -Protocol TCP -LocalPort $ConsolePort -Action Allow | Out-Null
        }
    } catch { }
}

# Set environment variables
$env:MINIO_ROOT_USER = "minioadmin"
$env:MINIO_ROOT_PASSWORD = "minioadmin123"

# Start MinIO server
Write-Host "🚀 Starting MinIO server on ports $Port (API) and $ConsolePort (Console)..." -ForegroundColor Yellow

try {
    # Check if MinIO is already running
    $MinioProcess = Get-Process -Name "minio" -ErrorAction SilentlyContinue
    if ($MinioProcess) {
        Write-Host "⚠️ MinIO is already running (PID: $($MinioProcess.Id))" -ForegroundColor Yellow
    } else {
        # Start MinIO in background
        Start-Process -FilePath ".\minio.exe" -ArgumentList "server", "--address", ":$Port", "--console-address", ":$ConsolePort", $DataDir -NoNewWindow

        # Wait a moment for startup
        Start-Sleep 3

        # Verify it started
        $TestResponse = try {
            Invoke-WebRequest -Uri "http://localhost:$Port/minio/health/ready" -UseBasicParsing -TimeoutSec 5
        } catch { $null }

        if ($TestResponse -and $TestResponse.StatusCode -eq 200) {
            Write-Host "✅ MinIO server started successfully!" -ForegroundColor Green
            Write-Host "🌐 API Endpoint: http://localhost:$Port" -ForegroundColor Cyan
            Write-Host "🎮 Console: http://localhost:$ConsolePort" -ForegroundColor Cyan
            Write-Host "🔑 Username: minioadmin" -ForegroundColor Cyan
            Write-Host "🔑 Password: minioadmin123" -ForegroundColor Cyan
        } else {
            Write-Host "⚠️ MinIO may still be starting up..." -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "❌ Failed to start MinIO: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "🎯 MinIO setup complete! You can now restart your upload service." -ForegroundColor Green
