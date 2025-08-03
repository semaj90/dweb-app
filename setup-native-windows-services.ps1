#!/usr/bin/env powershell
# Native Windows Services Setup for Legal AI System
# Sets up PostgreSQL, Qdrant, Redis (Memurai), and Ollama without Docker Desktop

param(
    [switch]$Setup = $false,
    [switch]$Start = $false,
    [switch]$Stop = $false,
    [switch]$Status = $false,
    [switch]$GPU = $false,
    [switch]$Reset = $false,
    [string]$PostgresPassword = "legal_ai_secure_2024",
    [string]$DataDir = "C:\LegalAIData"
)

$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "🏛️ Legal AI - Native Windows Services"

# Color scheme
$primaryColor = "Cyan"
$successColor = "Green"
$warningColor = "Yellow"
$errorColor = "Red"
$infoColor = "White"

Write-Host "🏛️ LEGAL AI - NATIVE WINDOWS SERVICES SETUP" -ForegroundColor $primaryColor
Write-Host "=============================================" -ForegroundColor $primaryColor
Write-Host "🖥️ Windows 10/11 | Native Services | No Docker Required" -ForegroundColor $infoColor
Write-Host ""

# Ensure running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ This script requires Administrator privileges" -ForegroundColor $errorColor
    Write-Host "   Please run PowerShell as Administrator and try again" -ForegroundColor $warningColor
    exit 1
}

# Create data directory
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    Write-Host "✅ Created data directory: $DataDir" -ForegroundColor $successColor
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

function Test-ServicePort {
    param([int]$Port, [string]$ServiceName)
    try {
        $connection = Test-NetConnection -ComputerName "localhost" -Port $Port -InformationLevel Quiet
        if ($connection) {
            Write-Host "   ✅ $ServiceName is running on port $Port" -ForegroundColor $successColor
            return $true
        }
    } catch {}
    Write-Host "   ❌ $ServiceName is not running on port $Port" -ForegroundColor $errorColor
    return $false
}

function Wait-ForService {
    param([int]$Port, [string]$ServiceName, [int]$MaxAttempts = 30)
    Write-Host "   ⏳ Waiting for $ServiceName to start on port $Port..." -ForegroundColor $warningColor

    for ($i = 1; $i -le $MaxAttempts; $i++) {
        if (Test-ServicePort -Port $Port -ServiceName $ServiceName) {
            Write-Host "   ✅ $ServiceName is ready!" -ForegroundColor $successColor
            return $true
        }
        Start-Sleep -Seconds 2
        Write-Host "      Attempt $i/$MaxAttempts..." -ForegroundColor $infoColor
    }

    Write-Host "   ❌ $ServiceName failed to start within timeout" -ForegroundColor $errorColor
    return $false
}

function Install-ChocolateyPackage {
    param([string]$PackageName, [string]$DisplayName, [string]$AdditionalArgs = "")

    Write-Host "📦 Installing $DisplayName via Chocolatey..." -ForegroundColor $warningColor

    # Check if Chocolatey is installed
    try {
        choco --version | Out-Null
    } catch {
        Write-Host "   Installing Chocolatey first..." -ForegroundColor $infoColor
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    }

    try {
        if ($AdditionalArgs) {
            Invoke-Expression "choco install $PackageName $AdditionalArgs -y"
        } else {
            choco install $PackageName -y
        }
        Write-Host "   ✅ $DisplayName installed successfully" -ForegroundColor $successColor
        return $true
    } catch {
        Write-Host "   ❌ Failed to install $DisplayName" -ForegroundColor $errorColor
        Write-Host "      Error: $($_.Exception.Message)" -ForegroundColor $errorColor
        return $false
    }
}

# ============================================================================
# POSTGRESQL NATIVE SETUP
# ============================================================================

function Setup-PostgreSQL {
    Write-Host "🐘 Setting up PostgreSQL with pgvector..." -ForegroundColor $primaryColor

    # Check if PostgreSQL is already installed
    try {
        $psqlVersion = psql --version
        Write-Host "   ✅ PostgreSQL already installed: $psqlVersion" -ForegroundColor $successColor
    } catch {
        Write-Host "   📦 Installing PostgreSQL..." -ForegroundColor $warningColor

        # Install PostgreSQL via Chocolatey
        if (Install-ChocolateyPackage -PackageName "postgresql" -DisplayName "PostgreSQL" -AdditionalArgs "--params `"/Password:$PostgresPassword`"") {
            # Add PostgreSQL to PATH
            $env:PATH += ";C:\Program Files\PostgreSQL\16\bin"
            [Environment]::SetEnvironmentVariable("PATH", $env:PATH, [EnvironmentVariableTarget]::Machine)
        } else {
            Write-Host "   ⚠️ Auto-install failed. Manual installation required:" -ForegroundColor $warningColor
            Write-Host "      1. Download: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads" -ForegroundColor $infoColor
            Write-Host "      2. Run installer and set password to: $PostgresPassword" -ForegroundColor $infoColor
            return $false
        }
    }

    # Start PostgreSQL service
    Write-Host "   🚀 Starting PostgreSQL service..." -ForegroundColor $warningColor
    try {
        Start-Service -Name "postgresql*" -ErrorAction SilentlyContinue
        Set-Service -Name "postgresql*" -StartupType Automatic -ErrorAction SilentlyContinue
    } catch {
        Write-Host "   ⚠️ Could not start service automatically" -ForegroundColor $warningColor
    }

    # Wait for PostgreSQL to be ready
    if (Wait-ForService -Port 5432 -ServiceName "PostgreSQL") {
        # Install pgvector extension (if not already installed)
        Write-Host "   🔧 Installing pgvector extension..." -ForegroundColor $warningColor
        try {
            $env:PGPASSWORD = $PostgresPassword
            psql -U postgres -d postgres -c 'CREATE EXTENSION IF NOT EXISTS vector;'
            psql -U postgres -d postgres -c 'CREATE DATABASE IF NOT EXISTS legal_ai_db;'
            Write-Host "   ✅ pgvector extension installed" -ForegroundColor $successColor
        } catch {
            Write-Host "   ⚠️ pgvector installation may need manual setup" -ForegroundColor $warningColor
            Write-Host "      Run: CREATE EXTENSION vector; in psql" -ForegroundColor $infoColor
        }
        return $true
    }

    return $false
}

# ============================================================================
# QDRANT NATIVE SETUP
# ============================================================================

function Setup-Qdrant {
    Write-Host "🔍 Setting up Qdrant vector database..." -ForegroundColor $primaryColor

    $qdrantDir = "$DataDir\qdrant"
    $qdrantExe = "$qdrantDir\qdrant.exe"

    # Check if Qdrant is already installed
    if (Test-Path $qdrantExe) {
        Write-Host "   ✅ Qdrant already installed" -ForegroundColor $successColor
    } else {
        Write-Host "   📦 Downloading Qdrant..." -ForegroundColor $warningColor

        if (-not (Test-Path $qdrantDir)) {
            New-Item -ItemType Directory -Path $qdrantDir -Force | Out-Null
        }

        # Download Qdrant for Windows
        $qdrantUrl = "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip"
        $qdrantZip = "$qdrantDir\qdrant.zip"

        try {
            Invoke-WebRequest -Uri $qdrantUrl -OutFile $qdrantZip
            Expand-Archive -Path $qdrantZip -DestinationPath $qdrantDir -Force
            Remove-Item $qdrantZip

            # Find the extracted executable
            $extractedExe = Get-ChildItem -Path $qdrantDir -Name "qdrant.exe" -Recurse | Select-Object -First 1
            if ($extractedExe) {
                Move-Item -Path (Join-Path $qdrantDir $extractedExe) -Destination $qdrantExe -Force
            }

            Write-Host "   ✅ Qdrant downloaded successfully" -ForegroundColor $successColor
        } catch {
            Write-Host "   ❌ Failed to download Qdrant" -ForegroundColor $errorColor
            Write-Host "      Please download manually from: https://qdrant.tech/documentation/guides/installation/" -ForegroundColor $infoColor
            return $false
        }
    }

    # Create Qdrant configuration
    $qdrantConfig = @"
service:
  http_port: 6333
  grpc_port: 6334

storage:
  storage_path: "$($qdrantDir.Replace('\', '/'))/storage"

telemetry_disabled: true
"@

    $qdrantConfigFile = Join-Path $qdrantDir "config.yaml"
    $qdrantConfig | Out-File -FilePath $qdrantConfigFile -Encoding UTF8

    # Start Qdrant as background process
    Write-Host "   🚀 Starting Qdrant..." -ForegroundColor $warningColor
    try {
        $qdrantProcess = Start-Process -FilePath $qdrantExe -ArgumentList "--config-path", $qdrantConfigFile -WindowStyle Hidden -PassThru

        if (Wait-ForService -Port 6333 -ServiceName "Qdrant") {
            Write-Host "   ✅ Qdrant is running (PID: $($qdrantProcess.Id))" -ForegroundColor $successColor
            return $true
        }
    } catch {
        Write-Host "   ❌ Failed to start Qdrant" -ForegroundColor $errorColor
        return $false
    }

    return $false
}

# ============================================================================
# REDIS (MEMURAI) NATIVE SETUP
# ============================================================================

function Setup-Redis {
    Write-Host "🔴 Setting up Redis (Memurai for Windows)..." -ForegroundColor $primaryColor

    # Check if Redis/Memurai is already installed
    try {
        $redisInfo = redis-cli ping
        if ($redisInfo -eq "PONG") {
            Write-Host "   ✅ Redis/Memurai already running" -ForegroundColor $successColor
            return $true
        }
    } catch {}

    # Install Memurai (Redis for Windows)
    Write-Host "   📦 Installing Memurai (Redis for Windows)..." -ForegroundColor $warningColor

    if (Install-ChocolateyPackage -PackageName "memurai" -DisplayName "Memurai (Redis for Windows)") {
        # Start Memurai service
        try {
            Start-Service -Name "Memurai" -ErrorAction SilentlyContinue
            Set-Service -Name "Memurai" -StartupType Automatic -ErrorAction SilentlyContinue

            if (Wait-ForService -Port 6379 -ServiceName "Redis/Memurai") {
                return $true
            }
        } catch {
            Write-Host "   ⚠️ Service start failed, trying manual start..." -ForegroundColor $warningColor
        }
    }

    # Alternative: try installing Redis via Chocolatey
    Write-Host "   📦 Trying alternative Redis installation..." -ForegroundColor $warningColor
    if (Install-ChocolateyPackage -PackageName "redis-64" -DisplayName "Redis") {
        try {
            # Start Redis manually
            Start-Process "redis-server" -WindowStyle Hidden
            if (Wait-ForService -Port 6379 -ServiceName "Redis") {
                return $true
            }
        } catch {}
    }

    Write-Host "   ⚠️ Redis setup incomplete. Manual installation may be required:" -ForegroundColor $warningColor
    Write-Host "      1. Download Memurai: https://www.memurai.com/" -ForegroundColor $infoColor
    Write-Host "      2. Or use WSL: wsl --install then apt install redis-server" -ForegroundColor $infoColor
    return $false
}

# ============================================================================
# OLLAMA NATIVE SETUP
# ============================================================================

function Setup-Ollama {
    Write-Host "🤖 Setting up Ollama LLM engine..." -ForegroundColor $primaryColor

    # Check if Ollama is already installed
    try {
        $ollamaVersion = ollama --version
        Write-Host "   ✅ Ollama already installed: $ollamaVersion" -ForegroundColor $successColor
    } catch {
        Write-Host "   📦 Installing Ollama..." -ForegroundColor $warningColor

        # Download and install Ollama
        $ollamaInstaller = "$env:TEMP\OllamaSetup.exe"
        try {
            Invoke-WebRequest -Uri "https://ollama.com/download/windows" -OutFile $ollamaInstaller
            Start-Process -FilePath $ollamaInstaller -ArgumentList "/S" -Wait
            Remove-Item $ollamaInstaller -Force

            # Add Ollama to PATH
            $ollamaPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
            $env:PATH += ";$ollamaPath"
            [Environment]::SetEnvironmentVariable("PATH", $env:PATH, [EnvironmentVariableTarget]::User)

            Write-Host "   ✅ Ollama installed successfully" -ForegroundColor $successColor
        } catch {
            Write-Host "   ❌ Failed to install Ollama automatically" -ForegroundColor $errorColor
            Write-Host "      Please download manually from: https://ollama.com/download/windows" -ForegroundColor $infoColor
            return $false
        }
    }

    # Configure GPU support if requested
    if ($GPU) {
        Write-Host "   🎮 Configuring GPU support..." -ForegroundColor $warningColor

        # Check for NVIDIA GPU
        try {
            $nvidiaInfo = nvidia-smi --query-gpu=name --format=csv,noheader
            Write-Host "   ✅ NVIDIA GPU detected: $nvidiaInfo" -ForegroundColor $successColor

            # Set CUDA environment variables
            $env:OLLAMA_HOST = "0.0.0.0:11434"
            $env:OLLAMA_GPU_ENABLED = "1"
            [Environment]::SetEnvironmentVariable("OLLAMA_GPU_ENABLED", "1", [EnvironmentVariableTarget]::User)
        } catch {
            Write-Host "   ⚠️ NVIDIA GPU not detected, using CPU mode" -ForegroundColor $warningColor
        }
    }

    # Start Ollama service
    Write-Host "   🚀 Starting Ollama service..." -ForegroundColor $warningColor
    try {
        # Start Ollama in background
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden

        if (Wait-ForService -Port 11434 -ServiceName "Ollama") {
            # Pull essential models
            Write-Host "   📥 Pulling essential AI models..." -ForegroundColor $warningColor
            ollama pull llama3.2:3b  # Fast model for quick responses
            ollama pull nomic-embed-text  # Embedding model

            if ($GPU) {
                ollama pull llama3.1:8b  # Larger model for GPU
            }

            Write-Host "   ✅ Ollama is ready with models!" -ForegroundColor $successColor
            return $true
        }
    } catch {
        Write-Host "   ❌ Failed to start Ollama" -ForegroundColor $errorColor
        return $false
    }

    return $false
}

# ============================================================================
# SERVICE MANAGEMENT FUNCTIONS
# ============================================================================

function Start-AllServices {
    Write-Host "🚀 Starting all native Windows services..." -ForegroundColor $primaryColor

    $services = @(
        @{ Name = "PostgreSQL"; Function = { Setup-PostgreSQL } },
        @{ Name = "Qdrant"; Function = { Setup-Qdrant } },
        @{ Name = "Redis"; Function = { Setup-Redis } },
        @{ Name = "Ollama"; Function = { Setup-Ollama } }
    )

    $results = @()
    foreach ($service in $services) {
        Write-Host ""
        $result = & $service.Function
        $results += @{ Name = $service.Name; Success = $result }
    }

    Write-Host ""
    Write-Host "📊 SERVICE STATUS SUMMARY" -ForegroundColor $primaryColor
    Write-Host "========================" -ForegroundColor $primaryColor

    foreach ($result in $results) {
        if ($result.Success) {
            Write-Host "   ✅ $($result.Name)" -ForegroundColor $successColor
        } else {
            Write-Host "   ❌ $($result.Name)" -ForegroundColor $errorColor
        }
    }

    Write-Host ""
    Show-ServiceStatus
}

function Stop-AllServices {
    Write-Host "🛑 Stopping all native Windows services..." -ForegroundColor $warningColor

    # Stop services
    try {
        Stop-Service -Name "postgresql*" -Force -ErrorAction SilentlyContinue
        Stop-Service -Name "Memurai" -Force -ErrorAction SilentlyContinue

        # Stop Ollama processes
        Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force

        # Stop Qdrant processes
        Get-Process -Name "qdrant" -ErrorAction SilentlyContinue | Stop-Process -Force

        Write-Host "✅ All services stopped" -ForegroundColor $successColor
    } catch {
        Write-Host "⚠️ Some services may still be running" -ForegroundColor $warningColor
    }
}

function Show-ServiceStatus {
    Write-Host "📊 CURRENT SERVICE STATUS" -ForegroundColor $primaryColor
    Write-Host "=========================" -ForegroundColor $primaryColor

    $services = @(
        @{ Name = "PostgreSQL"; Port = 5432 },
        @{ Name = "Qdrant"; Port = 6333 },
        @{ Name = "Redis/Memurai"; Port = 6379 },
        @{ Name = "Ollama"; Port = 11434 }
    )

    foreach ($service in $services) {
        Test-ServicePort -Port $service.Port -ServiceName $service.Name
    }

    Write-Host ""
    Write-Host "🌐 API Endpoints:" -ForegroundColor $infoColor
    Write-Host "   PostgreSQL: localhost:5432" -ForegroundColor $infoColor
    Write-Host "   Qdrant: http://localhost:6333" -ForegroundColor $infoColor
    Write-Host "   Redis: localhost:6379" -ForegroundColor $infoColor
    Write-Host "   Ollama: http://localhost:11434" -ForegroundColor $infoColor
}

function Reset-AllServices {
    Write-Host "🔄 Resetting all services and data..." -ForegroundColor $warningColor

    Stop-AllServices

    # Reset data directories
    if (Test-Path "$DataDir\qdrant") {
        Remove-Item "$DataDir\qdrant" -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Reset PostgreSQL data (requires manual intervention)
    Write-Host "⚠️ PostgreSQL data reset requires manual action:" -ForegroundColor $warningColor
    Write-Host "   1. Stop PostgreSQL service" -ForegroundColor $infoColor
    Write-Host "   2. Delete C:\Program Files\PostgreSQL\*\data" -ForegroundColor $infoColor
    Write-Host "   3. Reinitialize with: initdb -D /path/to/data" -ForegroundColor $infoColor

    Start-AllServices
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

switch ($true) {
    $Setup { Start-AllServices }
    $Start { Start-AllServices }
    $Stop { Stop-AllServices }
    $Status { Show-ServiceStatus }
    $Reset { Reset-AllServices }
    default {
        Write-Host "🏛️ Legal AI Native Windows Services Manager" -ForegroundColor $primaryColor
        Write-Host ""
        Write-Host "Usage:" -ForegroundColor $infoColor
        Write-Host "  .\setup-native-windows-services.ps1 -Setup     # First-time setup" -ForegroundColor $infoColor
        Write-Host "  .\setup-native-windows-services.ps1 -Start     # Start all services" -ForegroundColor $infoColor
        Write-Host "  .\setup-native-windows-services.ps1 -Stop      # Stop all services" -ForegroundColor $infoColor
        Write-Host "  .\setup-native-windows-services.ps1 -Status    # Check service status" -ForegroundColor $infoColor
        Write-Host "  .\setup-native-windows-services.ps1 -Reset     # Reset all data" -ForegroundColor $infoColor
        Write-Host ""
        Write-Host "Options:" -ForegroundColor $infoColor
        Write-Host "  -GPU                    # Enable GPU acceleration" -ForegroundColor $infoColor
        Write-Host "  -PostgresPassword       # Set PostgreSQL password (default: legal_ai_secure_2024)" -ForegroundColor $infoColor
        Write-Host "  -DataDir                # Set data directory (default: C:\LegalAIData)" -ForegroundColor $infoColor
        Write-Host ""
        Show-ServiceStatus
    }
}

Write-Host ""
Write-Host "🏛️ Legal AI Native Windows Services Setup Complete!" -ForegroundColor $primaryColor
