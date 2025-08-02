#!/usr/bin/env pwsh
# Complete Ollama GPU Setup and Verification Script

param(
    [switch]$SkipModelCreation,
    [switch]$TestOnly
)

$ErrorActionPreference = "Continue"
$baseDir = "C:\Users\james\Desktop\deeds-web\deeds-web-app\local-models"
Set-Location $baseDir

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "   OLLAMA GPU SETUP & VERIFICATION   " -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan

# Function to check prerequisites
function Test-Prerequisites {
    Write-Host "`n1️⃣ Checking Prerequisites..." -ForegroundColor Yellow
    
    $issues = @()
    
    # Check NVIDIA GPU
    $gpu = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($gpu) {
        Write-Host "✅ GPU: $gpu" -ForegroundColor Green
    } else {
        $issues += "No NVIDIA GPU detected"
        Write-Host "❌ No NVIDIA GPU detected" -ForegroundColor Red
    }
    
    # Check CUDA
    $cuda = & where.exe nvcc 2>$null
    if ($cuda) {
        $cudaVer = & nvcc --version 2>$null | Select-String "release" | Out-String
        Write-Host "✅ CUDA: Found at $cuda" -ForegroundColor Green
    } else {
        Write-Host "⚠️  CUDA compiler not in PATH (optional but recommended)" -ForegroundColor Yellow
    }
    
    # Check Ollama installation
    $ollama = & where.exe ollama 2>$null
    if ($ollama) {
        Write-Host "✅ Ollama: Found at $ollama" -ForegroundColor Green
    } else {
        $issues += "Ollama not found in PATH"
        Write-Host "❌ Ollama not found" -ForegroundColor Red
    }
    
    return $issues.Count -eq 0
}

# Function to setup GPU environment
function Set-GPUEnvironment {
    Write-Host "`n2️⃣ Configuring GPU Environment..." -ForegroundColor Yellow
    
    # Set environment variables
    $envVars = @{
        "OLLAMA_GPU_DRIVER" = "cuda"
        "CUDA_VISIBLE_DEVICES" = "0"
        "OLLAMA_NUM_GPU" = "1"
        "OLLAMA_GPU_LAYERS" = "999"
        "OLLAMA_HOST" = "127.0.0.1:11434"
    }
    
    foreach ($key in $envVars.Keys) {
        [Environment]::SetEnvironmentVariable($key, $envVars[$key], [EnvironmentVariableTarget]::Process)
        [Environment]::SetEnvironmentVariable($key, $envVars[$key], [EnvironmentVariableTarget]::User)
        Write-Host "   Set $key = $($envVars[$key])" -ForegroundColor Gray
    }
    
    Write-Host "✅ GPU environment configured" -ForegroundColor Green
}

# Function to restart Ollama
function Restart-Ollama {
    Write-Host "`n3️⃣ Restarting Ollama Service..." -ForegroundColor Yellow
    
    # Stop existing instances
    Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    
    # Start Ollama
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Write-Host "   Waiting for Ollama to start..." -ForegroundColor Gray
    
    # Wait for API to be ready
    $ready = $false
    for ($i = 0; $i -lt 10; $i++) {
        try {
            $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 2
            $ready = $true
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    
    if ($ready) {
        Write-Host "✅ Ollama service is running" -ForegroundColor Green
    } else {
        Write-Host "❌ Ollama failed to start" -ForegroundColor Red
        return $false
    }
    
    return $true
}

# Function to create models
function Create-Models {
    Write-Host "`n4️⃣ Creating/Updating Models..." -ForegroundColor Yellow
    
    $models = @(
        @{Name="gemma3-legal"; File="Modelfile.gemma3-legal"},
        @{Name="gemma3-quick"; File="Modelfile.gemma3-quick"}
    )
    
    foreach ($model in $models) {
        if (Test-Path $model.File) {
            Write-Host "`n   Creating $($model.Name)..." -ForegroundColor Cyan
            $output = & ollama create $model.Name -f $model.File 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   ✅ $($model.Name) created" -ForegroundColor Green
            } else {
                Write-Host "   ❌ Failed: $output" -ForegroundColor Red
            }
        } else {
            Write-Host "   ⚠️  $($model.File) not found" -ForegroundColor Yellow
        }
    }
}

# Function to test GPU acceleration
function Test-GPUAcceleration {
    Write-Host "`n5️⃣ Testing GPU Acceleration..." -ForegroundColor Yellow
    
    # Get list of models
    $modelList = & ollama list 2>$null
    if (-not $modelList -or $modelList -notmatch "gemma3") {
        Write-Host "❌ No models found. Please create models first." -ForegroundColor Red
        return
    }
    
    Write-Host "   Available models:" -ForegroundColor Gray
    Write-Host $modelList
    
    # Test with gemma3-legal
    Write-Host "`n   Testing gemma3-legal model..." -ForegroundColor Cyan
    
    # Start GPU monitoring
    $gpuJob = Start-Job -ScriptBlock {
        $readings = @()
        for ($i = 0; $i -lt 20; $i++) {
            $stats = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
            if ($stats) {
                $parts = $stats -split ','
                $readings += @{
                    GPU = [int]$parts[0].Trim()
                    Memory = [int]$parts[1].Trim()
                }
            }
            Start-Sleep -Milliseconds 500
        }
        return $readings
    }
    
    # Run inference
    $startTime = Get-Date
    Write-Host "   Running inference..." -ForegroundColor Gray
    $response = & ollama run gemma3-legal "Explain what a legal contract is in 2 sentences." 2>&1
    $duration = (Get-Date) - $startTime
    
    # Get GPU stats
    $gpuStats = Receive-Job -Job $gpuJob -Wait
    Remove-Job -Job $gpuJob
    
    # Analyze results
    $maxGPU = ($gpuStats.GPU | Measure-Object -Maximum).Maximum
    $maxMem = ($gpuStats.Memory | Measure-Object -Maximum).Maximum
    
    Write-Host "`n   📊 Results:" -ForegroundColor Green
    Write-Host "   Response time: $([math]::Round($duration.TotalSeconds, 2)) seconds" -ForegroundColor White
    Write-Host "   Peak GPU usage: $maxGPU%" -ForegroundColor White
    Write-Host "   Peak GPU memory: $maxMem MB" -ForegroundColor White
    Write-Host "   Response: $($response.Substring(0, [Math]::Min(150, $response.Length)))..." -ForegroundColor Gray
    
    if ($maxGPU -gt 15) {
        Write-Host "`n✅ GPU ACCELERATION IS WORKING!" -ForegroundColor Green
    } else {
        Write-Host "`n⚠️  Low GPU usage detected. Possible reasons:" -ForegroundColor Yellow
        Write-Host "   - Model may be too small for significant GPU usage" -ForegroundColor Gray
        Write-Host "   - First run may use less GPU (model loading)" -ForegroundColor Gray
        Write-Host "   - Try longer prompts for better GPU utilization" -ForegroundColor Gray
    }
}

# Main execution
if (-not $TestOnly) {
    if (-not (Test-Prerequisites)) {
        Write-Host "`n❌ Prerequisites not met. Please install required components." -ForegroundColor Red
        exit 1
    }
    
    Set-GPUEnvironment
    
    if (-not (Restart-Ollama)) {
        Write-Host "`n❌ Failed to start Ollama. Check installation." -ForegroundColor Red
        exit 1
    }
    
    if (-not $SkipModelCreation) {
        Create-Models
    }
}

Test-GPUAcceleration

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "        Setup Complete!              " -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Run 'nvidia-smi -l 1' in another window during inference" -ForegroundColor White
Write-Host "2. Try: ollama run gemma3-legal 'Your legal question here'" -ForegroundColor White
Write-Host "3. For issues, check: `$env:LOCALAPPDATA\Ollama\logs\server.log" -ForegroundColor White
