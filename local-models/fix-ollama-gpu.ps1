# Ollama GPU Acceleration Fix Script
Write-Host "`n🚀 FIXING OLLAMA GPU ACCELERATION" -ForegroundColor Green
Write-Host "==================================`n" -ForegroundColor Cyan

# Function to test GPU
function Test-GPUAcceleration {
    Write-Host "`n📊 GPU Status Check:" -ForegroundColor Yellow
    
    # Check NVIDIA GPU
    $gpu = & nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>$null
    if ($gpu) {
        Write-Host "✅ GPU Found: $gpu" -ForegroundColor Green
    } else {
        Write-Host "❌ No NVIDIA GPU detected" -ForegroundColor Red
        return $false
    }
    
    # Check CUDA
    $cuda = & where.exe nvcc 2>$null
    if ($cuda) {
        Write-Host "✅ CUDA Compiler found: $cuda" -ForegroundColor Green
    } else {
        Write-Host "⚠️  CUDA toolkit not in PATH" -ForegroundColor Yellow
    }
    
    return $true
}

# Step 1: Stop Ollama
Write-Host "1️⃣ Stopping Ollama service..." -ForegroundColor Yellow
Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Write-Host "✅ Ollama stopped" -ForegroundColor Green

# Step 2: Set GPU Environment Variables
Write-Host "`n2️⃣ Configuring GPU environment..." -ForegroundColor Yellow

# Set for current session
$env:OLLAMA_GPU_DRIVER = "cuda"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OLLAMA_NUM_GPU = "1"
$env:OLLAMA_GPU_LAYERS = "999"
$env:OLLAMA_HOST = "127.0.0.1:11434"

# Set permanently
[Environment]::SetEnvironmentVariable("OLLAMA_GPU_DRIVER", "cuda", [EnvironmentVariableTarget]::User)
[Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0", [EnvironmentVariableTarget]::User)
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_GPU", "1", [EnvironmentVariableTarget]::User)
[Environment]::SetEnvironmentVariable("OLLAMA_GPU_LAYERS", "999", [EnvironmentVariableTarget]::User)

Write-Host "✅ GPU environment configured" -ForegroundColor Green

# Step 3: Start Ollama with GPU
Write-Host "`n3️⃣ Starting Ollama with GPU support..." -ForegroundColor Yellow
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 3

# Test API
$apiTest = try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 } catch { $null }
if ($apiTest) {
    Write-Host "✅ Ollama API responding" -ForegroundColor Green
} else {
    Write-Host "⚠️  Ollama API not ready, waiting..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}

# Step 4: Create models
Write-Host "`n4️⃣ Creating models with GPU acceleration..." -ForegroundColor Yellow
Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app\local-models"

# Create each model
$models = @(
    @{Name="gemma3-legal"; File="Modelfile.gemma3-legal"},
    @{Name="gemma3-quick"; File="Modelfile.gemma3-quick"},
    @{Name="legal-ai"; File="Modelfile"}
)

foreach ($model in $models) {
    if (Test-Path $model.File) {
        Write-Host "`n📦 Creating $($model.Name)..." -ForegroundColor Cyan
        & ollama create $model.Name -f $model.File
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $($model.Name) created successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to create $($model.Name)" -ForegroundColor Red
        }
    } else {
        Write-Host "⚠️  $($model.File) not found" -ForegroundColor Yellow
    }
}

# Step 5: Test GPU acceleration
Write-Host "`n5️⃣ Testing GPU acceleration..." -ForegroundColor Yellow
Test-GPUAcceleration

Write-Host "`n✅ Setup complete! Run test-gpu.ps1 to verify GPU acceleration." -ForegroundColor Green
