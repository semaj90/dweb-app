# Ollama GPU Setup Script for Unsloth GGUF Models
param(
    [switch]$SkipModelCreation,
    [switch]$TestOnly
)

$ErrorActionPreference = "Continue"
$baseDir = "C:\Users\james\Desktop\deeds-web\deeds-web-app\local-models"
Set-Location $baseDir

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "   OLLAMA GGUF MODEL SETUP           " -ForegroundColor Green  
Write-Host "=====================================" -ForegroundColor Cyan

# Function to check prerequisites
function Test-Prerequisites {
    Write-Host "`n1️⃣ Checking Prerequisites..." -ForegroundColor Yellow
    
    $issues = @()
    
    # Check NVIDIA GPU
    $gpu = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($gpu) {
        Write-Host "✅ GPU: $gpu" -ForegroundColor Green
    } else {
        $issues += "No NVIDIA GPU detected"
        Write-Host "❌ No NVIDIA GPU detected" -ForegroundColor Red
    }
    
    # Check Ollama
    $ollama = & where.exe ollama 2>$null
    if ($ollama) {
        Write-Host "✅ Ollama: Found at $ollama" -ForegroundColor Green
    } else {
        $issues += "Ollama not found"
        Write-Host "❌ Ollama not found" -ForegroundColor Red
    }
    
    # Check for GGUF files
    $ggufFiles = Get-ChildItem -Filter "*.gguf" -ErrorAction SilentlyContinue
    if ($ggufFiles) {
        Write-Host "✅ GGUF files found:" -ForegroundColor Green
        $ggufFiles | ForEach-Object { Write-Host "   - $($_.Name)" -ForegroundColor Gray }
    } else {
        Write-Host "⚠️  No GGUF files found. You need to export from Unsloth first." -ForegroundColor Yellow
        Write-Host "   Expected files:" -ForegroundColor Gray
        Write-Host "   - gemma3-legal-q5_k_m.gguf" -ForegroundColor Gray
        Write-Host "   - gemma3-quick-q5_k_m.gguf" -ForegroundColor Gray
    }
    
    return $issues.Count -eq 0
}

# Function to check Modelfiles
function Test-Modelfiles {
    Write-Host "`n2️⃣ Checking Modelfiles..." -ForegroundColor Yellow
    
    $modelfiles = @(
        @{Name="Modelfile.gemma3-legal"; GGUF="gemma3-legal-q5_k_m.gguf"},
        @{Name="Modelfile.gemma3-quick"; GGUF="gemma3-quick-q5_k_m.gguf"}
    )
    
    foreach ($mf in $modelfiles) {
        if (Test-Path $mf.Name) {
            Write-Host "✅ Found $($mf.Name)" -ForegroundColor Green
            
            # Check if it references the correct GGUF
            $content = Get-Content $mf.Name -Raw
            if ($content -match "FROM \./(.+\.gguf)") {
                $referencedGGUF = $matches[1]
                if (Test-Path $referencedGGUF) {
                    Write-Host "   ✓ References existing file: $referencedGGUF" -ForegroundColor Gray
                } else {
                    Write-Host "   ⚠️  References missing file: $referencedGGUF" -ForegroundColor Yellow
                    Write-Host "   ⚠️  Expected: $($mf.GGUF)" -ForegroundColor Yellow
                }
            }
        } else {
            Write-Host "❌ Missing $($mf.Name)" -ForegroundColor Red
        }
    }
}

# Function to restart Ollama
function Restart-Ollama {
    Write-Host "`n3️⃣ Starting Ollama Server..." -ForegroundColor Yellow
    
    # Stop existing instances
    Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    
    # Set CUDA device
    $env:CUDA_VISIBLE_DEVICES = "0"
    
    # Start Ollama
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Write-Host "   Waiting for Ollama to start..." -ForegroundColor Gray
    
    # Wait for API
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
        Write-Host "✅ Ollama server is running" -ForegroundColor Green
    } else {
        Write-Host "❌ Ollama failed to start" -ForegroundColor Red
        return $false
    }
    
    return $true
}

# Function to create models
function Create-Models {
    Write-Host "`n4️⃣ Creating Models from GGUF files..." -ForegroundColor Yellow
    
    $models = @(
        @{Name="gemma3-legal"; Modelfile="Modelfile.gemma3-legal"; GGUF="gemma3-legal-q5_k_m.gguf"},
        @{Name="gemma3-quick"; Modelfile="Modelfile.gemma3-quick"; GGUF="gemma3-quick-q5_k_m.gguf"}
    )
    
    # List current models
    $existingModels = & ollama list 2>$null
    
    foreach ($model in $models) {
        Write-Host "`n   Processing $($model.Name)..." -ForegroundColor Cyan
        
        # Check if GGUF exists
        if (-not (Test-Path $model.GGUF)) {
            Write-Host "   ⚠️  Skipping - GGUF file not found: $($model.GGUF)" -ForegroundColor Yellow
            continue
        }
        
        # Check if Modelfile exists
        if (-not (Test-Path $model.Modelfile)) {
            Write-Host "   ⚠️  Skipping - Modelfile not found: $($model.Modelfile)" -ForegroundColor Yellow
            continue
        }
        
        # Remove existing model if present
        if ($existingModels -match $model.Name) {
            Write-Host "   Removing existing model..." -ForegroundColor Gray
            & ollama rm $model.Name 2>$null
        }
        
        # Create model
        Write-Host "   Creating model from GGUF..." -ForegroundColor Gray
        $output = & ollama create $model.Name -f $model.Modelfile 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $($model.Name) created successfully" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Failed to create $($model.Name)" -ForegroundColor Red
            Write-Host "   Error: $output" -ForegroundColor Red
        }
    }
}

# Function to test GPU acceleration
function Test-GPUAcceleration {
    Write-Host "`n5️⃣ Testing GPU Acceleration..." -ForegroundColor Yellow
    
    # List models
    $modelList = & ollama list 2>$null
    if (-not $modelList -or $modelList -notmatch "gemma3") {
        Write-Host "❌ No models found. Create models first." -ForegroundColor Red
        return
    }
    
    Write-Host "   Available models:" -ForegroundColor Gray
    Write-Host $modelList
    
    # Test with gemma3-legal
    $testModel = "gemma3-legal"
    if ($modelList -notmatch $testModel) {
        $testModel = "gemma3-quick"
    }
    
    Write-Host "`n   Testing $testModel model..." -ForegroundColor Cyan
    
    # Start GPU monitoring
    $gpuJob = Start-Job -ScriptBlock {
        $readings = @()
        for ($i = 0; $i -lt 30; $i++) {
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
    $testPrompt = "Explain the legal concept of consideration in contract law in 3 sentences."
    $response = & ollama run $testModel $testPrompt 2>&1
    $duration = (Get-Date) - $startTime
    
    # Get GPU stats
    $gpuStats = Receive-Job -Job $gpuJob -Wait
    Remove-Job -Job $gpuJob
    
    # Analyze results
    $maxGPU = ($gpuStats.GPU | Measure-Object -Maximum).Maximum
    $maxMem = ($gpuStats.Memory | Measure-Object -Maximum).Maximum
    $avgGPU = ($gpuStats.GPU | Measure-Object -Average).Average
    
    Write-Host "`n   📊 Performance Results:" -ForegroundColor Green
    Write-Host "   Response time: $([math]::Round($duration.TotalSeconds, 2)) seconds" -ForegroundColor White
    Write-Host "   Peak GPU usage: $maxGPU%" -ForegroundColor White
    Write-Host "   Average GPU usage: $([math]::Round($avgGPU, 1))%" -ForegroundColor White
    Write-Host "   Peak GPU memory: $maxMem MB" -ForegroundColor White
    
    # Calculate tokens/sec (rough estimate)
    $responseLength = $response.Length
    $tokensEstimate = $responseLength / 4  # Rough approximation
    $tokensPerSec = $tokensEstimate / $duration.TotalSeconds
    Write-Host "   Estimated speed: $([math]::Round($tokensPerSec, 1)) tokens/sec" -ForegroundColor White
    
    Write-Host "`n   Response preview:" -ForegroundColor Gray
    Write-Host "   $($response.Substring(0, [Math]::Min(200, $response.Length)))..." -ForegroundColor Gray
    
    if ($maxGPU -gt 15) {
        Write-Host "`n✅ GPU ACCELERATION IS WORKING!" -ForegroundColor Green
        Write-Host "   Your model is using the GPU for inference." -ForegroundColor Gray
    } else {
        Write-Host "`n⚠️  Low GPU usage detected ($maxGPU% peak)" -ForegroundColor Yellow
        Write-Host "   Possible reasons:" -ForegroundColor Gray
        Write-Host "   - First run (model loading)" -ForegroundColor Gray
        Write-Host "   - Small prompt/response" -ForegroundColor Gray
        Write-Host "   - Model may need recreation" -ForegroundColor Gray
    }
}

# Function to show web integration example
function Show-WebIntegration {
    Write-Host "`n6️⃣ Web App Integration Example..." -ForegroundColor Yellow
    
    Write-Host @"
   
   Your SvelteKit app can connect to Ollama at: http://localhost:11434
   
   Example API call:
   
   const response = await fetch('http://localhost:11434/api/generate', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       model: 'gemma3-legal',
       prompt: 'What is a breach of contract?',
       stream: false
     })
   });
   
   const data = await response.json();
   console.log(data.response);
"@ -ForegroundColor Cyan
}

# Main execution
if (-not $TestOnly) {
    if (-not (Test-Prerequisites)) {
        Write-Host "`n⚠️  Some prerequisites missing, but continuing..." -ForegroundColor Yellow
    }
    
    Test-Modelfiles
    
    if (-not (Restart-Ollama)) {
        Write-Host "`n❌ Failed to start Ollama." -ForegroundColor Red
        exit 1
    }
    
    if (-not $SkipModelCreation) {
        Create-Models
    }
}

Test-GPUAcceleration
Show-WebIntegration

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "        Setup Complete!              " -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Ensure GGUF files are in this directory" -ForegroundColor White
Write-Host "2. Run 'nvidia-smi -l 1' during inference to monitor GPU" -ForegroundColor White
Write-Host "3. Test: ollama run gemma3-legal 'Your legal question'" -ForegroundColor White
Write-Host "4. Check web app integration at src/lib/ai/ollama.ts" -ForegroundColor White
