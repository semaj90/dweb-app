# GPU Acceleration Test Script for Ollama
Write-Host "`n🧪 OLLAMA GPU ACCELERATION TEST" -ForegroundColor Green
Write-Host "================================`n" -ForegroundColor Cyan

# Function to monitor GPU during inference
function Monitor-GPUUsage {
    param($ModelName, $Prompt)
    
    Write-Host "`n📊 Testing $ModelName with GPU monitoring..." -ForegroundColor Yellow
    
    # Start GPU monitoring in background
    $gpuMonitor = Start-Job -ScriptBlock {
        $maxUsage = 0
        $maxMemory = 0
        for ($i = 0; $i -lt 30; $i++) {
            $stats = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
            if ($stats) {
                $parts = $stats -split ','
                $usage = [int]$parts[0].Trim()
                $memory = [int]$parts[1].Trim()
                if ($usage -gt $maxUsage) { $maxUsage = $usage }
                if ($memory -gt $maxMemory) { $maxMemory = $memory }
            }
            Start-Sleep -Milliseconds 500
        }
        return @{MaxGPU=$maxUsage; MaxMemory=$maxMemory}
    }
    
    # Run inference
    $startTime = Get-Date
    try {
        $response = & ollama run $ModelName $Prompt 2>&1
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        # Get GPU stats
        $gpuStats = Receive-Job -Job $gpuMonitor -Wait
        Remove-Job -Job $gpuMonitor
        
        # Display results
        Write-Host "`n✅ Results for $ModelName`:" -ForegroundColor Green
        Write-Host "⏱️  Response Time: $([math]::Round($duration, 2)) seconds" -ForegroundColor White
        Write-Host "🎮  Peak GPU Usage: $($gpuStats.MaxGPU)%" -ForegroundColor White
        Write-Host "💾  Peak GPU Memory: $($gpuStats.MaxMemory) MB" -ForegroundColor White
        Write-Host "📝  Response Preview: $($response.Substring(0, [Math]::Min(100, $response.Length)))..." -ForegroundColor Gray
        
        # Performance analysis
        if ($gpuStats.MaxGPU -gt 20) {
            Write-Host "✅ GPU acceleration is working!" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Low GPU usage detected - may be using CPU" -ForegroundColor Yellow
        }
        
        return @{
            Model = $ModelName
            Duration = $duration
            GPUUsage = $gpuStats.MaxGPU
            GPUMemory = $gpuStats.MaxMemory
            TokensPerSec = if ($response.Length -gt 0) { [math]::Round($response.Length / 4 / $duration, 1) } else { 0 }
        }
    } catch {
        Write-Host "❌ Error testing $ModelName`: $($_.Exception.Message)" -ForegroundColor Red
        Stop-Job -Job $gpuMonitor -ErrorAction SilentlyContinue
        Remove-Job -Job $gpuMonitor -ErrorAction SilentlyContinue
        return $null
    }
}

# Check GPU availability
Write-Host "1️⃣ Checking GPU availability..." -ForegroundColor Yellow
$gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
if ($gpuInfo) {
    Write-Host "✅ GPU: $gpuInfo" -ForegroundColor Green
} else {
    Write-Host "❌ No NVIDIA GPU detected!" -ForegroundColor Red
    exit 1
}

# Check Ollama status
Write-Host "`n2️⃣ Checking Ollama service..." -ForegroundColor Yellow
$ollamaRunning = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ollamaRunning) {
    Write-Host "✅ Ollama is running (PID: $($ollamaRunning.Id))" -ForegroundColor Green
} else {
    Write-Host "⚠️  Starting Ollama..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# List available models
Write-Host "`n3️⃣ Available models:" -ForegroundColor Yellow
$models = & ollama list 2>$null
if ($models) {
    Write-Host $models -ForegroundColor White
} else {
    Write-Host "⚠️  No models found. Run fix-ollama-gpu.ps1 first!" -ForegroundColor Yellow
    exit 1
}

# Test prompts
$testPrompts = @{
    "Quick" = "What is a contract?"
    "Medium" = "Explain the key elements of a valid contract under common law."
    "Complex" = "Analyze the enforceability of a verbal agreement for the sale of real property worth $500,000 made between two business partners, considering the statute of frauds and potential exceptions."
}

# Test each available model
Write-Host "`n4️⃣ Running GPU acceleration tests..." -ForegroundColor Yellow
$results = @()

$modelsToTest = @("gemma3-legal", "gemma3-quick", "legal-ai") | Where-Object {
    $models -match $_
}

foreach ($model in $modelsToTest) {
    foreach ($testType in $testPrompts.Keys) {
        Write-Host "`n🔄 Testing $model with $testType prompt..." -ForegroundColor Cyan
        $result = Monitor-GPUUsage -ModelName $model -Prompt $testPrompts[$testType]
        if ($result) {
            $result.TestType = $testType
            $results += $result
        }
        Start-Sleep -Seconds 2  # Cool down between tests
    }
}

# Summary
Write-Host "`n📊 PERFORMANCE SUMMARY" -ForegroundColor Green
Write-Host "=====================" -ForegroundColor Cyan
$results | Format-Table -Property Model, TestType, Duration, GPUUsage, GPUMemory, TokensPerSec -AutoSize

# GPU acceleration verdict
$avgGPUUsage = ($results | Measure-Object -Property GPUUsage -Average).Average
if ($avgGPUUsage -gt 20) {
    Write-Host "`n✅ GPU ACCELERATION IS WORKING!" -ForegroundColor Green
    Write-Host "Average GPU usage: $([math]::Round($avgGPUUsage, 1))%" -ForegroundColor White
} else {
    Write-Host "`n⚠️  GPU ACCELERATION MAY NOT BE WORKING PROPERLY" -ForegroundColor Yellow
    Write-Host "Average GPU usage: $([math]::Round($avgGPUUsage, 1))%" -ForegroundColor White
    Write-Host "Try running: .\fix-ollama-gpu.ps1" -ForegroundColor Yellow
}

Write-Host "`n💡 Tips for better GPU performance:" -ForegroundColor Cyan
Write-Host "  - Use larger models (9b, 13b) for better GPU utilization" -ForegroundColor White
Write-Host "  - Increase context length with larger prompts" -ForegroundColor White
Write-Host "  - Ensure CUDA is properly installed" -ForegroundColor White
Write-Host "  - Check nvidia-smi during inference for real-time stats" -ForegroundColor White
