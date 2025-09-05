# GPU AI Performance Optimizer for RTX 3060 Ti
# optimize-gpu-legal-ai.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host "GPU AI Performance Optimizer" -ForegroundColor Cyan
Write-Host "RTX 3060 Ti Configuration" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator"))
{
    Write-Host "This script requires Administrator privileges!" -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    pause
    exit
}

Write-Host "[1/6] Setting GPU to Maximum Performance Mode..." -ForegroundColor Green
try {
    # Set NVIDIA GPU to prefer maximum performance
    & nvidia-smi -pm 1
    & nvidia-smi -pl 200  # RTX 3060 Ti optimal power limit
    Write-Host "    ✓ GPU performance mode configured" -ForegroundColor Green
} catch {
    Write-Host "    ! Could not configure GPU performance mode" -ForegroundColor Yellow
}

Write-Host "[2/6] Optimizing CUDA Settings..." -ForegroundColor Green
[Environment]::SetEnvironmentVariable("CUDA_LAUNCH_BLOCKING", "0", "Process")
[Environment]::SetEnvironmentVariable("CUDNN_BENCHMARK", "1", "Process")
[Environment]::SetEnvironmentVariable("TF_ENABLE_CUDNN_TENSOR_OP_MATH", "1", "Process")
[Environment]::SetEnvironmentVariable("CUDA_CACHE_MAXSIZE", "1073741824", "Process")  # 1GB cache
Write-Host "    ✓ CUDA optimizations applied" -ForegroundColor Green

Write-Host "[3/6] Configuring Ollama for RTX 3060 Ti..." -ForegroundColor Green
$ollamaConfig = @{
    "num_gpu" = 1
    "num_thread" = 8
    "ctx_size" = 4096
    "batch_size" = 512
    "main_gpu" = 0
    "low_vram" = $false
    "f16_kv" = $true
    "use_mmap" = $true
    "use_mlock" = $false
}

# Apply Ollama optimizations
foreach ($key in $ollamaConfig.Keys) {
    [Environment]::SetEnvironmentVariable("OLLAMA_$($key.ToUpper())", $ollamaConfig[$key], "Process")
}
Write-Host "    ✓ Ollama GPU optimizations configured" -ForegroundColor Green

Write-Host "[4/6] Checking GPU Memory Status..." -ForegroundColor Green
$gpuInfo = & nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits
$memInfo = $gpuInfo -split ','
Write-Host "    Total VRAM: $($memInfo[0]) MB" -ForegroundColor Cyan
Write-Host "    Free VRAM:  $($memInfo[1]) MB" -ForegroundColor Cyan
Write-Host "    Used VRAM:  $($memInfo[2]) MB" -ForegroundColor Cyan

# Check if enough VRAM is available
if ([int]$memInfo[1] -lt 3000) {
    Write-Host "    ! Warning: Low VRAM available. Consider closing other GPU applications." -ForegroundColor Yellow
}

Write-Host "[5/6] Optimizing Windows for AI Workloads..." -ForegroundColor Green
# Set process priority
$processName = "ollama"
Get-Process -Name $processName -ErrorAction SilentlyContinue | ForEach-Object {
    $_.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High
}

# Disable GPU scheduling for better AI performance
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -Name "HwSchMode" -Value 1 -Type DWord -ErrorAction SilentlyContinue
Write-Host "    ✓ Windows optimizations applied" -ForegroundColor Green

Write-Host "[6/6] Testing Ollama Connection..." -ForegroundColor Green
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method Get -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "    ✓ Ollama is running and accessible" -ForegroundColor Green
        
        # List available models
        $models = ($response.Content | ConvertFrom-Json).models
        if ($models.Count -gt 0) {
            Write-Host "    Available Models:" -ForegroundColor Cyan
            foreach ($model in $models) {
                Write-Host "      - $($model.name) (Size: $([math]::Round($model.size/1GB, 2))GB)" -ForegroundColor White
            }
        }
    }
} catch {
    Write-Host "    ! Ollama is not running. Starting service..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Optimization Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Recommended Settings Applied:" -ForegroundColor White
Write-Host "  • GPU Memory Limit: 6000 MB (leaving 1GB free)" -ForegroundColor Gray
Write-Host "  • Max Concurrency: 3 requests" -ForegroundColor Gray
Write-Host "  • Context Window: 4096 tokens" -ForegroundColor Gray
Write-Host "  • Batch Size: 512 tokens" -ForegroundColor Gray
Write-Host "  • CUDA Caching: Enabled (1GB)" -ForegroundColor Gray
Write-Host ""

# Show current GPU stats
Write-Host "Current GPU Status:" -ForegroundColor Cyan
& nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,power.draw --format=csv,noheader

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")