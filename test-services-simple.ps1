# Simple Test for Legal AI Services
param([string]$Action = "test")

function Write-ColorOutput($Message, $Color = "White") {
    switch ($Color) {
        "Success" { Write-Host $Message -ForegroundColor Green }
        "Warning" { Write-Host $Message -ForegroundColor Yellow }
        "Error" { Write-Host $Message -ForegroundColor Red }
        "Info" { Write-Host $Message -ForegroundColor Cyan }
        default { Write-Host $Message -ForegroundColor White }
    }
}

function Test-Configuration {
    Write-ColorOutput "Testing Legal AI Platform Configuration..." "Info"
    
    # Check configuration file
    $configPath = "config\production.json"
    if (Test-Path $configPath) {
        Write-ColorOutput "✓ Production configuration found" "Success"
        
        try {
            $config = Get-Content $configPath | ConvertFrom-Json
            Write-ColorOutput "✓ Configuration is valid JSON" "Success"
            Write-ColorOutput "  Service: $($config.service_name)" "Info"
            Write-ColorOutput "  GPU Enabled: $($config.gpu.enabled)" "Info"
            Write-ColorOutput "  CUDA Version: $($config.gpu.cuda_version)" "Info"
            Write-ColorOutput "  GPU Memory Limit: $($config.gpu.memory_limit_mb) MB" "Info"
        }
        catch {
            Write-ColorOutput "✗ Configuration JSON is invalid" "Error"
            return $false
        }
    } else {
        Write-ColorOutput "✗ Configuration not found" "Error"
        return $false
    }
    
    # Check executables
    $executables = @("gpu-orchestrator-prod.exe", "health-server-prod.exe", "envutil-prod.exe")
    
    foreach ($exe in $executables) {
        if (Test-Path $exe) {
            $size = (Get-Item $exe).Length / 1MB
            Write-ColorOutput "✓ Executable found: $exe ($([math]::Round($size, 2)) MB)" "Success"
        } else {
            Write-ColorOutput "✗ Executable missing: $exe" "Error"
            return $false
        }
    }
    
    return $true
}

function Test-ServiceHealth {
    Write-ColorOutput "Testing Service Health..." "Info"
    
    # Start GPU orchestrator
    Write-ColorOutput "Starting GPU Orchestrator..." "Info"
    $gpuProcess = Start-Process -FilePath ".\gpu-orchestrator-prod.exe" -PassThru -WindowStyle Hidden
    
    Start-Sleep -Seconds 4
    
    try {
        # Test health endpoint
        $healthResponse = Invoke-RestMethod -Uri "http://localhost:8094/health" -Method GET -TimeoutSec 10
        Write-ColorOutput "✓ GPU Orchestrator health check passed" "Success"
        Write-ColorOutput "  Status: $($healthResponse.status)" "Info"
        
        # Test GPU processing
        $testData = @{
            data = @(1.0, 2.0, 3.0, 4.0, 5.0)
            operation = "multiply"
        } | ConvertTo-Json
        
        $processResult = Invoke-RestMethod -Uri "http://localhost:8094/api/gpu/process" -Method POST -Body $testData -ContentType "application/json" -TimeoutSec 10
        Write-ColorOutput "✓ GPU processing endpoint working" "Success"
        Write-ColorOutput "  Processed: $($processResult.processed) elements" "Info"
        
        $success = $true
    }
    catch {
        Write-ColorOutput "✗ Service health test failed: $($_.Exception.Message)" "Error"
        $success = $false
    }
    finally {
        if ($gpuProcess -and !$gpuProcess.HasExited) {
            Stop-Process -Id $gpuProcess.Id -Force
            Write-ColorOutput "✓ GPU Orchestrator stopped" "Info"
        }
    }
    
    return $success
}

function Test-SystemInfo {
    Write-ColorOutput "System Information..." "Info"
    
    # Check system memory
    $totalMemory = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
    Write-ColorOutput "Total System Memory: $totalMemory GB" "Info"
    
    if ($totalMemory -ge 16) {
        Write-ColorOutput "✓ Sufficient memory for production" "Success"
    } else {
        Write-ColorOutput "⚠ Consider increasing system memory" "Warning"
    }
    
    # Check GPU
    try {
        $gpuInfo = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
        if ($gpuInfo) {
            Write-ColorOutput "✓ NVIDIA GPU detected: $($gpuInfo.Name)" "Success"
            
            # Check CUDA
            if ($env:CUDA_PATH) {
                Write-ColorOutput "✓ CUDA installation detected: $($env:CUDA_PATH)" "Success"
            } else {
                Write-ColorOutput "⚠ CUDA path not found in environment variables" "Warning"
            }
        } else {
            Write-ColorOutput "⚠ No NVIDIA GPU detected" "Warning"
        }
    }
    catch {
        Write-ColorOutput "⚠ Could not detect GPU information" "Warning"
    }
    
    return $true
}

# Main execution
Write-ColorOutput "Legal AI Platform - Production Service Test Suite" "Info"
Write-ColorOutput "=================================================" "Info"

$allTestsPassed = $true

switch ($Action.ToLower()) {
    "test" {
        if (!(Test-Configuration)) { $allTestsPassed = $false }
        Write-ColorOutput "" "Info"
        
        if (!(Test-SystemInfo)) { $allTestsPassed = $false }
        Write-ColorOutput "" "Info"
        
        if (!(Test-ServiceHealth)) { $allTestsPassed = $false }
    }
    "config" {
        $allTestsPassed = Test-Configuration
    }
    "health" {
        $allTestsPassed = Test-ServiceHealth
    }
    "info" {
        $allTestsPassed = Test-SystemInfo
    }
    default {
        Write-ColorOutput "Usage: test-services-simple.ps1 -Action [test|config|health|info]" "Info"
        exit 0
    }
}

Write-ColorOutput "" "Info"
Write-ColorOutput "============================================" "Info"

if ($allTestsPassed) {
    Write-ColorOutput "🎉 ALL TESTS PASSED - Production Ready!" "Success"
} else {
    Write-ColorOutput "❌ Some tests failed. Check output above." "Error"
    exit 1
}

Write-ColorOutput "Test completed successfully." "Success"