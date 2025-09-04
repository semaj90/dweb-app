# Test Legal AI Services - No Admin Required
# Tests the production configuration and service functionality

param(
    [Parameter(Mandatory=$false)]
    [string]$Action = "test"
)

# Colors for output
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
}

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Test-Configuration {
    Write-ColorOutput "Testing Legal AI Platform Configuration..." "Info"
    Write-ColorOutput "=====================================" "Info"
    
    # Check configuration file
    $configPath = "config\production.json"
    if (Test-Path $configPath) {
        Write-ColorOutput "✓ Production configuration found: $configPath" "Success"
        
        try {
            $config = Get-Content $configPath | ConvertFrom-Json
            Write-ColorOutput "✓ Configuration is valid JSON" "Success"
            Write-ColorOutput "  Service: $($config.service_name)" "Info"
            Write-ColorOutput "  GPU Enabled: $($config.gpu.enabled)" "Info"
            Write-ColorOutput "  GPU Memory Limit: $($config.gpu.memory_limit_mb) MB" "Info"
            Write-ColorOutput "  Worker Count: $($config.performance.worker_count)" "Info"
        }
        catch {
            Write-ColorOutput "✗ Configuration JSON is invalid: $($_.Exception.Message)" "Error"
            return $false
        }
    } else {
        Write-ColorOutput "✗ Configuration not found: $configPath" "Error"
        return $false
    }
    
    # Check required directories
    $directories = @("config", "logs", "cache", "data")
    foreach ($dir in $directories) {
        if (Test-Path $dir) {
            Write-ColorOutput "✓ Directory exists: $dir" "Success"
        } else {
            Write-ColorOutput "✗ Directory missing: $dir" "Error"
            return $false
        }
    }
    
    # Check executables
    $executables = @(
        "gpu-orchestrator-prod.exe",
        "health-server-prod.exe", 
        "envutil-prod.exe"
    )
    
    foreach ($exe in $executables) {
        if (Test-Path $exe) {
            $size = (Get-Item $exe).Length
            Write-ColorOutput "✓ Executable found: $exe ($([math]::Round($size/1MB, 2)) MB)" "Success"
        } else {
            Write-ColorOutput "✗ Executable missing: $exe" "Error"
            return $false
        }
    }
    
    return $true
}

function Test-ServiceHealth {
    Write-ColorOutput "Testing Service Health Endpoints..." "Info"
    Write-ColorOutput "=================================" "Info"
    
    # Start GPU orchestrator in background
    Write-ColorOutput "Starting GPU Orchestrator..." "Info"
    $gpuProcess = Start-Process -FilePath ".\gpu-orchestrator-prod.exe" -PassThru -WindowStyle Hidden
    
    # Wait for startup
    Start-Sleep -Seconds 3
    
    try {
        # Test health endpoint
        $healthResponse = Invoke-RestMethod -Uri "http://localhost:8094/health" -Method GET -TimeoutSec 5
        Write-ColorOutput "✓ GPU Orchestrator health check passed" "Success"
        Write-ColorOutput "  Status: $($healthResponse.status)" "Info"
        Write-ColorOutput "  Service: $($healthResponse.service)" "Info"
        
        # Test GPU status endpoint
        $gpuStatus = Invoke-RestMethod -Uri "http://localhost:8094/api/gpu/status" -Method GET -TimeoutSec 5
        Write-ColorOutput "✓ GPU status endpoint accessible" "Success"
        Write-ColorOutput "  GPU Available: $($gpuStatus.gpu_available)" "Info"
        Write-ColorOutput "  CUDA Version: $($gpuStatus.cuda_version)" "Info"
        
        # Test GPU processing endpoint
        $testData = @{
            data = @(1.0, 2.0, 3.0, 4.0, 5.0)
            operation = "multiply"
        } | ConvertTo-Json
        
        $processResult = Invoke-RestMethod -Uri "http://localhost:8094/api/gpu/process" -Method POST -Body $testData -ContentType "application/json" -TimeoutSec 10
        Write-ColorOutput "✓ GPU processing endpoint working" "Success"
        Write-ColorOutput "  Processed: $($processResult.processed) elements" "Info"
        Write-ColorOutput "  Result: [$($processResult.result -join ', ')]" "Info"
        
        $success = $true
    }
    catch {
        Write-ColorOutput "✗ Service health test failed: $($_.Exception.Message)" "Error"
        $success = $false
    }
    finally {
        # Stop the process
        if ($gpuProcess -and !$gpuProcess.HasExited) {
            $gpuProcess.Kill()
            Write-ColorOutput "✓ GPU Orchestrator stopped" "Info"
        }
    }
    
    return $success
}

function Test-MemoryOptimization {
    Write-ColorOutput "Testing Memory Optimization..." "Info"
    Write-ColorOutput "=============================" "Info"
    
    # Check system memory
    $totalMemory = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
    Write-ColorOutput "Total System Memory: $totalMemory GB" "Info"
    
    if ($totalMemory -ge 16) {
        Write-ColorOutput "✓ Sufficient memory for production deployment" "Success"
    } else {
        Write-ColorOutput "⚠ Less than 16GB RAM detected. Consider reducing memory limits." "Warning"
    }
    
    # Check GPU memory if available
    try {
        $gpuInfo = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
        if ($gpuInfo) {
            $gpuMemory = [math]::Round($gpuInfo.AdapterRAM / 1GB, 2)
            Write-ColorOutput "✓ NVIDIA GPU detected: $($gpuInfo.Name)" "Success"
            Write-ColorOutput "  GPU Memory: $gpuMemory GB" "Info"
            
            if ($gpuMemory -ge 6) {
                Write-ColorOutput "✓ Sufficient GPU memory for production workloads" "Success"
            } else {
                Write-ColorOutput "⚠ GPU memory less than 6GB. Consider reducing GPU memory limits." "Warning"
            }
        } else {
            Write-ColorOutput "⚠ No NVIDIA GPU detected. GPU acceleration will be disabled." "Warning"
        }
    }
    catch {
        Write-ColorOutput "⚠ Could not detect GPU information: $($_.Exception.Message)" "Warning"
    }
    
    return $true
}

function Test-PerformanceBenchmark {
    Write-ColorOutput "Running Performance Benchmark..." "Info"
    Write-ColorOutput "===============================" "Info"
    
    # Start GPU orchestrator for benchmarking
    Write-ColorOutput "Starting services for benchmark..." "Info"
    $gpuProcess = Start-Process -FilePath ".\gpu-orchestrator-prod.exe" -PassThru -WindowStyle Hidden
    
    Start-Sleep -Seconds 3
    
    try {
        $totalRequests = 50
        $successful = 0
        $totalTime = 0
        
        Write-ColorOutput "Sending $totalRequests test requests..." "Info"
        
        for ($i = 1; $i -le $totalRequests; $i++) {
            try {
                $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
                
                # Generate test data of varying sizes
                $dataSize = 10 + ($i % 100)  # 10 to 110 elements
                $testData = @{
                    data = @(1..$dataSize | ForEach-Object { [float]($_ * 0.1) })
                    operation = @("multiply", "normalize", "relu")[$i % 3]
                } | ConvertTo-Json
                
                $result = Invoke-RestMethod -Uri "http://localhost:8094/api/gpu/process" -Method POST -Body $testData -ContentType "application/json" -TimeoutSec 5
                
                $stopwatch.Stop()
                $totalTime += $stopwatch.ElapsedMilliseconds
                $successful++
                
                if ($i % 10 -eq 0) {
                    Write-ColorOutput "  Completed $i/$totalRequests requests..." "Info"
                }
            }
            catch {
                Write-ColorOutput "  Request $i failed: $($_.Exception.Message)" "Warning"
            }
        }
        
        if ($successful -gt 0) {
            $avgLatency = [math]::Round($totalTime / $successful, 2)
            $throughput = [math]::Round($successful / ($totalTime / 1000), 2)
            
            Write-ColorOutput "✓ Benchmark completed" "Success"
            Write-ColorOutput "  Successful requests: $successful/$totalRequests" "Info"
            Write-ColorOutput "  Average latency: $avgLatency ms" "Info"
            Write-ColorOutput "  Throughput: $throughput requests/second" "Info"
            
            if ($avgLatency -lt 100) {
                Write-ColorOutput "✓ Excellent performance (< 100ms average)" "Success"
            } elseif ($avgLatency -lt 200) {
                Write-ColorOutput "✓ Good performance (< 200ms average)" "Success"
            } else {
                Write-ColorOutput "⚠ Performance could be improved (> 200ms average)" "Warning"
            }
            
            return $true
        } else {
            Write-ColorOutput "✗ All benchmark requests failed" "Error"
            return $false
        }
    }
    catch {
        Write-ColorOutput "✗ Benchmark failed: $($_.Exception.Message)" "Error"
        return $false
    }
    finally {
        if ($gpuProcess -and !$gpuProcess.HasExited) {
            $gpuProcess.Kill()
            Write-ColorOutput "✓ Benchmark services stopped" "Info"
        }
    }
}

# Main execution
Write-ColorOutput "Legal AI Platform - Production Service Test Suite" "Info"
Write-ColorOutput "=================================================" "Info"

$allTestsPassed = $true

# Run tests based on action
switch ($Action.ToLower()) {
    "test" {
        Write-ColorOutput "Running all tests..." "Info"
        
        if (!(Test-Configuration)) { $allTestsPassed = $false }
        Write-ColorOutput ""
        
        if (!(Test-MemoryOptimization)) { $allTestsPassed = $false }
        Write-ColorOutput ""
        
        if (!(Test-ServiceHealth)) { $allTestsPassed = $false }
        Write-ColorOutput ""
        
        if (!(Test-PerformanceBenchmark)) { $allTestsPassed = $false }
    }
    
    "config" {
        $allTestsPassed = Test-Configuration
    }
    
    "health" {
        $allTestsPassed = Test-ServiceHealth
    }
    
    "memory" {
        $allTestsPassed = Test-MemoryOptimization
    }
    
    "benchmark" {
        $allTestsPassed = Test-PerformanceBenchmark
    }
    
    default {
        Write-ColorOutput "Usage: test-services.ps1 -Action <test|config|health|memory|benchmark>" "Info"
        Write-ColorOutput ""
        Write-ColorOutput "Actions:" "Info"
        Write-ColorOutput "  test      - Run all tests" "Info"
        Write-ColorOutput "  config    - Test configuration only" "Info"
        Write-ColorOutput "  health    - Test service health endpoints" "Info"
        Write-ColorOutput "  memory    - Test memory optimization" "Info"
        Write-ColorOutput "  benchmark - Run performance benchmark" "Info"
        exit 0
    }
}

Write-ColorOutput ""
Write-ColorOutput "============================================" "Info"

if ($allTestsPassed) {
    Write-ColorOutput "🎉 ALL TESTS PASSED - Production Ready!" "Success"
    Write-ColorOutput "Your Legal AI Platform services are optimized and ready for deployment." "Success"
} else {
    Write-ColorOutput "❌ Some tests failed. Please review the output above." "Error"
    exit 1
}

Write-ColorOutput "Test completed successfully." "Success"