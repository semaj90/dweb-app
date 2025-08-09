@echo off
echo        ╔══════════════════════════════════════════════════════════════════╗
echo        ║                  🤖 YoRHa GPU Performance Test                  ║
echo        ╚══════════════════════════════════════════════════════════════════╝
echo.
echo        [TESTING] YoRHa GPU acceleration capabilities...
echo.

:: Check CUDA availability
nvidia-smi >nul 2>&1 && echo        [✓] NVIDIA GPU detected || echo        [!] NVIDIA GPU not found
nvcc --version 2>nul | find "release" && echo        [✓] CUDA Toolkit available || echo        [!] CUDA Toolkit not found

echo.
echo        [HARDWARE] GPU Information:
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader 2>nul || echo        No NVIDIA GPU information available

echo.
echo        [PERFORMANCE] Current GPU Status:
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw --format=csv 2>nul || echo        GPU performance data unavailable

echo.
echo        [TESTING] YoRHa Neural Processor Performance:
if exist go-microservice\yorha-processor-gpu.exe (
    echo        [INFO] GPU processor available - testing neural processing...
    curl -X POST -s http://localhost:8080/process -d "{\"test\":\"gpu_benchmark\"}" 2>nul | find "gpu_accelerated" >nul && echo        [✓] GPU processing confirmed || echo        [!] GPU processing test failed
) else (
    echo        [!] YoRHa GPU processor not available
)

echo.
echo        [BENCHMARK] Running GPU neural benchmark...
if exist go-microservice\yorha-processor-gpu.exe (
    powershell -Command "$start = Get-Date; try { Invoke-WebRequest -Uri 'http://localhost:8080/process-batch' -Method POST -Body '{\"documents\":[\"test1\",\"test2\",\"test3\",\"test4\",\"test5\"],\"batch_size\":5}' -ContentType 'application/json' -UseBasicParsing >$null; $end = Get-Date; $ms = ($end - $start).TotalMilliseconds; Write-Host \"        [✓] Batch processing completed in $([math]::Round($ms, 2))ms\" } catch { Write-Host '        [!] Benchmark failed - API not responding' }" 2>nul
) else (
    echo        [!] GPU processor not available for benchmark
)

echo.
echo        [MEMORY] CUDA Memory Information:
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>nul | powershell -Command "$input | ForEach-Object { $parts = $_.Split(','); $used = [int]$parts[0]; $free = [int]$parts[1]; $total = [int]$parts[2]; $usedPercent = [math]::Round(($used / $total) * 100, 1); Write-Host \"        Memory Usage: $used MB / $total MB ($usedPercent%)\" }" || echo        CUDA memory information not available

echo.
echo        ════════════════════════════════════════════════════════════════════
echo        [YoRHa] GPU performance test complete
pause