@echo off
echo        ╔══════════════════════════════════════════════════════════════════╗
echo        ║                🤖 YoRHa System Graceful Shutdown                ║
echo        ╚══════════════════════════════════════════════════════════════════╝
echo.
echo        [YoRHa] Initiating graceful system shutdown sequence...

echo        [1/5] Notifying YoRHa API processing units...
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8080/shutdown' -Method POST -TimeoutSec 5 -UseBasicParsing >$null 2>&1 } catch { }" 2>nul

echo        [2/5] Allowing active neural processes to complete...
timeout /t 5 /nobreak >nul

echo        [3/5] Terminating YoRHa neural processors...
taskkill /F /IM yorha-processor-cpu.exe 2>nul
taskkill /F /IM yorha-processor-gpu.exe 2>nul
echo        [✓] YoRHa neural processors terminated

echo        [4/5] Shutting down cache systems...
taskkill /F /IM redis-server.exe 2>nul
echo        [✓] YoRHa cache systems stopped

echo        [5/5] Stopping monitoring systems...
taskkill /F /IM cmd.exe /FI "WINDOWTITLE eq YoRHa Neural Monitor*" 2>nul
echo        [✓] YoRHa monitoring systems stopped

echo.
echo        ╔══════════════════════════════════════════════════════════════════╗
echo        ║              ✅ YoRHa System Shutdown Complete                   ║
echo        ╚══════════════════════════════════════════════════════════════════╝
echo        [INFO] All YoRHa neural networks safely terminated
echo        [INFO] System ready for maintenance or restart procedures