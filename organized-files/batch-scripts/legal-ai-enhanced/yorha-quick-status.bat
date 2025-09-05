@echo off
echo        ╔══════════════════════════════════════════════════════════════════╗
echo        ║              🤖 YoRHa Quick Neural Status Check                 ║
echo        ╚══════════════════════════════════════════════════════════════════╝
echo.

echo        [SCANNING] YoRHa neural network components...
echo.

:: Check YoRHa neural processors
tasklist /FI "IMAGENAME eq yorha-processor*" 2>nul | find "yorha-processor" >nul && echo        [✓] Neural Processors: OPERATIONAL || echo        [✗] Neural Processors: OFFLINE

:: Check YoRHa API processing unit
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:8080/health' -TimeoutSec 3 -UseBasicParsing; if ($response.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1 && echo        [✓] API Processing: OPERATIONAL || echo        [✗] API Processing: OFFLINE

:: Check YoRHa database neural link
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U yorha_neural_admin -h localhost -d yorha_neural_ai_db -t -c "SELECT 1" >nul 2>&1 && echo        [✓] Database Neural Link: ONLINE || echo        [✗] Database Neural Link: OFFLINE

:: Check YoRHa cache network
powershell -Command "try { $tcp = New-Object System.Net.Sockets.TcpClient; $tcp.Connect('localhost', 6379); $tcp.Close(); exit 0 } catch { exit 1 }" >nul 2>&1 && echo        [✓] Cache Network: OPERATIONAL || echo        [✗] Cache Network: OFFLINE

:: Check CUDA processing capability
nvidia-smi >nul 2>&1 && echo        [✓] CUDA Processing: AVAILABLE || echo        [!] CUDA Processing: CPU MODE

echo.
echo        ════════════════════════════════════════════════════════════════════
echo        [YoRHa] Neural network status check complete