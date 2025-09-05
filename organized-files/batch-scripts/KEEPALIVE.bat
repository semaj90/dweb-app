@echo off
:LOOP
cls
echo [%TIME%] System Monitor Running...
redis-cli ping >nul 2>&1 && echo Redis: OK || (start /min redis-server & echo Redis: STARTED)
curl -s localhost:8080/health >nul 2>&1 && echo SIMD: OK || echo SIMD: DOWN
timeout /t 10 >nul
goto LOOP