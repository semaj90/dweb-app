@echo off
REM Memory Optimizer Launcher
REM Quick access to memory optimization features

cls
echo =====================================
echo    Legal AI Memory Optimizer
echo =====================================
echo.
echo   1. Show Memory Status
echo   2. Optimize Memory Now
echo   3. Start Memory Monitor
echo   4. Monitor with Auto-Optimize
echo   5. Generate Memory Report
echo   6. View Crash Prevention Logs
echo   7. Exit
echo.

set /p choice="Select option (1-7): "

if "%choice%"=="1" goto status
if "%choice%"=="2" goto optimize
if "%choice%"=="3" goto monitor
if "%choice%"=="4" goto automonitor
if "%choice%"=="5" goto report
if "%choice%"=="6" goto logs
if "%choice%"=="7" goto end

:status
echo.
powershell -ExecutionPolicy Bypass -File scripts\memory-optimizer.ps1
pause
goto end

:optimize
echo.
echo Optimizing memory...
powershell -ExecutionPolicy Bypass -File scripts\memory-optimizer.ps1 -Optimize
pause
goto end

:monitor
echo.
echo Starting memory monitor (Press Ctrl+C to stop)...
powershell -ExecutionPolicy Bypass -File scripts\memory-optimizer.ps1 -Monitor
goto end

:automonitor
echo.
echo Starting monitor with auto-optimization...
powershell -ExecutionPolicy Bypass -File scripts\memory-optimizer.ps1 -Monitor -AutoOptimize
goto end

:report
echo.
echo Generating memory report...
powershell -ExecutionPolicy Bypass -Command "& { cd scripts; .\memory-optimizer.ps1 | Out-Null; Get-Content ..\logs\memory\memory-report-*.json | Select-Object -Last 1 | ConvertFrom-Json | ConvertTo-Json -Depth 5 }"
pause
goto end

:logs
echo.
echo Opening logs folder...
if exist "logs\memory\crash-prevention" (
    explorer "logs\memory\crash-prevention"
) else (
    echo No crash prevention logs found yet.
    pause
)
goto end

:end
exit /b