@echo off
echo        ╔══════════════════════════════════════════════════════════════════╗
echo        ║              📋 YoRHa Neural Log Analysis Interface             ║
echo        ╚══════════════════════════════════════════════════════════════════╝
echo.
echo        [ANALYZING] YoRHa neural network logs...
echo.

if exist monitoring\yorha\*.log (
    echo        Recent YoRHa Neural Network Activity:
    echo        ════════════════════════════════════════════════════════════════════
    for /f %%f in ('dir /b /o-d monitoring\yorha\*.log') do (
        echo.
        echo        === monitoring\yorha\%%f ===
        powershell -Command "Get-Content 'monitoring\yorha\%%f' | Select-Object -Last 20" 2>nul || (
            echo        Last 20 entries:
            more +0 "monitoring\yorha\%%f"
        )
        goto :neural_log_done
    )
    :neural_log_done
) else (
    echo        [INFO] No YoRHa neural logs found
)

if exist logs\yorha_*.log (
    echo        Additional YoRHa System Logs:
    echo        ────────────────────────────────────────────────────────────────
    for /f %%f in ('dir /b /o-d logs\yorha_*.log') do (
        echo        %%f - Last modified: 
        forfiles /M %%f /C "cmd /c echo @fdate @ftime" 2>nul
        goto :system_log_done
    )
    :system_log_done
)

echo.
echo        ════════════════════════════════════════════════════════════════════
echo        [YoRHa] Neural log analysis complete
pause