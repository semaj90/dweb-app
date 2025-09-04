@echo off
REM Simple polling collector writing snapshots to logs/perf
SET PORT=8098
SET OUTDIR=%~dp0..\logs\perf
IF NOT EXIST "%OUTDIR%" mkdir "%OUTDIR%"
:loop
echo [POLL] %date% %time% fetching metrics...
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing http://localhost:%PORT%/metrics/runtime -TimeoutSec 5).Content } catch { '{}' }"`) do (
  echo %%A > "%OUTDIR%\perf-%date:~10,4%-%date:~4,2%-%date:~7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%.json"
)
REM Sleep 30 seconds
powershell -NoProfile -Command "Start-Sleep -Seconds 30"
goto loop
