@echo off
echo        ╔══════════════════════════════════════════════════════════════════╗
echo        ║                     🤖 YoRHa System Restart                     ║
echo        ╚══════════════════════════════════════════════════════════════════╝
echo.
echo        [YoRHa] Initiating system restart sequence...
call yorha-shutdown.bat
timeout /t 3 /nobreak >nul
call "FINAL-SETUP-AND-RUN-GPU-ENHANCED.bat"