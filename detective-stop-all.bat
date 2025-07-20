@echo off
echo 🕵️ Detective Evidence Synthesizer - Stop All Services
echo.
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"
powershell -ExecutionPolicy Bypass -File manage-detective-containers.ps1 -Action stop -Version both
echo.
echo ✅ All Detective services stopped
pause