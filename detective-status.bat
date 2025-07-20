@echo off
echo 🕵️ Detective Evidence Synthesizer - Status Check
echo.
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"
powershell -ExecutionPolicy Bypass -File manage-detective-containers.ps1 -Action status -Version both
pause