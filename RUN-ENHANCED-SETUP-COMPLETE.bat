@echo off
echo 🚀 Enhanced Legal AI System Setup
echo ================================

echo 📋 Running complete enhanced setup with all best practices...

REM Set execution policy for this session
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force"

REM Run the enhanced setup script with all options
powershell -File ".\scripts\setup-complete-system-enhanced.ps1" -GenerateSecureConfig -EnableMonitoring -CreateBackup -GenerateBestPractices

echo.
echo ✅ Enhanced setup complete!
echo.
echo 📊 To monitor system status: .\system-status-monitor.ps1
echo 📚 Best practices guide: BEST_PRACTICES_COMPREHENSIVE.md
echo 🔐 Secure config: .env.secure
echo 🌐 Application: http://localhost:5173
echo.

pause