@echo off
echo.
echo 🚀 Enhanced Legal AI System - Smart Launcher
echo ===========================================
echo.

cd /d "C:\Users\james\Desktop\web-app"

echo 📋 Choose your startup mode:
echo.
echo 1) 🔧 Auto-Fix and Start (Recommended for first run)
echo 2) 🔍 Verify System Setup (Check if ready)
echo 3) 💾 Low Memory Mode (Recommended - 8GB+ RAM)
echo 4) 🚀 Full Power Mode (12GB+ RAM recommended)
echo 5) ⚡ Quick Start Low Memory (Skip setup steps)
echo 6) 🧪 Test AI Integration (Check AI features)
echo 7) 🏥 Health Check (Service status)
echo 8) 📊 View Reports (Open latest report)
echo 9) 🛠️  Advanced Options (More choices)
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" (
    echo.
    echo 🔧 Running auto-fix and start...
    echo 📝 This will detect and fix common issues, then start the system
    powershell -ExecutionPolicy Bypass -File "fix-all-errors.ps1"
    echo.
    echo 🚀 Now starting the system...
    powershell -ExecutionPolicy Bypass -File "start-lowmem-legal-ai.ps1"
    goto end
)

if "%choice%"=="2" (
    echo.
    echo 🔍 Running system verification...
    powershell -ExecutionPolicy Bypass -File "verify-ultimate-legal-ai.ps1"
    goto end
)

if "%choice%"=="3" (
    echo.
    echo 💾 Starting Enhanced Legal AI (Low Memory Mode)...
    echo 🎯 Optimized for 8GB+ RAM systems (6GB model + overhead)
    echo ⏱️  Estimated startup time: 2-3 minutes
    echo.
    powershell -ExecutionPolicy Bypass -File "start-lowmem-legal-ai.ps1"
    goto end
)

if "%choice%"=="4" (
    echo.
    echo 🚀 Starting Enhanced Legal AI (Full Power Mode)...
    echo 🎯 Requires 12GB+ RAM for optimal performance
    echo ⏱️  Estimated startup time: 3-5 minutes
    echo.
    powershell -ExecutionPolicy Bypass -File "start-ultimate-legal-ai.ps1"
    goto end
)

if "%choice%"=="5" (
    echo.
    echo ⚡ Quick starting (Low Memory)...
    echo 🎯 Skipping model setup and some initialization
    powershell -ExecutionPolicy Bypass -File "start-lowmem-legal-ai.ps1" -QuickStart
    goto end
)

if "%choice%"=="6" (
    echo.
    echo 🧪 Running AI integration tests...
    cd sveltekit-frontend
    npm run ai:test
    cd ..
    goto end
)

if "%choice%"=="7" (
    echo.
    echo 🏥 Running comprehensive health check...
    cd sveltekit-frontend
    npm run ai:health
    cd ..
    goto end
)

if "%choice%"=="8" (
    echo.
    echo 📊 Opening available reports...
    if exist "LOW_MEMORY_STARTUP_REPORT.md" (
        echo Opening Low Memory Report...
        start notepad "LOW_MEMORY_STARTUP_REPORT.md"
    ) else if exist "ENHANCED_LEGAL_AI_STARTUP_REPORT.md" (
        echo Opening Enhanced Report...
        start notepad "ENHANCED_LEGAL_AI_STARTUP_REPORT.md"
    ) else (
        echo ❌ No reports found. Run a startup option first.
    )
    goto end
)

if "%choice%"=="8" (
    echo.
    echo 🛠️  Advanced Options:
    echo.
    echo A) Setup Missing Components
    echo B) Docker Services Only
    echo C) Database Seeding Only
    echo D) Model Setup Only
    echo E) Verbose Startup (Full logs)
    echo.
    set /p advanced="Choose advanced option (A-E): "
    
    if /i "%advanced%"=="A" (
        echo Running setup script...
        powershell -ExecutionPolicy Bypass -File "setup-enhanced-legal-ai.ps1"
    ) else if /i "%advanced%"=="B" (
        echo Starting Docker services only...
        docker compose -f docker-compose.lowmem.yml up -d
    ) else if /i "%advanced%"=="C" (
        echo Running database seeding...
        cd sveltekit-frontend
        npm run seed:lowmem
        cd ..
    ) else if /i "%advanced%"=="D" (
        echo Setting up AI models...
        docker compose -f docker-compose.lowmem.yml exec ollama /tmp/setup-models.sh
    ) else if /i "%advanced%"=="E" (
        echo Starting with verbose output...
        powershell -ExecutionPolicy Bypass -File "start-lowmem-legal-ai.ps1" -Verbose
    ) else (
        echo Invalid advanced option.
    )
    goto end
)

echo ❌ Invalid choice. Please select 1-8.

:end
echo.
echo ✅ Operation completed!
echo.
echo 💡 Quick Tips:
echo   - Use option 2 (Low Memory) for most development
echo   - First startup downloads models (may take 5-10 minutes)
echo   - Access frontend at http://localhost:5173 after startup
echo   - Sample login: prosecutor@legalai.demo
echo.
pause
