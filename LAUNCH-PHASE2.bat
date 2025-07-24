@echo off
title Prosecutor AI - Phase 2: Enhanced UI/UX with AI Foundations
color 0A

echo =====================================================
echo 🎯 PROSECUTOR AI - PHASE 2 LAUNCHER
echo 🚀 Enhanced UI/UX with AI Foundations
echo =====================================================
echo.
echo 🔥 COMPREHENSIVE INTEGRATION FEATURES:
echo    ✅ Melt UI + Bits UI v2 Integration
echo    ✅ AI Command Parsing ^& Real-time Updates
echo    ✅ XState Machine for AI Command Processing
echo    ✅ Enhanced Component System with Prop Merging
echo    ✅ shadcn-svelte + UnoCSS Integration
echo.
echo =====================================================
echo 📋 EVIDENCE SYSTEM FEATURES
echo =====================================================
echo.
echo The evidence system supports:
echo 1. 📂 Drag files from your computer onto the evidence board
echo 2. ➕ Click "ADD EVIDENCE" to open the upload dialog
echo 3. 📄 Multiple file types: PDF, images (JPG, PNG, GIF), videos (MP4, MOV, AVI), documents
echo 4. 🏷️ File metadata: Automatic file size, type, and thumbnail generation
echo 5. 📊 Evidence organization: Categorize and prioritize uploaded evidence
echo.
echo =====================================================
echo 🗺️ 7-PHASE ROADMAP
echo =====================================================
echo.
echo Phase 1: ✅ Foundation Setup (Complete)
echo Phase 2: 🔥 Enhanced UI/UX with AI foundations (Current)
echo Phase 3: 🚀 AI Core (LLM + Vector search + RAG)
echo Phase 4: 📊 Data Management (Loki.js + Redis + RabbitMQ + Neo4j)
echo Phase 5: 🤖 AI-driven UI updates in real-time
echo Phase 6: 🧠 Advanced AI (self-prompting + recommendations)
echo Phase 7: 🏭 Production optimization
echo.

REM Navigate to the correct directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo 📍 Current Directory: %CD%
echo.

REM Check for Phase 2 script first
if exist "PHASE2-PROSECUTOR-AI.ps1" (
    echo 🚀 Running Phase 2 Advanced Integration Setup...
    echo 📦 This will install Melt UI + Bits UI v2 + AI foundations
    echo.
    powershell.exe -ExecutionPolicy Bypass -File "PHASE2-PROSECUTOR-AI.ps1"
    goto END
)

REM Fallback to enhanced launcher
if exist "LAUNCH-PROSECUTOR-AI.ps1" (
    echo 🚀 Running enhanced PowerShell launcher...
    powershell.exe -ExecutionPolicy Bypass -File "LAUNCH-PROSECUTOR-AI.ps1"
    goto END
)

REM Fallback to comprehensive fix script
if exist "FIX-AND-START-PROSECUTOR-AI.ps1" (
    echo 🔧 Running comprehensive fix script...
    powershell.exe -ExecutionPolicy Bypass -File "FIX-AND-START-PROSECUTOR-AI.ps1"
    goto END
)

REM Manual fallback
echo ❌ PowerShell scripts not found!
echo 📍 Attempting manual startup...
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

if exist "package.json" (
    echo 📦 Installing dependencies...
    npm install
    echo.
    echo 🔧 Installing Phase 2 advanced dependencies...
    echo    📦 UnoCSS integration...
    npm install -D @unocss/preset-wind @unocss/preset-typography @unocss/preset-icons
    npm install -D @unocss/transformer-variant-group @unocss/svelte-scoped
    echo    🎨 UI components...
    npm install tailwind-merge bits-ui@latest clsx@latest class-variance-authority
    echo    🤖 AI and state management...
    npm install @xstate/svelte xstate
    echo.
    echo 🚀 Starting development server...
    npm run dev
) else (
    echo ❌ package.json not found! Please check your project structure.
    echo 📍 Expected location: C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend
)

:END
echo.
echo =====================================================
echo 🎉 PROSECUTOR AI PHASE 2 SESSION COMPLETED
echo =====================================================
echo.
echo 🔥 PHASE 2 FEATURES:
echo • Melt UI + Bits UI v2 with prop merging
echo • AI command parsing with parseAICommand()
echo • XState machine for AI command processing
echo • Real-time UI updates via ai-controlled classes
echo • Enhanced component system
echo • Legacy .yorha-* class support
echo.
echo 📱 Access your app at: http://localhost:5173
echo.
pause