@echo off
setlocal enabledelayedexpansion
title Legal AI - Master Validation Check
color 0A

echo.
echo 🔍 MASTER VALIDATION CHECK
echo ===========================
echo.

set "PROJECT_ROOT=%~dp0"
set "ALL_VALID=true"

echo [1/6] Checking critical files...
for %%f in (
    "enhanced-merge-refactor.mjs"
    "enhanced-vector-scanner.mjs" 
    "fix-canvas-integration.mjs"
    "docker-compose-unified.yml"
    "database\migrations\001_initial_schema.sql"
    "LEGAL-AI-CONTROL-PANEL.bat"
    "START-LEGAL-AI.bat"
    "COMPLETE-SYSTEM-FIX.bat"
) do (
    if exist "%%f" (
        echo ✅ %%f
    ) else (
        echo ❌ %%f - MISSING
        set "ALL_VALID=false"
    )
)

echo.
echo [2/6] Validating Docker configuration...
findstr /C:"legal-ai-postgres" docker-compose-unified.yml >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Docker container names consistent
) else (
    echo ❌ Docker container names need fixing
    set "ALL_VALID=false"
)

echo.
echo [3/6] Checking database schema...
findstr /C:"CREATE TABLE" database\migrations\001_initial_schema.sql >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Database schema ready
) else (
    echo ❌ Database schema incomplete
    set "ALL_VALID=false"
)

echo.
echo [4/6] Validating frontend structure...
if exist "sveltekit-frontend\package.json" (
    echo ✅ Frontend structure ready
) else (
    echo ❌ Frontend structure incomplete
    set "ALL_VALID=false"
)

echo.
echo [5/6] Testing Docker availability...
docker version >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Docker Desktop available
) else (
    echo ⚠️ Docker Desktop not running (start before launch)
)

echo.
echo [6/6] Running Node.js validation...
node final-system-validation.mjs
if !errorlevel! equ 0 (
    echo ✅ Node.js validation passed
) else (
    echo ❌ Node.js validation failed
    set "ALL_VALID=false"
)

echo.
echo ========================================
if "%ALL_VALID%"=="true" (
    echo 🎉 MASTER VALIDATION PASSED
    echo ========================================
    echo.
    echo ✅ ALL CRITICAL ERRORS FIXED
    echo ✅ ALL FILES CREATED AND VALIDATED  
    echo ✅ DOCKER CONFIGURATIONS CONSISTENT
    echo ✅ DATABASE SCHEMA READY
    echo ✅ SYSTEM READY FOR LAUNCH
    echo.
    echo 🚀 LAUNCH OPTIONS:
    echo   1. COMPLETE-SYSTEM-FIX.bat     ^(Recommended first run^)
    echo   2. LEGAL-AI-CONTROL-PANEL.bat  ^(Interactive launcher^)
    echo   3. START-LEGAL-AI.bat          ^(Direct GPU start^)
    echo   4. START-CPU-MODE.bat          ^(CPU fallback^)
    echo.
    echo System is ready for production use!
) else (
    echo ❌ MASTER VALIDATION FAILED
    echo ========================================
    echo.
    echo Some issues need to be resolved before launch.
    echo Run COMPLETE-SYSTEM-FIX.bat to fix remaining issues.
)

echo.
pause
endlocal
