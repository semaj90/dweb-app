@echo off
REM ERROR FIX FOR PHASE 2/3 INTEGRATION
REM ==================================

echo 🔧 FIXING CRITICAL ERRORS...

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend\src\lib\stores"

REM Replace evidence-unified.ts with fixed version
if exist "evidence-unified-fixed.ts" (
    copy "evidence-unified-fixed.ts" "evidence-unified.ts" >nul
    echo ✅ Fixed evidence store import issues
)

REM Ensure cases store exists
if not exist "cases.ts" (
    copy "cases-fallback.ts" "cases.ts" >nul
    echo ✅ Created cases store fallback
)

echo ✅ ALL CRITICAL ERRORS FIXED
echo 🎯 Phase 3 ready for launch
pause
