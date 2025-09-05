@echo off
setlocal EnableDelayedExpansion
echo ========================================
echo Legal AI - Quick Error Diagnosis
echo ========================================
echo.

echo [STEP 1] Running npm run check...
echo ----------------------------------------
npm run check > error-diagnosis.txt 2>&1
set CHECK_EXIT_CODE=!ERRORLEVEL!

if !CHECK_EXIT_CODE! EQU 0 (
    echo ✅ No TypeScript errors found!
    goto :end
)

echo ❌ TypeScript errors detected. Analyzing...
echo.

echo [STEP 2] Error Analysis:
echo ----------------------------------------
type error-diagnosis.txt | findstr /N "error"
echo ----------------------------------------
echo.

echo [STEP 3] Common Issue Detection:
echo.

findstr /C:"Cannot find module" error-diagnosis.txt > nul
if !ERRORLEVEL! EQU 0 (
    echo 🔍 FOUND: Missing module imports
    echo   Solution: npm install missing packages
)

findstr /C:"has no exported member" error-diagnosis.txt > nul
if !ERRORLEVEL! EQU 0 (
    echo 🔍 FOUND: Incorrect import/export statements
    echo   Solution: Check export names in modules
)

findstr /C:"Property.*does not exist" error-diagnosis.txt > nul
if !ERRORLEVEL! EQU 0 (
    echo 🔍 FOUND: Missing properties on types
    echo   Solution: Update type definitions
)

findstr /C:"Type.*is not assignable" error-diagnosis.txt > nul
if !ERRORLEVEL! EQU 0 (
    echo 🔍 FOUND: Type mismatch errors
    echo   Solution: Fix type annotations
)

findstr /C:"Cannot resolve type" error-diagnosis.txt > nul
if !ERRORLEVEL! EQU 0 (
    echo 🔍 FOUND: Unresolved type references
    echo   Solution: Add type imports
)

echo.
echo [STEP 4] Quick Fixes Available:
echo   1. Run CHECK-AND-FIX.bat for automated fixes
echo   2. Check error-diagnosis.txt for full details
echo   3. Manual fixes may be needed for complex errors

:end
echo.
echo 📊 Exit Code: !CHECK_EXIT_CODE!
echo 📝 Full report: error-diagnosis.txt
echo.
echo ==========================================
echo Press any key to close this window...
echo Or press Ctrl+C to keep window open
echo ==========================================
pause > nul
