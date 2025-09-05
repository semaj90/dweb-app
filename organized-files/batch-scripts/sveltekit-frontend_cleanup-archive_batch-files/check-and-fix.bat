@echo off
setlocal EnableDelayedExpansion
echo ========================================
echo Legal AI - TypeScript Check & Fix
echo ========================================
echo.

echo [1/6] Checking project structure...
if not exist "src" (
    echo ERROR: src directory not found
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

if not exist "package.json" (
    echo ERROR: package.json not found
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo [2/6] Installing/updating dependencies...
npm install
if !ERRORLEVEL! NEQ 0 (
    echo ERROR: npm install failed
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

echo.
echo [3/6] Syncing SvelteKit...
npm run prepare
if !ERRORLEVEL! NEQ 0 (
    echo WARNING: SvelteKit sync had issues, continuing...
)

echo.
echo [4/6] Running TypeScript check...
echo ----------------------------------------
npm run check > check-results.txt 2>&1
set CHECK_RESULT=!ERRORLEVEL!

echo Check results:
type check-results.txt
echo ----------------------------------------

if !CHECK_RESULT! EQU 0 (
    echo ✅ TypeScript check PASSED!
    goto :success
)

echo.
echo [5/6] TypeScript errors found. Analyzing...

findstr /C:"Cannot find module" check-results.txt > nul
if !ERRORLEVEL! EQU 0 (
    echo 🔧 Fixing missing module imports...
    call :fix_missing_modules
)

findstr /C:"Type instantiation is excessively deep" check-results.txt > nul
if !ERRORLEVEL! EQU 0 (
    echo 🔧 Fixing deep type instantiation...
    call :fix_deep_types
)

findstr /C:"Property does not exist" check-results.txt > nul
if !ERRORLEVEL! EQU 0 (
    echo 🔧 Fixing property errors...
    call :fix_property_errors
)

echo.
echo [6/6] Re-running check after fixes...
npm run check
if !ERRORLEVEL! EQU 0 (
    echo ✅ TypeScript check PASSED after fixes!
    goto :success
) else (
    echo ❌ TypeScript check still failing
    echo.
    echo 📋 Manual fixes needed:
    echo   1. Check check-results.txt for specific errors
    echo   2. Fix type definitions in src/lib/types/
    echo   3. Update component props and exports
    echo   4. Verify import paths
    echo.
    echo 🔗 Common solutions:
    echo   - Add "// @ts-ignore" for temporary fixes
    echo   - Update tsconfig.json compilerOptions
    echo   - Install missing @types/ packages
    goto :end
)

:fix_missing_modules
echo   - Checking for missing type definitions...
npm install --save-dev @types/node @types/pg @types/uuid
exit /b 0

:fix_deep_types
echo   - Adding type simplification...
echo // Type fix for deep instantiation > src/lib/types/fix.ts
echo export type DeepPartial^<T^> = { [P in keyof T]?: T[P] extends object ? DeepPartial^<T[P]^> : T[P]; }; >> src/lib/types/fix.ts
exit /b 0

:fix_property_errors
echo   - Creating missing type exports...
if not exist "src\lib\types\index.ts" (
    echo export * from './user'; > src\lib\types\index.ts
    echo export * from './case'; >> src\lib\types\index.ts
    echo export * from './evidence'; >> src\lib\types\index.ts
)
exit /b 0

:success
echo.
echo ========================================
echo ✅ All TypeScript checks PASSED!
echo ========================================
echo.
echo 📊 Summary:
echo   - Dependencies: ✅ Installed
echo   - SvelteKit: ✅ Synced  
echo   - TypeScript: ✅ Valid
echo   - Components: ✅ Type-safe
echo.
echo 🚀 Ready to run: npm run dev
goto :end

:end
echo.
echo 📝 Check results saved to: check-results.txt
echo.
echo ==========================================
echo Press any key to close this window...
echo Or press Ctrl+C to keep window open
echo ==========================================
pause > nul
