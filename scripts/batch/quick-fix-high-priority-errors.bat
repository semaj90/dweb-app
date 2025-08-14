@echo off
echo 🔧 TypeScript Error Quick Fix - High Priority Issues
echo ===================================================
echo.

echo 📦 Installing missing Node.js and Playwright types...
npm install --save-dev @types/node @playwright/test

if %ERRORLEVEL% EQU 0 (
    echo ✅ Dependencies installed successfully
    echo.
    echo 🧪 Testing impact on error count...
    echo Running quick TypeScript check on worker-thread.js...
    
    timeout 10 npx tsc --noEmit --skipLibCheck src/lib/clustering/worker-thread.js 2>worker-errors.log
    
    echo.
    echo 📊 Checking remaining errors in worker-thread.js:
    if exist worker-errors.log (
        type worker-errors.log
    ) else (
        echo ✅ No compilation errors found!
    )
    
    echo.
    echo 🎯 Next steps:
    echo 1. Run full 'npm run check' to see improvement
    echo 2. Address remaining auto-fixable type annotations
    echo 3. Review Firebase auth integration
    
) else (
    echo ❌ Failed to install dependencies
    echo Please check your npm configuration
)

echo.
pause