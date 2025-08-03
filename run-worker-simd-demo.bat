@echo off
echo.
echo 🚀 Running Worker Threads + SIMD + Copilot Regex Demo
echo ====================================================
echo.

echo 📋 Checking Node.js version...
node --version

echo.
echo 🔍 Checking if demo file exists...
if not exist "worker-simd-copilot-demo.mjs" (
    echo ❌ Demo file not found!
    echo Please make sure worker-simd-copilot-demo.mjs is in the current directory
    pause
    exit /b 1
)

echo ✅ Demo file found

echo.
echo 🏃 Running the demo...
echo.

node worker-simd-copilot-demo.mjs

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Demo completed successfully!
    echo.
    echo 📊 Check demo-results.json for detailed results
    if exist "demo-results.json" (
        echo 💾 Results file size:
        for %%A in (demo-results.json) do echo    %%~zA bytes
    )
) else (
    echo.
    echo ❌ Demo failed with error code: %ERRORLEVEL%
)

echo.
echo Press any key to exit...
pause >nul
