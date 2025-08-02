@echo off
title Enhanced Legal AI Verification
echo =======================================
echo Enhanced Legal AI Verification
echo =======================================
echo.

echo [INFO] This will verify all Enhanced Legal AI features mentioned in the README.
echo.

if not exist "src" (
    echo [ERROR] src directory not found. Run this from sveltekit-frontend directory.
    pause
    exit /b 1
)

echo [INFO] Running comprehensive verification...
echo.

node verification/verify-enhanced-legal-ai.mjs

echo.
echo [INFO] Verification completed!
echo.
echo Check the results above and the generated report in:
echo verification/verification-report.json
echo.
pause
