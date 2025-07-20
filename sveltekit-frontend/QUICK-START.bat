@echo off
title Enhanced Legal AI - Quick Start
echo ===================================
echo Enhanced Legal AI - Quick Start
echo ===================================
echo.

REM Quick validation
if not exist "package.json" (
    echo [ERROR] Not in the right directory. Please run from sveltekit-frontend folder.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo [INFO] Node modules not found. Installing dependencies...
    npm install
)

echo [SUCCESS] Enhanced Legal AI is ready!
echo.
echo Starting development server...
echo Your app will be available at: http://localhost:5173
echo.
echo Key features available:
echo - Thinking Style AI Analysis
echo - Document Processing
echo - Evidence Analysis
echo - Legal Compliance Checking
echo.
echo Press Ctrl+C to stop the server.
echo.

npm run dev
