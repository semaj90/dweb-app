@echo off
title Enhanced Legal AI - Simple Setup
echo =====================================
echo Enhanced Legal AI - Simple Setup
echo =====================================
echo.

echo [INFO] This is a simple setup script that works without PowerShell.
echo.

echo [INFO] Checking if we're in the right directory...
if not exist "package.json" (
    echo [ERROR] package.json not found. Please run this from the sveltekit-frontend directory.
    pause
    exit /b 1
)

echo [SUCCESS] Found package.json - we're in the right directory.
echo.

echo [INFO] Installing dependencies...
call npm install
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [SUCCESS] Dependencies installed.
echo.

echo [INFO] Setting up database...
call npm run db:push
if %errorlevel% neq 0 (
    echo [WARNING] Database setup failed - continuing anyway.
) else (
    echo [SUCCESS] Database setup completed.
)
echo.

echo [INFO] Setting up AI features...
call npm run thinking:setup
if %errorlevel% neq 0 (
    echo [WARNING] AI setup failed - continuing anyway.
) else (
    echo [SUCCESS] AI features setup completed.
)
echo.

echo [INFO] Processing legal documents...
call npm run docs:process
if %errorlevel% neq 0 (
    echo [WARNING] Document processing failed - continuing anyway.
) else (
    echo [SUCCESS] Legal documents processed.
)
echo.

echo [INFO] Running tests...
call npm run thinking:test
if %errorlevel% neq 0 (
    echo [WARNING] Tests failed - continuing anyway.
) else (
    echo [SUCCESS] Tests passed.
)
echo.

echo =====================================
echo Setup completed! Starting development server...
echo =====================================
echo.
echo Your Enhanced Legal AI will be available at:
echo http://localhost:5173
echo.
echo Features available:
echo * Thinking Style AI Analysis
echo * Document Processing and OCR
echo * Evidence Analysis
echo * Legal Compliance Checking
echo * Interactive Evidence Canvas
echo.
echo Press Ctrl+C to stop the server when done.
echo.

call npm run dev

pause
