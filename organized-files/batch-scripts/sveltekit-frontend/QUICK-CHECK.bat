@echo off
title Quick Installation and Error Check
color 0A

echo.
echo ====================================
echo    QUICK INSTALL AND ERROR CHECK
echo ====================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

echo [1/6] Checking Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Node.js not found!
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do echo [OK] Node.js: %%i

echo.
echo [2/6] Installing dependencies...
echo      This may take a few minutes...
call npm install --silent 2>install-errors.log
if %errorlevel% neq 0 (
    echo [!] Some packages had warnings/errors (see install-errors.log)
) else (
    echo [OK] Dependencies installed
)

echo.
echo [3/6] Checking TypeScript compilation...
call npx tsc --noEmit --skipLibCheck 2>typescript-errors.log
if %errorlevel% neq 0 (
    echo [!] TypeScript errors found
    echo.
    echo First 5 errors:
    type typescript-errors.log | findstr /i "error" | more +0 /e +5
    echo.
    echo Full errors saved to: typescript-errors.log
) else (
    echo [OK] TypeScript compilation successful
)

echo.
echo [4/6] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama not running
    echo     Run: ollama serve
) else (
    echo [OK] Ollama is running
    
    echo.
    echo Available models:
    for /f "tokens=*" %%i in ('ollama list 2^>nul') do (
        echo     %%i | findstr /i "gemma llama nomic" >nul
        if not errorlevel 1 echo     %%i
    )
)

echo.
echo [5/6] Checking required services...
netstat -an | findstr ":5432.*LISTENING" >nul 2>&1
if %errorlevel% eq 0 (
    echo [OK] PostgreSQL (port 5432)
) else (
    echo [!] PostgreSQL not running
)

netstat -an | findstr ":6379.*LISTENING" >nul 2>&1
if %errorlevel% eq 0 (
    echo [OK] Redis (port 6379)
) else (
    echo [!] Redis not running
)

netstat -an | findstr ":5173.*LISTENING" >nul 2>&1
if %errorlevel% eq 0 (
    echo [OK] Dev Server (port 5173)
) else (
    echo [!] Dev Server not running
)

echo.
echo [6/6] Checking AI configuration files...
set AI_FILES_OK=1

if exist "src\lib\server\ai\ollama-config.ts" (
    echo [OK] ollama-config.ts
) else (
    echo [X] Missing: ollama-config.ts
    set AI_FILES_OK=0
)

if exist "src\lib\server\ai\ollama-service.ts" (
    echo [OK] ollama-service.ts
) else (
    echo [X] Missing: ollama-service.ts
    set AI_FILES_OK=0
)

if exist "src\lib\server\ai\types.ts" (
    echo [OK] types.ts
) else (
    echo [X] Missing: types.ts
    set AI_FILES_OK=0
)

echo.
echo ====================================
echo    CHECK COMPLETE
echo ====================================
echo.

REM Count issues
set /a ISSUES=0

if exist typescript-errors.log (
    for /f %%a in ('type typescript-errors.log ^| find /c "error"') do set /a ISSUES=ISSUES+%%a
)

if %ISSUES% gtr 0 (
    echo Status: %ISSUES% TypeScript errors found
    echo.
    echo To fix TypeScript errors:
    echo   1. Review: typescript-errors.log
    echo   2. Run: npx tsc --noEmit --skipLibCheck
    echo.
) else (
    echo Status: No TypeScript errors
)

if %AI_FILES_OK% equ 0 (
    echo.
    echo Some AI files are missing!
    echo Run the AI setup again if needed.
)

echo.
echo Quick Actions:
echo   - Start dev server: npm run dev
echo   - Start Ollama: ollama serve
echo   - Run tests: npm test
echo   - Full check: VERIFY-SYSTEM.bat
echo.

REM Clean up temp files
if exist install-errors.log del /q install-errors.log 2>nul
if exist typescript-errors.log (
    echo TypeScript errors saved to: typescript-errors.log
)

pause
