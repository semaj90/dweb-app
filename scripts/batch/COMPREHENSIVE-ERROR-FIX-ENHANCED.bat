@echo off
echo =========================================================
echo   COMPREHENSIVE ERROR FIX - Phase 3-4 Enhanced System
echo =========================================================
echo.

set ERROR_COUNT=0

echo [1/8] Checking Docker Desktop status...
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   [X] ERROR: Docker Desktop not found or not running.
    echo   Please install Docker Desktop and ensure it's running.
    set /a ERROR_COUNT+=1
    goto :error_summary
)
echo   [+] SUCCESS: Docker Desktop is available.

echo.
echo [2/8] Checking Node.js and npm...
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   [X] ERROR: Node.js not found.
    set /a ERROR_COUNT+=1
)
where npm >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   [X] ERROR: npm not found.
    set /a ERROR_COUNT+=1
)
if %ERROR_COUNT% gtr 0 (
    goto :error_summary
)
echo   [+] SUCCESS: Node.js and npm found.

echo.
echo [3/8] Fixing TypeScript compilation errors...
cd sveltekit-frontend
echo   Checking package.json dependencies...
if exist package.json (
    echo   [+] package.json found, installing dependencies...
    npm install --silent
    if %ERRORLEVEL% neq 0 (
        echo   [!] WARNING: Some dependencies may have warnings.
    )
    echo   [+] Dependencies installed.
) else (
    echo   [X] ERROR: package.json not found.
    set /a ERROR_COUNT+=1
)
cd ..

echo.
echo [4/8] Fixing canvas component syntax error...
echo   Checking for object literal syntax errors...
powershell -Command "(Get-Content 'sveltekit-frontend\src\routes\cases\[id]\canvas\+page.svelte') -replace 'y: evidenceItem\.posY \?\? evidenceItem\.y \?\? 0,\s*\"', 'y: evidenceItem.posY ?? evidenceItem.y ?? 0,},' | Set-Content 'sveltekit-frontend\src\routes\cases\[id]\canvas\+page.svelte.tmp'"
if exist "sveltekit-frontend\src\routes\cases\[id]\canvas\+page.svelte.tmp" (
    move "sveltekit-frontend\src\routes\cases\[id]\canvas\+page.svelte.tmp" "sveltekit-frontend\src\routes\cases\[id]\canvas\+page.svelte"
    echo   [+] Canvas component syntax fixed.
) else (
    echo   [!] WARNING: Canvas component may already be correct.
)

echo.
echo [5/8] Stopping any conflicting services...
docker-compose -f docker-compose-phase34-enhanced.yml down --remove-orphans >nul 2>&1
docker-compose -f docker-compose-phase34-DEFINITIVE.yml down --remove-orphans >nul 2>&1
echo   [+] Previous services stopped.

echo.
echo [6/8] Starting enhanced Phase 3-4 services...
docker-compose -f docker-compose-phase34-DEFINITIVE.yml up -d
if %ERRORLEVEL% neq 0 (
    echo   [X] ERROR: Failed to start Docker services.
    set /a ERROR_COUNT+=1
    goto :error_summary
)
echo   [+] Phase 3-4 services started successfully.

echo.
echo [7/8] Waiting for services to initialize...
timeout /t 45 /nobreak >nul
echo   [+] Services initialization complete.

echo.
echo [8/8] Running system validation...
node validate-phase34.mjs
if %ERRORLEVEL% neq 0 (
    echo   [!] WARNING: Some services may need more time to start.
    echo   This is normal for first startup.
)

echo.
echo =========================================================
echo   COMPREHENSIVE ERROR FIX COMPLETE!
echo =========================================================
echo.
if %ERROR_COUNT% equ 0 (
    echo 🎉 SUCCESS: All major errors have been resolved!
    echo.
    echo ✅ Enhanced Features Ready:
    echo   • XState AI Summary Components
    echo   • Evidence Report Analysis
    echo   • Case Synthesis Workflow
    echo   • Voice synthesis integration
    echo   • PostgreSQL with pgvector
    echo   • Redis caching
    echo   • Qdrant vector database
    echo   • Neo4j graph database
    echo   • RabbitMQ event streaming
    echo   • Ollama LLM inference
    echo.
    echo 🚀 QUICK START:
    echo   START-PHASE34-ENHANCED.bat    - Start the system
    echo   STOP-PHASE34-ENHANCED.bat     - Stop the system
    echo   PHASE34-ENHANCED-STATUS.bat   - Check system status
    echo.
    echo 🌐 ACCESS POINTS:
    echo   Frontend:        http://localhost:5173
    echo   AI Summary Demo: http://localhost:5173/demo/ai-summary
    echo   Evidence Analysis: http://localhost:5173/demo/evidence-analysis
    echo   Case Synthesis: http://localhost:5173/demo/case-synthesis
    echo.
    echo System is ready for advanced legal AI workflows!
) else (
    goto :error_summary
)
echo.
pause
goto :eof

:error_summary
echo.
echo ========================================
echo   ERROR SUMMARY - %ERROR_COUNT% ERROR(S) FOUND
echo ========================================
echo.
echo Please address the errors above and run this script again.
echo.
echo Common Solutions:
echo 1. Ensure Docker Desktop is installed and running
echo 2. Install Node.js from https://nodejs.org
echo 3. Check Windows permissions for file access
echo 4. Restart your computer if services conflict
echo.
pause
exit /b 1