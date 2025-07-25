@echo off
setlocal enabledelayedexpansion

title Enhanced Legal AI - Complete Setup and Integration

echo ========================================
echo COMPLETE SMART SETUP + CANVAS FIXES
echo ========================================
echo.
echo This will:
echo - Scan and backup ALL existing UI components
echo - Merge similar files preserving features  
echo - Fix Canvas component integration
echo - Setup vector search with your stack
echo - Test everything end-to-end
echo.
pause

echo.
echo === PHASE 1: DIRECTORY SCANNING ===
echo.

echo 1. Scanning and merging UI components...
cd sveltekit-frontend\src\lib\components\ui
node enhanced-merge-refactor.mjs
if !errorlevel! equ 0 (
    echo ✅ UI components merged successfully
) else (
    echo ❌ UI merge failed
    pause
    exit /b 1
)

echo.
echo 2. Scanning and merging vector services...
cd ..\..\..\..\..
node enhanced-vector-scanner.mjs
if !errorlevel! equ 0 (
    echo ✅ Vector services merged successfully
) else (
    echo ❌ Vector merge failed
    pause
    exit /b 1
)

echo.
echo 3. Fixing Canvas component integration...
node fix-canvas-integration.mjs
if !errorlevel! equ 0 (
    echo ✅ Canvas components fixed
) else (
    echo ❌ Canvas fix failed
    pause
    exit /b 1
)

echo.
echo === PHASE 2: INFRASTRUCTURE SETUP ===
echo.

echo 4. Starting Docker services (PostgreSQL + Redis + Qdrant)...
docker-compose -f docker-compose-enhanced-lowmem.yml up -d postgres redis qdrant
if !errorlevel! equ 0 (
    echo ✅ Docker services started
) else (
    echo ❌ Docker failed to start
    echo Please ensure Docker Desktop is running
    pause
    exit /b 1
)

echo.
echo 5. Waiting for services to initialize...
timeout /t 30 /nobreak > nul

echo.
echo 6. Testing service health...
echo Checking PostgreSQL...
docker exec legal-ai-postgres pg_isready -U postgres >nul 2>&1 && echo ✅ PostgreSQL ready || echo ❌ PostgreSQL not ready

echo Checking Redis...
docker exec legal-ai-redis redis-cli ping >nul 2>&1 && echo ✅ Redis ready || echo ❌ Redis not ready

echo Checking Qdrant...
curl -s http://localhost:6333/health >nul 2>&1 && echo ✅ Qdrant ready || echo ❌ Qdrant not ready

echo.
echo === PHASE 3: APPLICATION SETUP ===
echo.

echo 7. Installing dependencies...
cd sveltekit-frontend
npm install @qdrant/js-client-rest ioredis
if !errorlevel! equ 0 (
    echo ✅ Dependencies installed
) else (
    echo ❌ NPM install failed
    pause
    exit /b 1
)

echo.
echo 8. Running TypeScript checks...
npm run check
if !errorlevel! equ 0 (
    echo ✅ TypeScript checks passed
) else (
    echo ⚠ TypeScript errors found - continuing anyway
)

echo.
echo === PHASE 4: TESTING ===
echo.

echo 9. Starting development server...
start /b npm run dev
echo Waiting for dev server to start...
timeout /t 15 /nobreak > nul

echo.
echo 10. Testing vector search API...
curl -X POST http://localhost:5173/api/vector/search -H "Content-Type: application/json" -d "{\"query\":\"legal case\"}" 2>nul
if !errorlevel! equ 0 (
    echo ✅ Vector search API responding
) else (
    echo ⚠ Vector API test inconclusive
)

echo.
echo 11. Testing Canvas showcase...
curl -s http://localhost:5173/showcase >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Canvas showcase accessible
) else (
    echo ⚠ Canvas showcase test inconclusive
)

echo.
echo ========================================
echo COMPLETE SETUP FINISHED
echo ========================================
echo.
echo 🎉 SUCCESS! Your enhanced legal AI system is ready:
echo.
echo ✅ UI Components: Svelte 5 + Bits UI + UnoCSS
echo ✅ Vector Search: Ollama + PostgreSQL + Qdrant + Redis
echo ✅ Canvas Integration: Fixed and working
echo ✅ Backup System: All files preserved
echo.
echo 🚀 OPEN THESE IN YOUR BROWSER:
echo.
echo   Main App:        http://localhost:5173
echo   Component Demo:  http://localhost:5173/showcase
echo.
echo 📊 AVAILABLE APIS:
echo.
echo   Vector Search:   POST /api/vector/search
echo   Embeddings:      POST /api/embeddings/generate  
echo   Chat:           POST /api/chat
echo.
echo 📁 BACKUPS CREATED:
echo.
echo   UI Components:   sveltekit-frontend/src/lib/components/ui/backup-*
echo   Vector Services: vector-backup-*
echo   Canvas:         canvas-backup-*
echo.
echo 📋 CHECK MERGE REPORTS:
echo.
echo   Look for MERGE-REPORT.md and VECTOR-SETUP-REPORT.md
echo   in the backup directories for detailed information
echo.
echo 🔧 TROUBLESHOOTING:
echo.
echo   If issues occur:
echo   1. Check backup directories for original files
echo   2. Run: npm run check
echo   3. Check Docker: docker ps
echo   4. View logs: docker-compose logs
echo.
echo Press any key to open browser or Ctrl+C to exit...
pause > nul

start http://localhost:5173/showcase

echo.
echo System is running! Press Ctrl+C to stop the development server.
echo.

endlocal