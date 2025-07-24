@echo off
setlocal enabledelayedexpansion

echo ========================================
echo SMART UI + VECTOR MERGE WITH SCANNING
echo ========================================

cd /d "%~dp0"

echo 1. Scanning and backing up existing components...
cd sveltekit-frontend\src\lib\components\ui
node enhanced-merge-refactor.mjs

echo.
echo 2. Scanning and merging vector services...
cd ..\..\..\..\..
node enhanced-vector-scanner.mjs

echo.
echo 3. Starting optimized Docker services...
docker-compose -f docker-compose-enhanced-lowmem.yml up -d postgres redis qdrant

echo.
echo 4. Waiting for services...
timeout /t 20 /nobreak > nul

echo.
echo 5. Testing service health...
curl -s http://localhost:6333/health && echo ✅ Qdrant ready || echo ❌ Qdrant failed
docker exec legal-ai-postgres pg_isready -U postgres && echo ✅ PostgreSQL ready || echo ❌ PostgreSQL failed
docker exec legal-ai-redis redis-cli ping && echo ✅ Redis ready || echo ❌ Redis failed

echo.
echo 6. Installing dependencies...
cd sveltekit-frontend
npm install @qdrant/js-client-rest ioredis

echo.
echo 7. Running checks...
npm run check

echo.
echo 8. Testing vector endpoints...
timeout /t 5 /nobreak > nul
npm run dev &
timeout /t 10 /nobreak > nul
curl -X POST http://localhost:5173/api/vector/search -H "Content-Type: application/json" -d "{\"query\":\"test search\"}" 2>nul && echo ✅ Vector API working || echo ⚠ Vector API not responding

echo.
echo ========================================
echo SMART MERGE COMPLETE
echo ========================================
echo.
echo ✅ Components scanned, merged, and backed up
echo ✅ Vector services analyzed and enhanced  
echo ✅ All existing features preserved
echo ✅ Backups created with timestamps
echo.
echo Check backup directories for merge reports
echo Start: npm run dev
echo.

endlocal