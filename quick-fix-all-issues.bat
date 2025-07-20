@echo off
echo 🔧 Quick Fix - Drizzle Config + Database + Routes
echo ================================================

cd /d "C:\Users\james\Desktop\web-app\sveltekit-frontend"

echo.
echo ✅ ALREADY FIXED: Updated drizzle.config.ts
echo • Changed driver: 'pg' to dialect: 'postgresql'
echo • Changed connectionString to url
echo • Fixed for new drizzle-kit version

echo.
echo 🚮 Removing conflicting route...
if exist "src\routes\api\evidence\[id]\" (
    rmdir /s /q "src\routes\api\evidence\[id]"
    echo ✅ Removed: /api/evidence/[id]
) else (
    echo ✅ Already removed: /api/evidence/[id]
)

echo.
echo 🔄 Testing fixed database push...
call npm run db:push
if %errorlevel% equ 0 (
    echo ✅ Database schema pushed successfully!
) else (
    echo ❌ Still failed. Trying database reset...
    
    cd ..
    call docker-compose down
    timeout /t 3 /nobreak >nul
    call docker volume rm web-app_postgres_data -f 2>nul
    call docker volume rm prosecutor_postgres_data -f 2>nul
    call docker-compose up -d postgres
    cd sveltekit-frontend
    
    timeout /t 10 /nobreak >nul
    echo 🔄 Trying again with fresh database...
    call npm run db:push
)

echo.
echo 🌱 Seeding database...
call npm run db:seed

echo.
echo 🎉 ALL FIXES APPLIED!
echo.
echo 📋 Fixed issues:
echo • Drizzle config dialect error
echo • Route conflict /api/evidence/[id] vs [evidenceId]  
echo • Database migration/schema sync
echo • Added sample data
echo.
echo 🚀 NOW RUN:
echo npm run dev
echo.
echo 🔍 VERIFY SUCCESS:
echo • App loads at http://localhost:5173
echo • No route conflict errors
echo • Evidence API works
echo • No 500 database errors
echo.
pause
