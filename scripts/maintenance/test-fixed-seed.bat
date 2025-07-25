@echo off
echo 🔧 Testing Fixed seed.ts File
echo ============================

cd /d "C:\Users\james\Desktop\web-app\sveltekit-frontend"

echo.
echo ✅ FIXED: Replaced seed.ts with corrected version
echo • Removed syntax error at line 569
echo • Fixed malformed function calls
echo • Simplified data structures
echo • Added proper TypeScript casting

echo.
echo 🔄 Testing the fix...

echo.
echo 🌱 Attempting database seeding...
call npm run db:seed

if %errorlevel% equ 0 (
    echo.
    echo 🎉 SUCCESS! Database seeding completed!
    echo.
    echo ✅ What was seeded:
    echo • 3 users (admin, prosecutor, detective)
    echo • 2 sample cases 
    echo • 1 sample evidence item
    echo.
    echo 👥 Login credentials:
    echo • admin@example.com / password123
    echo • prosecutor@example.com / password123
    echo • detective@example.com / password123
    echo.
    echo 🚀 Ready to start your app:
    echo npm run dev
) else (
    echo.
    echo ❌ Seeding failed. Common fixes:
    echo 1. Make sure database is running: docker-compose up -d
    echo 2. Run schema sync first: npm run db:push
    echo 3. Check database connection in .env file
)

echo.
pause
