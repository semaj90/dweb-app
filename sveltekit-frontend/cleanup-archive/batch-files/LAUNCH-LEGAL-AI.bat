@echo off
cls
color 0A
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║            🚀 LEGAL AI QUICK LAUNCHER                        ║
echo ║                Database + Development Server                 ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo 📋 Checking system status...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo 🔍 1. Schema file check...
if exist "src\lib\db\schema.ts" (
    echo ✅ Database schema: READY
) else (
    echo ❌ Database schema: MISSING
    echo.
    echo 🔧 FIXING: Creating database schema...
    echo    This is a one-time setup process.
    echo.
    pause
    call FIX-DATABASE-SCHEMA.bat
    goto end
)

echo.
echo 🔍 2. Configuration check...
if exist "drizzle.config.ts" (
    echo ✅ Drizzle config: READY
) else (
    echo ❌ Drizzle config: MISSING
    echo.
    echo 🔧 Please run FIX-DATABASE-SCHEMA.bat first
    pause
    goto end
)

echo.
echo 🔍 3. Database migration check...
if exist "drizzle" (
    dir /b drizzle\*.sql >nul 2>&1
    if errorlevel 1 (
        echo ⚠️ Migrations: NEED TO GENERATE
        echo.
        echo 🔧 Generating database migrations...
        call npx drizzle-kit generate
        if errorlevel 1 (
            echo ❌ Migration generation failed
            echo Please check your database configuration
            pause
            goto end
        )
        echo ✅ Migrations generated successfully
    ) else (
        echo ✅ Migrations: READY
    )
) else (
    echo ⚠️ Migration directory: MISSING
    echo.
    echo 🔧 Generating migrations...
    call npx drizzle-kit generate
    if errorlevel 1 (
        echo ❌ Migration generation failed
        pause
        goto end
    )
)

echo.
echo 🔍 4. Database schema deployment...
echo 🔄 Applying schema to database...
call npx drizzle-kit push
if errorlevel 1 (
    echo ❌ Schema deployment failed
    echo.
    echo 🔧 Common issues:
    echo   • PostgreSQL not running
    echo   • Incorrect DATABASE_URL
    echo   • Database permissions
    echo.
    echo 💡 Try running: FIX-DATABASE-SCHEMA.bat for guided troubleshooting
    pause
    goto end
)
echo ✅ Database schema deployed successfully

echo.
echo 📊 SYSTEM STATUS: ALL READY! 
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo 🚀 LAUNCHING LEGAL AI CASE MANAGEMENT SYSTEM...
echo.
echo 📝 What's happening:
echo   ✓ Database schema is deployed
echo   ✓ Development server starting
echo   ✓ Web interface initializing
echo.

echo 🌐 Your Legal AI system will be available at:
echo    ➤ http://localhost:5173
echo.

echo 📋 Default login credentials:
echo    Email: admin@legal-ai.local
echo    Password: (set during first run)
echo.

echo 🔄 Starting development server...
start "Legal AI Development Server" cmd /k "npm run dev"

echo.
echo ✅ LAUNCH COMPLETE!
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo 🎯 Next Steps:
echo   1. 🌐 Open http://localhost:5173 in your browser
echo   2. 📝 Create your first legal case
echo   3. 📄 Upload evidence documents for AI analysis
echo   4. 👥 Test real-time collaboration features
echo.

echo 🔧 Additional Tools:
echo   • Database Browser: npm run db:studio
echo   • System Diagnostics: FIX-DATABASE-SCHEMA.bat
echo   • Advanced Controls: LEGAL-AI-MASTER-CONTROL.bat
echo.

echo 🎉 Your Legal AI Case Management System is now running!
echo    The development server is active in the background.
echo.

:end
echo 📖 Press any key to close this launcher...
pause >nul