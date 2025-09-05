@echo off
cls
echo 🗄️ Legal AI Database Fix Script
echo ================================

:menu
echo.
echo 📋 Current Status:
echo --------------------------------
if exist "src\lib\db\schema.ts" (
    echo ✅ Schema file: EXISTS
) else (
    echo ❌ Schema file: MISSING
)

if exist "drizzle.config.ts" (
    echo ✅ Drizzle config: EXISTS
) else (
    echo ❌ Drizzle config: MISSING
)

echo.
echo 🚀 Available Actions:
echo --------------------------------
echo [1] 📋 Generate Database Migration
echo [2] 📤 Push Schema to Database 
echo [3] 🔍 Verify Database Connection
echo [4] 🌐 Start Development Server
echo [5] 📈 Open Database Studio
echo [6] 🔄 Complete Fix (Run All Steps)
echo [7] 📊 Check System Status
echo [0] ❌ Exit
echo.

set /p choice="👉 Select option [0-7]: "

if "%choice%"=="1" goto generate_migration
if "%choice%"=="2" goto push_schema
if "%choice%"=="3" goto verify_database
if "%choice%"=="4" goto start_dev
if "%choice%"=="5" goto open_studio
if "%choice%"=="6" goto complete_fix
if "%choice%"=="7" goto check_status
if "%choice%"=="0" goto exit

echo ❌ Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:generate_migration
cls
echo 📋 Step 1: Generating Database Migration...
echo ================================================
echo.
if not exist "drizzle.config.ts" (
    echo ❌ drizzle.config.ts not found!
    echo Please ensure you're in the sveltekit-frontend directory
    pause
    goto menu
)

echo ✅ Drizzle config found
echo.
echo 🔄 Running: npx drizzle-kit generate...
call npx drizzle-kit generate
if errorlevel 1 (
    echo ❌ Migration generation failed
    echo.
    echo 🔧 Troubleshooting:
    echo   • Check if schema.ts exists
    echo   • Verify DATABASE_URL in .env
    echo   • Ensure PostgreSQL is accessible
    pause
    goto menu
)

echo.
echo ✅ Migration generated successfully!
echo 📁 Check the ./drizzle folder for new migration files
echo.
pause
goto menu

:push_schema
cls
echo 📤 Step 2: Pushing Schema to Database...
echo =============================================
echo.
echo 🔄 Running: npx drizzle-kit push...
call npx drizzle-kit push
if errorlevel 1 (
    echo ❌ Schema push failed
    echo.
    echo 🔧 Troubleshooting:
    echo   • Ensure PostgreSQL is running
    echo   • Check DATABASE_URL in .env file
    echo   • Verify database exists and is accessible
    echo   • Check database permissions
    pause
    goto menu
)

echo.
echo ✅ Schema pushed to database successfully!
echo 🗄️ All tables have been created/updated
echo.
pause
goto menu

:verify_database
cls
echo 🔍 Step 3: Verifying Database Connection...
echo ===============================================
echo.
echo 🔄 Running simple database verification...
call verify-database-simple.bat
echo.
echo ✅ Database verification completed!
echo.
pause
goto menu

:start_dev
cls
echo 🌐 Step 4: Starting Development Server...
echo =========================================
echo.
echo 🚀 Launching Legal AI Case Management System...
echo 🔗 Server will be available at: http://localhost:5173
echo.
echo 📝 Note: This will start the development server in a new window.
echo        You can return to this menu by closing the dev server.
echo.
echo ✅ Starting server now...
start cmd /k "npm run dev"
echo.
echo ✅ Development server started in new window!
echo 🌐 Open http://localhost:5173 to access your Legal AI system
echo.
pause
goto menu

:open_studio
cls
echo 📈 Step 5: Opening Database Studio...
echo ====================================
echo.
echo 🔄 Launching Drizzle Studio...
echo 🔗 Studio will be available at: http://localhost:4983
echo.
echo 📝 Note: This opens a visual database browser.
echo        Perfect for viewing tables and data.
echo.
start cmd /k "npm run db:studio"
echo.
echo ✅ Database Studio started in new window!
echo 📈 Open http://localhost:4983 to view your database
echo.
pause
goto menu

:complete_fix
cls
echo 🔄 Complete Database Fix - All Steps...
echo =======================================
echo.
echo 🚀 Running complete database setup process...
echo    This will: Generate migrations, Push schema, Verify connection
echo.
echo 📋 Step 1/3: Generating migration...
call npx drizzle-kit generate
if errorlevel 1 (
    echo ❌ Migration failed - stopping process
    pause
    goto menu
)

echo ✅ Migration generated
echo.
echo 📋 Step 2/3: Pushing to database...
call npx drizzle-kit push
if errorlevel 1 (
    echo ❌ Database push failed - stopping process
    pause
    goto menu
)

echo ✅ Schema pushed successfully
echo.
echo 📋 Step 3/3: Verifying connection...
node verify-database.mjs

echo.
echo 🎉 COMPLETE FIX FINISHED!
echo ========================
echo.
echo ✅ All steps completed successfully
echo 🚀 Your Legal AI database is ready!
echo.
echo 📝 Next steps:
echo   • Start development server (option 4)
echo   • Open http://localhost:5173
echo   • Create your first legal case
echo.
pause
goto menu

:check_status
cls
echo 📊 System Status Check...
echo ========================
echo.
echo 🔍 Checking file structure...
if exist "src\lib\db\schema.ts" (
    echo ✅ Main schema file: EXISTS
    findstr /C:"export const users" src\lib\db\schema.ts >nul
    if errorlevel 1 (
        echo ❌ Users table: MISSING
    ) else (
        echo ✅ Users table: DEFINED
    )
    
    findstr /C:"export const cases" src\lib\db\schema.ts >nul
    if errorlevel 1 (
        echo ❌ Cases table: MISSING
    ) else (
        echo ✅ Cases table: DEFINED
    )
    
    findstr /C:"export const evidence" src\lib\db\schema.ts >nul
    if errorlevel 1 (
        echo ❌ Evidence table: MISSING
    ) else (
        echo ✅ Evidence table: DEFINED
    )
) else (
    echo ❌ Main schema file: MISSING
)

echo.
echo 🔍 Checking configuration...
if exist "drizzle.config.ts" (
    echo ✅ Drizzle config: EXISTS
) else (
    echo ❌ Drizzle config: MISSING
)

echo.
echo 🔍 Checking migrations...
if exist "drizzle" (
    echo ✅ Migration directory: EXISTS
    dir /b drizzle\*.sql >nul 2>&1
    if errorlevel 1 (
        echo ⚠️ Migration files: NONE (run option 1)
    ) else (
        echo ✅ Migration files: PRESENT
    )
) else (
    echo ⚠️ Migration directory: MISSING (will be created)
)

echo.
echo 🔍 Checking database connection...
node verify-database.mjs 2>nul
if errorlevel 1 (
    echo ⚠️ Database connection: NEEDS ATTENTION
) else (
    echo ✅ Database connection: WORKING
)

echo.
echo 📊 Status Summary:
echo ================
echo ℹ️ If you see any ❌ or ⚠️ above, use the menu options to fix them
echo ✅ All ✅ means your system is ready for development!
echo.
pause
goto menu

:exit
cls
echo.
echo 👋 Thank you for using the Legal AI Database Fix Script!
echo.
echo 📝 Quick Reference:
echo ===============
echo • npm run dev           - Start development server
echo • npm run db:studio     - Open database browser
echo • npm run db:generate   - Generate new migrations
echo • npm run db:push       - Apply schema changes
echo.
echo 🌐 Access your Legal AI system at: http://localhost:5173
echo.
echo 🎉 Your Legal AI Case Management System is ready!
exit /b 0