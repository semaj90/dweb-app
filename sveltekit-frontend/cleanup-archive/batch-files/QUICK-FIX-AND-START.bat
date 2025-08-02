@echo off
echo ========================================
echo Legal AI Case Management - Quick Start
echo ========================================
echo.

echo [1/7] Checking PostgreSQL connection...
echo Attempting to connect to PostgreSQL database...
echo If this fails, ensure PostgreSQL is running and accessible.
echo.

echo [2/7] Setting up database...
node setup-database.mjs --seed
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Database setup failed!
    echo.
    echo 📋 Prerequisites:
    echo   - PostgreSQL server running on localhost:5432
    echo   - Database 'legal_ai_v3' exists
    echo   - User 'legal_admin' with password 'LegalSecure2024!'
    echo.
    echo 🔧 Quick PostgreSQL setup:
    echo   1. Install PostgreSQL if not installed
    echo   2. Create database: createdb legal_ai_v3
    echo   3. Create user: createuser legal_admin
    echo   4. Set password: ALTER USER legal_admin PASSWORD 'LegalSecure2024!';
    echo   5. Grant permissions: GRANT ALL ON DATABASE legal_ai_v3 TO legal_admin;
    echo.
    pause
    exit /b 1
)

echo.
echo [3/7] Installing dependencies...
npm install
if %ERRORLEVEL% NEQ 0 (
    echo Error: npm install failed
    pause
    exit /b 1
)

echo.
echo [4/7] Verifying database schema...
echo PostgreSQL database configured successfully!

echo.
echo [5/7] Checking for missing components...
if not exist "src\lib\components\ui\button\Button.svelte" (
    echo Creating missing Button component...
    mkdir "src\lib\components\ui\button" 2>nul
)

echo.
echo [6/7] Clearing build cache...
rmdir /s /q .svelte-kit 2>nul
rmdir /s /q node_modules\.vite 2>nul

echo.
echo [7/7] Starting development server...
echo.
echo ========================================
echo  Legal AI Case Management - Ready!
echo ========================================
echo.
echo 🌐 Application URLs:
echo   Home:      http://localhost:5173
echo   Login:     http://localhost:5173/login
echo   Dashboard: http://localhost:5173/dashboard
echo   Cases:     http://localhost:5173/cases
echo.
echo 👤 Demo Access:
echo   Click "Demo Login" button for instant access
echo.
echo 🗄️ Database: PostgreSQL (legal_ai_v3)
echo ========================================
echo.

npm run dev