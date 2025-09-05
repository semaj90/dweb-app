@echo off
echo ========================================
echo pgvector Installation for PostgreSQL 17
echo ========================================
echo.
echo This script will copy pgvector files to PostgreSQL directories.
echo Make sure you are running this as Administrator!
echo.
pause

echo.
echo Step 1: Copying vector.dll to PostgreSQL lib directory...
copy "pgvector-install\lib\vector.dll" "C:\Program Files\PostgreSQL\17\lib\" /Y
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy vector.dll
    echo Make sure you are running as Administrator!
    pause
    exit /b 1
) else (
    echo ✅ vector.dll copied successfully
)

echo.
echo Step 2: Copying extension files to PostgreSQL share\extension directory...
copy "pgvector-install\share\extension\*" "C:\Program Files\PostgreSQL\17\share\extension\" /Y
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy extension files
    echo Make sure you are running as Administrator!
    pause
    exit /b 1
) else (
    echo ✅ Extension files copied successfully
)

echo.
echo Step 3: Verifying installation files...
dir "C:\Program Files\PostgreSQL\17\lib\vector.dll"
if %errorlevel% neq 0 (
    echo ❌ vector.dll not found in lib directory
) else (
    echo ✅ vector.dll found in lib directory
)

dir "C:\Program Files\PostgreSQL\17\share\extension\vector.control"
if %errorlevel% neq 0 (
    echo ❌ vector.control not found in extension directory
) else (
    echo ✅ vector.control found in extension directory
)

dir "C:\Program Files\PostgreSQL\17\share\extension\vector--0.8.0.sql"
if %errorlevel% neq 0 (
    echo ❌ vector--0.8.0.sql not found in extension directory
) else (
    echo ✅ vector--0.8.0.sql found in extension directory
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next step: Enable the extension in your database
echo Run this command in a regular command prompt:
echo.
echo node -e "const { Pool } = require('pg'); const pool = new Pool({connectionString: 'postgresql://postgres:123456@localhost:5432/legal_ai_db'}); pool.query('CREATE EXTENSION IF NOT EXISTS vector').then(r =^> console.log('✅ pgvector extension enabled!')).catch(e =^> console.log('❌ Error:', e.message)).finally(() =^> pool.end());"
echo.
echo Or use pgAdmin 4 and run: CREATE EXTENSION IF NOT EXISTS vector;
echo.
pause