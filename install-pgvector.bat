@echo off
echo Installing pgvector for PostgreSQL 17...
echo.

echo Copying vector.dll to PostgreSQL lib directory...
copy "pgvector-install\lib\vector.dll" "C:\Program Files\PostgreSQL\17\lib\" /Y
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy vector.dll - make sure to run as Administrator
    pause
    exit /b 1
)

echo Copying extension files to PostgreSQL share directory...
xcopy "pgvector-install\share\extension\*" "C:\Program Files\PostgreSQL\17\share\extension\" /Y /I
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy extension files - make sure to run as Administrator
    pause
    exit /b 1
)

echo Copying include files to PostgreSQL include directory...
xcopy "pgvector-install\include\*" "C:\Program Files\PostgreSQL\17\include\" /Y /I /S
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy include files - make sure to run as Administrator
    pause
    exit /b 1
)

echo.
echo ✅ pgvector installation completed successfully!
echo.
echo Next steps:
echo 1. Connect to your database as a superuser
echo 2. Run: CREATE EXTENSION vector;
echo 3. Verify with: SELECT extname,extversion FROM pg_extension WHERE extname='vector';
echo.
pause