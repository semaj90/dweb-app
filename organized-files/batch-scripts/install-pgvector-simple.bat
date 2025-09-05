@echo off
echo Installing pgvector for PostgreSQL 17 from GitHub...

REM Check if running as Administrator
net session >nul 2>&1
if errorlevel 1 (
    echo This script requires Administrator privileges.
    echo Please run Command Prompt as Administrator.
    pause
    exit /b 1
)

set "PGROOT=C:\Program Files\PostgreSQL\17"
set "PGBIN=%PGROOT%\bin"

REM Verify PostgreSQL installation
if not exist "%PGROOT%" (
    echo PostgreSQL 17 not found at %PGROOT%
    echo Please install PostgreSQL 17 first.
    pause
    exit /b 1
)

echo Found PostgreSQL 17 at %PGROOT%

REM Create temporary directory
set "TEMPDIR=%TEMP%\pgvector-build-%RANDOM%"
mkdir "%TEMPDIR%"
cd /d "%TEMPDIR%"

echo.
echo Downloading pgvector source...
git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git
if errorlevel 1 (
    echo Failed to clone pgvector repository
    goto cleanup
)

cd pgvector

echo.
echo Setting up build environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo Failed to set up Visual Studio environment
    goto cleanup
)

echo.
echo Setting PostgreSQL environment...
set "PATH=%PGBIN%;%PATH%"

echo.
echo Building pgvector...
nmake /F Makefile.win
if errorlevel 1 (
    echo Build failed
    goto cleanup
)

echo.
echo Installing pgvector...
nmake /F Makefile.win install
if errorlevel 1 (
    echo Installation failed
    goto cleanup
)

echo.
echo pgvector installation completed successfully!

REM Verify installation
if exist "%PGROOT%\share\extension\vector.control" (
    if exist "%PGROOT%\lib\vector.dll" (
        echo ✓ Extension files installed successfully
        echo   Control file: %PGROOT%\share\extension\vector.control
        echo   Library file: %PGROOT%\lib\vector.dll
    ) else (
        echo ⚠ Library file not found
    )
) else (
    echo ⚠ Control file not found
)

goto success

:cleanup
echo.
echo Installation failed. Check the error messages above.
cd /d %TEMP%
rmdir /s /q "%TEMPDIR%" 2>nul
pause
exit /b 1

:success
echo.
echo Next steps:
echo 1. Restart PostgreSQL service (recommended):
echo    net stop postgresql-x64-17 ^&^& net start postgresql-x64-17
echo 2. Connect to your database and create the extension:
echo    psql -U postgres -d legal_ai_db -c "CREATE EXTENSION vector;"
echo 3. Run your setup script: setup-postgres.sql

REM Cleanup
cd /d %TEMP%
rmdir /s /q "%TEMPDIR%" 2>nul

echo.
pause