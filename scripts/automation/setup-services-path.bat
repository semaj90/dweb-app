@echo off
echo Setting up services and PATH for Legal AI System...

:: Get current directory
set CURRENT_DIR=%~dp0

:: Add Redis to PATH (temporarily for this session)
set PATH=%PATH%;%CURRENT_DIR%redis-windows

:: Add Qdrant to PATH (temporarily for this session)
set PATH=%PATH%;%CURRENT_DIR%qdrant-windows

:: Add PostgreSQL to PATH (if not already there)
set PATH=%PATH%;C:\Program Files\PostgreSQL\17\bin

:: Add Go to PATH (if installed)
if exist "C:\Program Files\Go\bin" (
    set PATH=%PATH%;C:\Program Files\Go\bin
)

:: Permanently add paths to user PATH environment variable
echo Adding Redis to permanent PATH...
powershell -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';%CURRENT_DIR%redis-windows', 'User')"

echo Adding Qdrant to permanent PATH...
powershell -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';%CURRENT_DIR%qdrant-windows', 'User')"

echo Adding PostgreSQL to permanent PATH...
powershell -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';C:\Program Files\PostgreSQL\17\bin', 'User')"

:: Check if Go is installed and add to PATH if needed
if exist "C:\Program Files\Go\bin" (
    echo Adding Go to permanent PATH...
    powershell -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';C:\Program Files\Go\bin', 'User')"
) else (
    echo Go not found at C:\Program Files\Go\bin - you may need to install Go
)

:: Verify PM2 is working
echo Checking PM2 installation...
pm2 --version
if %ERRORLEVEL% NEQ 0 (
    echo Installing PM2 globally...
    npm install -g pm2
)

:: Test Redis
echo Testing Redis...
if exist "%CURRENT_DIR%redis-windows\redis-server.exe" (
    echo Redis found at: %CURRENT_DIR%redis-windows\redis-server.exe
) else (
    echo ERROR: Redis not found in expected location
)

:: Test Qdrant
echo Testing Qdrant...
if exist "%CURRENT_DIR%qdrant-windows\qdrant.exe" (
    echo Qdrant found at: %CURRENT_DIR%qdrant-windows\qdrant.exe
) else (
    echo ERROR: Qdrant not found in expected location
)

:: Test PostgreSQL
echo Testing PostgreSQL...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PostgreSQL not found or not working
)

:: Test Ollama
echo Testing Ollama...
ollama --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama not found or not working
)

echo.
echo ====================================
echo Service Setup Complete!
echo ====================================
echo.
echo IMPORTANT: You may need to restart your command prompt or VS Code
echo for the PATH changes to take effect.
echo.
echo Services added to PATH:
echo - Redis: %CURRENT_DIR%redis-windows
echo - Qdrant: %CURRENT_DIR%qdrant-windows  
echo - PostgreSQL: C:\Program Files\PostgreSQL\17\bin
echo - Go: C:\Program Files\Go\bin (if installed)
echo.
echo You can now run:
echo - redis-server
echo - redis-cli
echo - qdrant
echo - psql
echo - pm2
echo - ollama
echo.
pause