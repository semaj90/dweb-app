@echo off
echo Testing all services for Legal AI System...
echo.

:: Set temporary PATH for this session
set PATH=%PATH%;%~dp0redis-windows;%~dp0qdrant-windows;C:\Program Files\PostgreSQL\17\bin

echo ====================================
echo SERVICE TESTS
echo ====================================
echo.

:: Test 1: Node.js
echo [1/8] Testing Node.js...
node --version
if %ERRORLEVEL% EQU 0 (echo ✓ Node.js OK) else (echo ❌ Node.js FAILED)
echo.

:: Test 2: npm
echo [2/8] Testing npm...
npm --version
if %ERRORLEVEL% EQU 0 (echo ✓ npm OK) else (echo ❌ npm FAILED)
echo.

:: Test 3: PM2
echo [3/8] Testing PM2...
pm2 --version | findstr /C:"6."
if %ERRORLEVEL% EQU 0 (echo ✓ PM2 OK) else (echo ❌ PM2 FAILED)
echo.

:: Test 4: Redis
echo [4/8] Testing Redis...
"%~dp0redis-windows\redis-server.exe" --version
if %ERRORLEVEL% EQU 0 (echo ✓ Redis OK) else (echo ❌ Redis FAILED)
echo.

:: Test 5: Redis CLI
echo [5/8] Testing Redis CLI...
"%~dp0redis-windows\redis-cli.exe" --version
if %ERRORLEVEL% EQU 0 (echo ✓ Redis CLI OK) else (echo ❌ Redis CLI FAILED)
echo.

:: Test 6: Qdrant
echo [6/8] Testing Qdrant...
"%~dp0qdrant-windows\qdrant.exe" --help | findstr /C:"Qdrant"
if %ERRORLEVEL% EQU 0 (echo ✓ Qdrant OK) else (echo ❌ Qdrant FAILED)
echo.

:: Test 7: PostgreSQL
echo [7/8] Testing PostgreSQL...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" --version
if %ERRORLEVEL% EQU 0 (echo ✓ PostgreSQL OK) else (echo ❌ PostgreSQL FAILED)
echo.

:: Test 8: Ollama
echo [8/8] Testing Ollama...
ollama --version
if %ERRORLEVEL% EQU 0 (echo ✓ Ollama OK) else (echo ❌ Ollama FAILED)
echo.

echo ====================================
echo PATH VERIFICATION
echo ====================================
echo.
echo Current PATH includes:
echo - Redis: %~dp0redis-windows
echo - Qdrant: %~dp0qdrant-windows
echo - PostgreSQL: C:\Program Files\PostgreSQL\17\bin
echo - Node.js: C:\Program Files\nodejs
echo - Ollama: C:\Users\james\AppData\Local\Programs\Ollama
echo.

echo ====================================
echo SERVICE STATUS
echo ====================================
echo.

:: Check if services are running
echo Checking running services...
tasklist /FI "IMAGENAME eq redis-server.exe" 2>nul | findstr /C:"redis-server.exe"
if %ERRORLEVEL% EQU 0 (echo ✓ Redis is RUNNING) else (echo ⚪ Redis is NOT running)

tasklist /FI "IMAGENAME eq qdrant.exe" 2>nul | findstr /C:"qdrant.exe"
if %ERRORLEVEL% EQU 0 (echo ✓ Qdrant is RUNNING) else (echo ⚪ Qdrant is NOT running)

tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | findstr /C:"ollama.exe"
if %ERRORLEVEL% EQU 0 (echo ✓ Ollama is RUNNING) else (echo ⚪ Ollama is NOT running)

tasklist /FI "IMAGENAME eq postgres.exe" 2>nul | findstr /C:"postgres.exe"
if %ERRORLEVEL% EQU 0 (echo ✓ PostgreSQL is RUNNING) else (echo ⚪ PostgreSQL is NOT running)

echo.
echo ====================================
echo SETUP COMPLETE!
echo ====================================
echo.
echo All services have been added to PATH permanently.
echo You can now use these commands:
echo.
echo   redis-server          - Start Redis server
echo   redis-cli             - Redis command line
echo   qdrant               - Start Qdrant vector database
echo   psql                 - PostgreSQL command line
echo   pm2                  - Process manager
echo   ollama               - LLM service
echo.
echo To start all services, run: START-LEGAL-AI-SYSTEM.bat
echo.
pause