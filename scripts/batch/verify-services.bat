@echo off
echo Verifying all services and dependencies...
echo.

:: Check Node.js and npm
echo [1/8] Checking Node.js...
node --version
if %ERRORLEVEL% NEQ 0 (echo ERROR: Node.js not found) else (echo ✓ Node.js OK)

echo [2/8] Checking npm...
npm --version
if %ERRORLEVEL% NEQ 0 (echo ERROR: npm not found) else (echo ✓ npm OK)

:: Check PM2
echo [3/8] Checking PM2...
pm2 --version
if %ERRORLEVEL% NEQ 0 (echo ERROR: PM2 not found) else (echo ✓ PM2 OK)

:: Check Redis
echo [4/8] Checking Redis...
where redis-server >nul 2>&1
if %ERRORLEVEL% NEQ 0 (echo ERROR: Redis not in PATH) else (echo ✓ Redis in PATH)

:: Check Qdrant
echo [5/8] Checking Qdrant...
where qdrant >nul 2>&1
if %ERRORLEVEL% NEQ 0 (echo ERROR: Qdrant not in PATH) else (echo ✓ Qdrant in PATH)

:: Check PostgreSQL
echo [6/8] Checking PostgreSQL...
where psql >nul 2>&1
if %ERRORLEVEL% NEQ 0 (echo ERROR: PostgreSQL not in PATH) else (echo ✓ PostgreSQL in PATH)

:: Check Ollama
echo [7/8] Checking Ollama...
ollama --version
if %ERRORLEVEL% NEQ 0 (echo ERROR: Ollama not found) else (echo ✓ Ollama OK)

:: Check Go (optional)
echo [8/8] Checking Go...
go version
if %ERRORLEVEL% NEQ 0 (echo WARNING: Go not found - needed for microservice) else (echo ✓ Go OK)

echo.
echo ====================================
echo Service Verification Complete!
echo ====================================
echo.
echo Current PATH includes:
echo %PATH%
echo.
pause