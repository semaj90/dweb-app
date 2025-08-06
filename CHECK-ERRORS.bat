@echo off
echo ========================================
echo CHECKING FOR ERRORS
echo ========================================
echo.

echo [1] Checking Node/NPM...
node --version 2>nul || echo ERROR: Node not found
npm --version 2>nul || echo ERROR: NPM not found
echo.

echo [2] Checking Go installation...
go version 2>nul || echo ERROR: Go not installed
echo.

echo [3] Checking Redis...
redis-cli ping 2>nul || echo ERROR: Redis not responding
echo.

echo [4] Checking PostgreSQL...
psql --version 2>nul || echo ERROR: PostgreSQL not found
echo.

echo [5] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% NEQ 0 (
    echo ERROR: Ollama not running
) else (
    echo Ollama is running
)
echo.

echo [6] Checking Go Server...
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% NEQ 0 (
    echo ERROR: Go server not running
) else (
    echo Go server is running
)
echo.

echo [7] Checking Frontend...
cd sveltekit-frontend 2>nul
if exist package.json (
    echo Running npm install...
    npm install 2>&1 | findstr /C:"error" /C:"ERR!"
    echo.
    echo Checking TypeScript...
    npx tsc --noEmit 2>&1 | head -20
) else (
    echo ERROR: SvelteKit frontend not found
)
cd ..
echo.

echo [8] Checking Go module...
cd go-microservice 2>nul
if exist go.mod (
    echo Checking Go dependencies...
    go mod tidy 2>&1 | findstr /C:"error"
    echo.
    echo Building Go server...
    go build -o test-server.exe legal-ai-server.go 2>&1 | findstr /C:"error" /C:"cannot"
) else (
    echo ERROR: Go module not found
)
cd ..

pause
