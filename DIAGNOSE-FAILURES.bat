@echo off
REM DIAGNOSE-FAILURES.bat
REM Detailed diagnostic to find exact failures

echo ========================================
echo    DIAGNOSTIC: Finding Exact Failures
echo ========================================
echo.

echo [STEP 1] Basic Requirements Check
echo ---------------------------------
where node >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [FAIL] Node.js not installed or not in PATH
    echo       Install from: https://nodejs.org/
) else (
    echo [PASS] Node.js found
)

where go >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [FAIL] Go not installed or not in PATH
    echo       Install from: https://go.dev/
) else (
    echo [PASS] Go found
)

where npm >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [FAIL] NPM not found
) else (
    echo [PASS] NPM found
)

echo.
echo [STEP 2] Service Availability
echo ------------------------------
netstat -an | findstr :5432 >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [FAIL] PostgreSQL not listening on port 5432
    echo       Start PostgreSQL service
) else (
    echo [PASS] PostgreSQL port 5432 open
)

netstat -an | findstr :6379 >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [WARN] Redis not listening on port 6379
    echo       Redis is optional but recommended
) else (
    echo [PASS] Redis port 6379 open
)

curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [FAIL] Ollama not responding
    echo       Run: ollama serve
) else (
    echo [PASS] Ollama is running
)

echo.
echo [STEP 3] Check Frontend
echo -----------------------
if exist sveltekit-frontend\package.json (
    cd sveltekit-frontend
    
    if not exist node_modules (
        echo [FAIL] Dependencies not installed
        echo       Run: npm install
    ) else (
        echo [PASS] Dependencies installed
    )
    
    REM Check for TypeScript errors
    echo.
    echo Checking TypeScript compilation...
    npx tsc --version >nul 2>&1
    if %errorlevel% NEQ 0 (
        echo [FAIL] TypeScript not available
    ) else (
        echo [INFO] Running TypeScript check (first 10 errors)...
        npx tsc --noEmit 2>&1 | findstr /N "." | findstr "^[1-9]: ^10:"
    )
    
    cd ..
) else (
    echo [FAIL] SvelteKit frontend directory not found
)

echo.
echo [STEP 4] Check Go Server
echo ------------------------
if exist go-microservice\go.mod (
    cd go-microservice
    
    echo Testing Go compilation...
    
    REM Try to build the simple version
    go build -o test-simple.exe enhanced-server-simple.go 2>&1
    if %errorlevel% == 0 (
        echo [PASS] Simple enhanced server compiles
        del test-simple.exe
    ) else (
        echo [FAIL] Go compilation failed
        echo.
        echo Compilation errors:
        go build enhanced-server-simple.go 2>&1
    )
    
    cd ..
) else (
    echo [FAIL] Go microservice directory not found
)

echo.
echo [STEP 5] Check Store Files
echo --------------------------
if exist sveltekit-frontend\src\lib\stores (
    dir sveltekit-frontend\src\lib\stores\*.ts /B 2>nul | findstr "." >nul
    if %errorlevel% == 0 (
        echo [PASS] Store files exist
    ) else (
        echo [FAIL] No TypeScript store files found
    )
) else (
    echo [FAIL] Stores directory not found
)

echo.
echo [STEP 6] Quick API Test
echo -----------------------
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] Go server is running
    curl http://localhost:8080/health 2>nul
) else (
    echo [INFO] Go server not running
    echo       This is normal if you haven't started it yet
)

echo.
echo ========================================
echo    DIAGNOSIS COMPLETE
echo ========================================
echo.
echo Next Steps Based on Failures:
echo 1. Fix any [FAIL] items above
echo 2. For TypeScript errors: npm run check in sveltekit-frontend
echo 3. For Go errors: go mod tidy in go-microservice
echo 4. To start everything: Use simpler startup script
echo.

pause
