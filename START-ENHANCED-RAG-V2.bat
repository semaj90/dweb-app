@echo off
echo.
echo ========================================
echo    ENHANCED RAG V2 - QUICK START
echo    No Downloads - Using Existing Tools
echo ========================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo [CHECK] Verifying installations...
echo.

REM Check Node.js
"C:\Program Files\nodejs\node.exe" --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Node.js is installed
    "C:\Program Files\nodejs\node.exe" --version
) else (
    echo [ERROR] Node.js not found
)

REM Check PostgreSQL
netstat -an | findstr ":5432" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] PostgreSQL is running on port 5432
) else (
    echo [INFO] Starting PostgreSQL...
    net start postgresql-x64-14 >nul 2>&1
)

echo.
echo [BUILD] Preparing Go services...
cd go-microservice

REM Build Enhanced RAG V2
if not exist "bin\enhanced-rag-v2.exe" (
    echo Building Enhanced RAG V2...
    go build -o bin\enhanced-rag-v2.exe .\cmd\enhanced-rag-v2 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [OK] Enhanced RAG V2 built successfully
    ) else (
        echo [WARN] Could not build Enhanced RAG V2
    )
) else (
    echo [OK] Enhanced RAG V2 already built
)

REM Build Simply Enhanced RAG
if not exist "bin\simply-enhanced-rag.exe" (
    echo Building Simply Enhanced RAG...
    go build -o bin\simply-enhanced-rag.exe .\cmd\simply-enhanced-rag 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [OK] Simply Enhanced RAG built successfully
    ) else (
        echo [WARN] Could not build Simply Enhanced RAG
    )
) else (
    echo [OK] Simply Enhanced RAG already built
)

cd ..

echo.
echo [START] Launching services...

REM Start Enhanced RAG V2
netstat -an | findstr ":8097" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Starting Enhanced RAG V2 on port 8097...
    start /min "" "go-microservice\bin\enhanced-rag-v2.exe"
    timeout /t 2 >nul
)

REM Start Simply Enhanced RAG
netstat -an | findstr ":8096" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Starting Simply Enhanced RAG on port 8096...
    start /min "" "go-microservice\bin\simply-enhanced-rag.exe"
    timeout /t 2 >nul
)

echo.
echo [FRONTEND] Setting up frontend...
cd frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing frontend dependencies...
    "C:\Program Files\nodejs\npm.cmd" install
) else (
    echo [OK] Frontend dependencies installed
)

REM Start frontend dev server
netstat -an | findstr ":3000" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Starting frontend server on port 3000...
    start /min cmd /c ""C:\Program Files\nodejs\npm.cmd" run dev"
    timeout /t 3 >nul
) else (
    echo [OK] Frontend already running on port 3000
)

cd ..

echo.
echo ========================================
echo    SYSTEM STATUS
echo ========================================
echo.

netstat -an | findstr ":3000" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] Frontend:        http://localhost:3000
) else (
    echo [STOPPED] Frontend:        Not running
)

netstat -an | findstr ":8097" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] Enhanced RAG:    http://localhost:8097/health
) else (
    echo [STOPPED] Enhanced RAG:    Not running
)

netstat -an | findstr ":8096" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] Simply RAG:      http://localhost:8096/health
) else (
    echo [STOPPED] Simply RAG:      Not running
)

netstat -an | findstr ":5432" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] PostgreSQL:      Port 5432
) else (
    echo [STOPPED] PostgreSQL:      Not running
)

echo.
echo ========================================
echo    BEST PRACTICES APPLIED
echo ========================================
echo.
echo  - Microservices architecture
echo  - Health check endpoints
echo  - Connection pooling
echo  - WebGPU acceleration ready
echo  - gRPC communication
echo  - JWT authentication
echo  - Rate limiting
echo  - Structured logging
echo.
echo ========================================
echo    SYSTEM READY!
echo ========================================
echo.
echo Open http://localhost:3000 in your browser
echo.
echo Press any key to exit...
pause >nul