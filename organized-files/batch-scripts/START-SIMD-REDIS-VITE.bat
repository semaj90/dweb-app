@echo off
REM START-SIMD-REDIS-VITE.bat
REM Complete startup script for SIMD + Redis + Vite Integration System

cls
echo.
echo ================================================================
echo     SIMD JSON + Redis + Vite Integration System Launcher
echo ================================================================
echo.

REM Check for Redis
echo [1/4] Checking Redis installation...
where redis-server >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Redis not found in PATH. Checking local installation...
    if exist "redis-windows\redis-server.exe" (
        echo [+] Found local Redis installation
        set REDIS_PATH=redis-windows\
    ) else (
        echo [!] Redis not installed. Please install Redis first.
        echo     Download from: https://github.com/microsoftarchive/redis/releases
        pause
        exit /b 1
    )
) else (
    echo [+] Redis found in system PATH
    set REDIS_PATH=
)

REM Check for Go
echo [2/4] Checking Go installation...
where go >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Go not found. Please install Go first.
    echo     Download from: https://golang.org/dl/
    pause
    exit /b 1
)
echo [+] Go installation found

REM Check for Node.js
echo [3/4] Checking Node.js installation...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Node.js not found. Please install Node.js first.
    echo     Download from: https://nodejs.org/
    pause
    exit /b 1
)
echo [+] Node.js installation found

REM Check for required npm packages
echo [4/4] Checking npm packages...
if not exist "node_modules" (
    echo [!] Node modules not found. Installing dependencies...
    call npm install
    if %errorlevel% neq 0 (
        echo [!] Failed to install npm dependencies
        pause
        exit /b 1
    )
)
echo [+] NPM packages ready

echo.
echo ================================================================
echo                    Starting Services
echo ================================================================
echo.

REM Start Redis Server
echo [*] Starting Redis Server...
if defined REDIS_PATH (
    start "Redis Server" /min cmd /c "%REDIS_PATH%redis-server.exe"
) else (
    start "Redis Server" /min cmd /c "redis-server"
)
timeout /t 2 /nobreak >nul

REM Test Redis connection
echo [*] Testing Redis connection...
if defined REDIS_PATH (
    %REDIS_PATH%redis-cli ping >nul 2>&1
) else (
    redis-cli ping >nul 2>&1
)
if %errorlevel% neq 0 (
    echo [!] Redis is not responding
    pause
    exit /b 1
)
echo [+] Redis is running

REM Check for Redis JSON module
echo [*] Checking Redis JSON module...
if defined REDIS_PATH (
    %REDIS_PATH%redis-cli --raw JSON.GET test_key 2>nul | findstr /C:"ERR unknown command" >nul
) else (
    redis-cli --raw JSON.GET test_key 2>nul | findstr /C:"ERR unknown command" >nul
)
if %errorlevel% equ 0 (
    echo [!] Redis JSON module not available (optional)
    echo     The system will use standard Redis caching
) else (
    echo [+] Redis JSON module detected
)

REM Compile and start Go SIMD server
echo.
echo [*] Building Go SIMD server...
cd go-microservice

REM Check if simd-redis-vite-server.go exists
if not exist "simd-redis-vite-server.go" (
    echo [!] simd-redis-vite-server.go not found
    echo [*] Creating the file now...
    REM The file should be created by the user from the artifact
    echo     Please copy the simd-redis-vite-server.go file to go-microservice/
    cd ..
    pause
    exit /b 1
)

REM Install Go dependencies
echo [*] Installing Go dependencies...
go mod tidy
if %errorlevel% neq 0 (
    echo [!] Failed to install Go dependencies
    echo [*] Initializing go module...
    go mod init legal-ai-server
    go get github.com/gin-gonic/gin
    go get github.com/gin-contrib/cors
    go get github.com/go-redis/redis/v8
    go get github.com/valyala/fastjson
    go get github.com/gorilla/websocket
    go mod tidy
)

REM Build the server
echo [*] Compiling SIMD server...
go build -o simd-redis-vite.exe simd-redis-vite-server.go
if %errorlevel% neq 0 (
    echo [!] Failed to compile Go server
    cd ..
    pause
    exit /b 1
)

REM Start the Go server
echo [*] Starting SIMD server on port 8080...
start "SIMD Redis Vite Server" cmd /k simd-redis-vite.exe
cd ..
timeout /t 3 /nobreak >nul

REM Test server health
echo [*] Testing SIMD server health...
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] SIMD server is not responding
    echo     Waiting for server to start...
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:8080/health >nul 2>&1
    if %errorlevel% neq 0 (
        echo [!] SIMD server failed to start
        pause
        exit /b 1
    )
)
echo [+] SIMD server is running

REM Start Vite development server
echo.
echo [*] Starting Vite development server...
start "Vite Dev Server" cmd /k "npm run dev"
timeout /t 5 /nobreak >nul

echo.
echo ================================================================
echo              System Successfully Started!
echo ================================================================
echo.
echo Services Running:
echo   [+] Redis Server        : localhost:6379
echo   [+] SIMD Go Server      : http://localhost:8080
echo   [+] Vite Dev Server     : http://localhost:3130
echo.
echo Available Endpoints:
echo   - Health Check       : http://localhost:8080/health
echo   - SIMD Parse         : POST http://localhost:8080/simd-parse
echo   - Batch Processing   : POST http://localhost:8080/simd-batch
echo   - Document Process   : POST http://localhost:8080/process-document
echo   - Legal Analysis     : POST http://localhost:8080/legal/analyze
echo   - Metrics            : http://localhost:8080/metrics
echo   - WebSocket          : ws://localhost:8080/ws
echo.
echo Vite Proxy Routes:
echo   - /api/go/*          : Proxied to Go server
echo   - /api/parse         : SIMD JSON parsing
echo   - /api/llm/*         : Ollama LLM
echo   - /api/qdrant/*      : Qdrant vector DB
echo.
echo ================================================================
echo.
echo Press Ctrl+C in this window to stop monitoring
echo Close individual windows to stop specific services
echo.

REM Optional: Run tests
choice /C YN /T 10 /D N /M "Do you want to run integration tests"
if %errorlevel% equ 1 (
    echo.
    echo Running integration tests...
    echo.
    if exist "test-simd-redis-vite.mjs" (
        node test-simd-redis-vite.mjs
    ) else (
        echo [!] Test file not found: test-simd-redis-vite.mjs
        echo     Please create the test file from the provided artifact
    )
)

echo.
echo System is ready for use!
echo.
pause
