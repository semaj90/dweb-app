@echo off
echo.
echo ========================================
echo    ENHANCED RAG V2 - COMPLETE SETUP
echo    Building and Installing Everything
echo ========================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo [STEP 1] Checking installations...
echo ========================================

REM Check Node.js
where node >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Node.js found
    node --version
) else (
    "C:\Program Files\nodejs\node.exe" --version >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo [OK] Node.js found at C:\Program Files\nodejs
        set "PATH=C:\Program Files\nodejs;%PATH%"
    ) else (
        echo [ERROR] Node.js not found - please install from nodejs.org
        pause
        exit /b 1
    )
)

REM Check Go
where go >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Go found
    go version
) else (
    "C:\Program Files\Go\bin\go.exe" version >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo [OK] Go found at C:\Program Files\Go
        set "PATH=C:\Program Files\Go\bin;%PATH%"
    ) else (
        echo [ERROR] Go not found - please install from go.dev
        echo.
        echo To install Go:
        echo 1. Download from https://go.dev/dl/
        echo 2. Run the installer
        echo 3. Restart this script
        pause
        exit /b 1
    )
)

REM Check PostgreSQL
netstat -an | findstr ":5432" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] PostgreSQL is running
) else (
    echo [INFO] Starting PostgreSQL...
    net start postgresql-x64-14 >nul 2>&1
    timeout /t 2 >nul
)

echo.
echo [STEP 2] Building Go Services...
echo ========================================
cd go-microservice

REM Disable CGO for pure Go build
set CGO_ENABLED=0
set GOOS=windows
set GOARCH=amd64

REM Create bin directory
if not exist "bin" mkdir bin

REM Build Enhanced RAG V2
echo Building Enhanced RAG V2...
go build -ldflags="-s -w" -o bin\enhanced-rag-v2.exe .\cmd\enhanced-rag-v2\main.go
if exist "bin\enhanced-rag-v2.exe" (
    echo [OK] Enhanced RAG V2 built successfully
) else (
    echo [WARN] Enhanced RAG V2 build failed
)

REM Build Simply Enhanced RAG
echo Building Simply Enhanced RAG...
go build -ldflags="-s -w" -o bin\simply-enhanced-rag.exe .\cmd\simply-enhanced-rag\main.go
if exist "bin\simply-enhanced-rag.exe" (
    echo [OK] Simply Enhanced RAG built successfully
) else (
    echo [WARN] Simply Enhanced RAG build failed
)

REM Build main Legal AI service
echo Building Legal AI Service (main.go)...
go build -ldflags="-s -w" -o bin\legal-ai-service.exe main.go
if exist "bin\legal-ai-service.exe" (
    echo [OK] Legal AI Service built successfully
) else (
    echo [WARN] Legal AI Service build failed
)

cd ..

echo.
echo [STEP 3] Installing Frontend Dependencies...
echo ========================================
cd frontend

if not exist "node_modules" (
    echo Installing npm packages (this may take a few minutes)...
    call npm install
    echo [OK] Frontend dependencies installed
) else (
    echo [OK] Frontend dependencies already installed
    echo Updating packages...
    call npm update
)

cd ..

echo.
echo [STEP 4] Starting Services...
echo ========================================

REM Kill any existing services to avoid conflicts
taskkill /F /IM legal-ai-service.exe >nul 2>&1
taskkill /F /IM enhanced-rag-v2.exe >nul 2>&1
taskkill /F /IM simply-enhanced-rag.exe >nul 2>&1

REM Start Legal AI Service (8084)
if exist "go-microservice\bin\legal-ai-service.exe" (
    echo Starting Legal AI Service on port 8084...
    start /min "" "go-microservice\bin\legal-ai-service.exe"
    timeout /t 2 >nul
)

REM Start Enhanced RAG V2 (8097)
if exist "go-microservice\bin\enhanced-rag-v2.exe" (
    echo Starting Enhanced RAG V2 on port 8097...
    start /min "" "go-microservice\bin\enhanced-rag-v2.exe"
    timeout /t 2 >nul
)

REM Start Simply Enhanced RAG (8096)
if exist "go-microservice\bin\simply-enhanced-rag.exe" (
    echo Starting Simply Enhanced RAG on port 8096...
    start /min "" "go-microservice\bin\simply-enhanced-rag.exe"
    timeout /t 2 >nul
)

REM Check Ollama
netstat -an | findstr ":11434" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Ollama is running on port 11434
) else (
    echo [INFO] Starting Ollama...
    where ollama >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        start /min cmd /c "ollama serve"
        timeout /t 3 >nul
    ) else (
        echo [WARN] Ollama not found - install from ollama.ai
    )
)

REM Start Frontend
echo Starting Frontend Development Server...
cd frontend
start /min cmd /c "npm run dev"
cd ..
timeout /t 5 >nul

echo.
echo ========================================
echo    SYSTEM STATUS
echo ========================================
echo.

echo Checking services...
netstat -an | findstr ":5173" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] Frontend:        http://localhost:5173
) else (
    echo [STOPPED] Frontend:        Not running
)

netstat -an | findstr ":8084" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] Legal AI API:    http://localhost:8084/api/health
) else (
    echo [STOPPED] Legal AI API:    Not running
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

netstat -an | findstr ":11434" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] Ollama:          http://localhost:11434
) else (
    echo [STOPPED] Ollama:          Not running
)

netstat -an | findstr ":5432" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] PostgreSQL:      Port 5432
) else (
    echo [STOPPED] PostgreSQL:      Not running
)

echo.
echo ========================================
echo    SETUP COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Open http://localhost:5173 in your browser
echo 2. Check API health at http://localhost:8084/api/health
echo 3. If services aren't running, check the build output above
echo.
echo Press any key to exit...
pause >nul