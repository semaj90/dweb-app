@echo off
echo.
echo ==========================================
echo    ENHANCED RAG V2 - NODE.JS QUICK START
echo    No Go Required - Using Node.js Only
echo ==========================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo [STEP 1] Checking Node.js...
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    "C:\Program Files\nodejs\node.exe" --version >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PATH=C:\Program Files\nodejs;%PATH%"
        echo [OK] Node.js found
    ) else (
        echo [ERROR] Node.js not found
        pause
        exit /b 1
    )
) else (
    echo [OK] Node.js found
    node --version
)

echo.
echo [STEP 2] Installing Frontend Dependencies...
cd frontend
if not exist "node_modules" (
    echo Installing packages (this may take a few minutes)...
    call npm install
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies already installed
)
cd ..

echo.
echo [STEP 3] Starting Services...

REM Kill any existing Node.js servers on port 8084
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8084') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM Start API server (port 8084)
echo Starting API server on port 8084...
start /min cmd /c "node node-api-server.js"
timeout /t 2 >nul

REM Start frontend (port 5173)
echo Starting frontend on port 5173...
cd frontend
start /min cmd /c "npm run dev"
cd ..
timeout /t 5 >nul

echo.
echo ==========================================
echo    SYSTEM STATUS
echo ==========================================
echo.

netstat -an | findstr ":8084" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] API Server:  http://localhost:8084/api/health
) else (
    echo [STOPPED] API Server:  Not running
)

netstat -an | findstr ":5173" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [RUNNING] Frontend:    http://localhost:5173
) else (
    echo [STOPPED] Frontend:    Not running
)

echo.
echo ==========================================
echo    READY TO USE!
echo ==========================================
echo.
echo 1. Open http://localhost:5173 in your browser
echo 2. API is available at http://localhost:8084
echo.
echo This is a temporary Node.js solution.
echo For full functionality, install Go from https://go.dev
echo.
echo Press any key to exit...
pause >nul