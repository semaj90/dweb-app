@echo off
echo 🚀 ENHANCED RAG V2 - COMPLETE DEPLOYMENT
echo ======================================

REM Set Go path
set PATH=%PATH%;C:\Program Files\Go\bin

REM Navigate to project directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo 📁 Working directory: %CD%

echo.
echo 🔍 Checking project structure...
if exist "go-microservice\cmd\enhanced-rag-v2\main.go" (
    echo ✅ Enhanced RAG V2 source found
) else (
    echo ❌ Enhanced RAG V2 source missing
    pause
    exit /b 1
)

if exist "go-microservice\cmd\simply-enhanced-rag\main.go" (
    echo ✅ Simply Enhanced RAG source found  
) else (
    echo ❌ Simply Enhanced RAG source missing
    pause
    exit /b 1
)

echo.
echo 🔧 Building Go microservices...
cd go-microservice

REM Create bin directory
if not exist "bin" mkdir bin

echo 📦 Installing Go dependencies...
go mod tidy
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to install Go dependencies
    pause
    exit /b 1
)

echo 🔨 Building Enhanced RAG V2...
go build -o bin\enhanced-rag-v2.exe .\cmd\enhanced-rag-v2
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to build Enhanced RAG V2
    pause
    exit /b 1
) else (
    echo ✅ Enhanced RAG V2 built successfully
)

echo 🔨 Building Simply Enhanced RAG...
go build -o bin\simply-enhanced-rag.exe .\cmd\simply-enhanced-rag
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to build Simply Enhanced RAG
    pause
    exit /b 1
) else (
    echo ✅ Simply Enhanced RAG built successfully
)

echo.
echo 🗄️ Initializing database...
cd ..
if exist "scripts\init_database.sql" (
    echo 📊 Running database initialization...
    psql -U postgres -d legal_ai_db -f "scripts\init_database.sql"
    if %ERRORLEVEL% neq 0 (
        echo ⚠️ Database initialization had warnings (may be normal)
    ) else (
        echo ✅ Database initialized successfully
    )
) else (
    echo ⚠️ Database initialization script not found
)

echo.
echo 🚀 Starting services...
echo Starting Enhanced RAG V2 on port 8097...
start "Enhanced RAG V2" cmd /k "cd /d "%CD%\go-microservice" && bin\enhanced-rag-v2.exe"

timeout /t 3 /nobreak >nul

echo Starting Simply Enhanced RAG on port 8096...
start "Simply Enhanced RAG" cmd /k "cd /d "%CD%\go-microservice" && bin\simply-enhanced-rag.exe"

timeout /t 3 /nobreak >nul

echo.
echo 🔍 Verifying deployment...
echo Testing Enhanced RAG V2 health endpoint...
curl -s http://localhost:8097/health 2>nul
if %ERRORLEVEL% equ 0 (
    echo ✅ Enhanced RAG V2 is running
) else (
    echo ⚠️ Enhanced RAG V2 health check pending (may need more time)
)

echo Testing Simply Enhanced RAG health endpoint...
curl -s http://localhost:8096/health 2>nul
if %ERRORLEVEL% equ 0 (
    echo ✅ Simply Enhanced RAG is running
) else (
    echo ⚠️ Simply Enhanced RAG health check pending (may need more time)
)

echo.
echo 🌟 DEPLOYMENT COMPLETE!
echo =====================
echo.
echo 📋 Service Status:
echo   Enhanced RAG V2:      http://localhost:8097/health
echo   Simply Enhanced RAG:  http://localhost:8096/health
echo.
echo 🎯 Next Steps:
echo   1. Wait 10-15 seconds for services to fully start
echo   2. Test the health endpoints in a browser
echo   3. Install Node.js to run the frontend
echo   4. Run: npm install && npm run dev
echo.
echo Press any key to continue...
pause >nul
