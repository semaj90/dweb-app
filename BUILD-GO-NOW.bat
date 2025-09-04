@echo off
echo.
echo ========================================
echo    BUILDING GO SERVICES NOW!
echo ========================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice"

echo Checking Go installation...
go version
echo.

echo Setting build environment...
set CGO_ENABLED=0
set GOOS=windows
set GOARCH=amd64

echo Creating bin directory...
if not exist "bin" mkdir bin

echo.
echo [1] Building main Legal AI Service...
go build -ldflags="-s -w" -o legal-ai-server.exe main.go
if exist "legal-ai-server.exe" (
    echo SUCCESS: legal-ai-server.exe built!
    move legal-ai-server.exe bin\ >nul 2>&1
) else (
    echo FAILED: Could not build main service
)

echo.
echo [2] Building Enhanced RAG V2...
if exist "cmd\enhanced-rag-v2\main.go" (
    go build -ldflags="-s -w" -o bin\enhanced-rag-v2.exe .\cmd\enhanced-rag-v2\main.go
    if exist "bin\enhanced-rag-v2.exe" (
        echo SUCCESS: enhanced-rag-v2.exe built!
    )
)

echo.
echo [3] Building Simply Enhanced RAG...
if exist "cmd\simply-enhanced-rag\main.go" (
    go build -ldflags="-s -w" -o bin\simply-enhanced-rag.exe .\cmd\simply-enhanced-rag\main.go
    if exist "bin\simply-enhanced-rag.exe" (
        echo SUCCESS: simply-enhanced-rag.exe built!
    )
)

echo.
echo [4] Building GPU Legal AI...
if exist "gpu-legal-ai-server.go" (
    go build -ldflags="-s -w" -o bin\gpu-legal-ai-8084.exe gpu-legal-ai-server.go
    if exist "bin\gpu-legal-ai-8084.exe" (
        echo SUCCESS: gpu-legal-ai-8084.exe built!
    )
)

echo.
echo ========================================
echo    BUILD COMPLETE - STARTING SERVICES
echo ========================================
echo.

cd ..

REM Kill any existing services
taskkill /F /IM legal-ai-server.exe >nul 2>&1
taskkill /F /IM gpu-legal-ai-8084.exe >nul 2>&1
taskkill /F /IM enhanced-rag-v2.exe >nul 2>&1

REM Start the main service
if exist "go-microservice\bin\legal-ai-server.exe" (
    echo Starting Legal AI Server on port 8084...
    start /B go-microservice\bin\legal-ai-server.exe
) else if exist "go-microservice\bin\gpu-legal-ai-8084.exe" (
    echo Starting GPU Legal AI on port 8084...
    start /B go-microservice\bin\gpu-legal-ai-8084.exe
)

timeout /t 3 >nul

REM Check if service is running
netstat -an | findstr ":8084" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo    SERVICE RUNNING SUCCESSFULLY!
    echo ========================================
    echo.
    echo API Health: http://localhost:8084/api/health
    echo.
) else (
    echo.
    echo Service may not have started. Check for errors above.
)

echo.
echo Starting frontend (if node_modules exists)...
if exist "sveltekit-frontend\node_modules" (
    cd sveltekit-frontend
    start /min cmd /c "npm run dev"
    cd ..
    echo Frontend starting on http://localhost:5173
) else (
    echo.
    echo Frontend not ready. Run:
    echo   cd sveltekit-frontend
    echo   npm install
    echo   npm run dev
)

echo.
echo Press any key to exit...
pause >nul