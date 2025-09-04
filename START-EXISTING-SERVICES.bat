@echo off
echo.
echo ==========================================
echo    ENHANCED RAG V2 - READY TO RUN!
echo    Found existing built services
echo ==========================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo Found these services in go-microservice\bin:
echo   - enhanced-rag.exe
echo   - cluster-http.exe
echo   - summarizer-service.exe
echo   - upload-service.exe
echo   - xstate-manager.exe
echo.

REM Kill any running services
taskkill /F /IM enhanced-rag.exe >nul 2>&1
taskkill /F /IM cluster-http.exe >nul 2>&1
taskkill /F /IM summarizer-service.exe >nul 2>&1

echo Starting Enhanced RAG service...
start /B go-microservice\bin\enhanced-rag.exe
timeout /t 2 >nul

echo Starting Cluster HTTP service...
start /B go-microservice\bin\cluster-http.exe
timeout /t 2 >nul

echo Starting Summarizer service...
start /B go-microservice\bin\summarizer-service.exe
timeout /t 2 >nul

echo.
echo Checking services...
echo.

netstat -an | findstr ":8080" >nul 2>&1
if %ERRORLEVEL% equ 0 (echo [RUNNING] Port 8080 active) else (echo [STOPPED] Port 8080)

netstat -an | findstr ":8084" >nul 2>&1
if %ERRORLEVEL% equ 0 (echo [RUNNING] Port 8084 active) else (echo [STOPPED] Port 8084)

netstat -an | findstr ":8093" >nul 2>&1
if %ERRORLEVEL% equ 0 (echo [RUNNING] Port 8093 active) else (echo [STOPPED] Port 8093)

netstat -an | findstr ":8094" >nul 2>&1
if %ERRORLEVEL% equ 0 (echo [RUNNING] Port 8094 active) else (echo [STOPPED] Port 8094)

echo.
echo Starting frontend...
if exist "sveltekit-frontend\node_modules" (
    cd sveltekit-frontend
    start /min cmd /c "npm run dev"
    cd ..
    timeout /t 5 >nul
    echo Frontend starting on http://localhost:5173
) else (
    echo Frontend dependencies not installed.
    echo Run: cd sveltekit-frontend ^&^& npm install
)

echo.
echo ==========================================
echo    SYSTEM READY!
echo ==========================================
echo.
echo Services running - check these URLs:
echo   http://localhost:8080
echo   http://localhost:8084
echo   http://localhost:8093
echo   http://localhost:8094
echo   http://localhost:5173 (frontend)
echo.
pause