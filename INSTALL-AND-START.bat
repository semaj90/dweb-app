@echo off
echo.
echo ==========================================
echo    ENHANCED RAG V2 - COMPLETE SETUP
echo    Installing All Dependencies & Building
echo ==========================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo [1] Installing root dependencies...
call npm install

echo.
echo [2] Installing frontend dependencies...
cd sveltekit-frontend
call npm install
cd ..

echo.
echo [3] Starting services...

REM Start frontend
echo Starting frontend on port 5173...
start /min cmd /c "npm run dev:frontend"

REM Start the Node.js API server
echo Starting API server on port 8084...
start /min cmd /c "node node-api-server.js"

timeout /t 5 >nul

echo.
echo ==========================================
echo    SYSTEM READY!
echo ==========================================
echo.
echo Frontend:  http://localhost:5173
echo API:       http://localhost:8084/api/health
echo.
echo For full functionality, install Go:
echo https://go.dev/dl/
echo.
pause