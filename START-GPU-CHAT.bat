@echo off
title GPU-Accelerated Chat System - Port 5174
color 0A

echo.
echo ============================================
echo    GPU-ACCELERATED CHAT SYSTEM STARTUP
echo    Running on Port 5174
echo ============================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

echo [1/5] Checking Prerequisites...
echo --------------------------------

REM Check Node.js
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed
    pause
    exit /b 1
)
echo [OK] Node.js found

REM Check Ollama
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if %errorlevel% neq 0 (
    echo [INFO] Starting Ollama service...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
)
echo [OK] Ollama service

REM Check GPU
nvidia-smi >nul 2>&1
if %errorlevel% eq 0 (
    echo [OK] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
) else (
    echo [WARNING] No NVIDIA GPU detected - running in CPU mode
)

echo.
echo [2/5] Installing Dependencies...
echo --------------------------------
if not exist "node_modules\ws" (
    echo Installing WebSocket support...
    call npm install ws --save
)
if not exist "node_modules\ioredis" (
    echo Installing Redis client...
    call npm install ioredis --save
)
echo [OK] Dependencies installed

echo.
echo [3/5] Fixing Known Errors...
echo --------------------------------
node scripts/fix-gpu-errors.mjs
echo [OK] Error fixes applied

echo.
echo [4/5] Starting GPU Chat Server...
echo --------------------------------
echo Starting on port 5174...
echo.

REM Kill any process on port 5174
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5174') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM Start with GPU configuration
set VITE_PORT=5174
set NODE_ENV=development
set ENABLE_GPU=true
set CUDA_VISIBLE_DEVICES=0

echo Configuration:
echo   Port: 5174
echo   GPU: Enabled
echo   WebSocket: Enabled
echo   Mode: Development
echo.

REM Start the development server with GPU config
echo Starting development server...
start /B cmd /c "npm run dev -- --port 5174 --host --config vite.config.gpu.js"

timeout /t 5 /nobreak >nul

echo.
echo [5/5] Verifying Services...
echo --------------------------------

REM Check if server is running
curl -s http://localhost:5174 >nul 2>&1
if %errorlevel% eq 0 (
    echo [OK] Server is running on port 5174
) else (
    echo [WARNING] Server is still starting...
)

echo.
echo ============================================
echo    GPU CHAT SYSTEM READY
echo ============================================
echo.
echo Access Points:
echo   Main App:    http://localhost:5174
echo   GPU Chat:    http://localhost:5174/gpu-chat
echo   WebSocket:   ws://localhost:5174/ws
echo   API Health:  http://localhost:5174/api/health
echo.
echo Features:
echo   ✓ GPU Acceleration (CUDA)
echo   ✓ WebSocket Real-time
echo   ✓ Health Monitoring
echo   ✓ Typing Indicators
echo   ✓ Legal AI Models
echo.
echo Press Ctrl+C to stop the server
echo.

REM Keep the window open
cmd /k
