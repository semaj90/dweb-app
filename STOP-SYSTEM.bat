@echo off
chcp 65001 > nul
echo.
echo ================================================================
echo 🛑 Legal AI System Shutdown
echo ================================================================
echo.

echo 📋 Stopping services...

REM Stop PM2 processes
echo [1/6] 📦 Stopping PM2 processes...
pm2 kill >nul 2>&1
echo ✅ PM2 processes stopped

REM Stop Node.js processes (SvelteKit dev server)
echo [2/6] 📄 Stopping Node.js processes...
taskkill /F /IM node.exe >nul 2>&1
echo ✅ Node.js processes stopped

REM Stop Go server
echo [3/6] ⚡ Stopping Go server...
taskkill /F /IM legal-ai-server.exe >nul 2>&1
echo ✅ Go server stopped

REM Stop Ollama
echo [4/6] 🧠 Stopping Ollama...
taskkill /F /IM ollama.exe >nul 2>&1
echo ✅ Ollama stopped

REM Stop Qdrant
echo [5/6] 🔍 Stopping Qdrant...
taskkill /F /IM qdrant.exe >nul 2>&1
echo ✅ Qdrant stopped

REM Stop Redis
echo [6/6] 🔴 Stopping Redis...
taskkill /F /IM redis-server.exe >nul 2>&1
echo ✅ Redis stopped

echo.
echo ================================================================
echo ✅ Legal AI System Shutdown Complete!
echo ================================================================
echo.
echo All services have been stopped.
echo PostgreSQL remains running (system service).
echo.
pause