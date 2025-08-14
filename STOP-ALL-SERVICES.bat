@echo off
echo.
echo ========================================================
echo        🛑 Legal AI System - Stop All Services
echo ========================================================
echo.

echo Stopping all Legal AI services...

:: Stop Node.js processes (SvelteKit frontend)
echo 🌐 Stopping SvelteKit Frontend...
taskkill /F /IM node.exe >nul 2>&1

:: Stop Go microservices
echo 🔬 Stopping Enhanced RAG Service...
taskkill /F /IM enhanced-rag.exe >nul 2>&1

echo 📤 Stopping Upload Service...
taskkill /F /IM upload-service.exe >nul 2>&1

:: Stop storage services
echo 🗄️ Stopping MinIO...
taskkill /F /IM minio.exe >nul 2>&1

echo 🧠 Stopping Qdrant...
taskkill /F /IM qdrant.exe >nul 2>&1

:: Stop Redis
echo 📊 Stopping Redis...
taskkill /F /IM redis-server.exe >nul 2>&1

:: Stop Ollama
echo 🤖 Stopping Ollama...
taskkill /F /IM ollama.exe >nul 2>&1

:: Stop Neo4j (if running as process)
echo 🕸️ Stopping Neo4j...
taskkill /F /IM java.exe >nul 2>&1

echo.
echo ✅ All services stopped.
echo.
echo Note: PostgreSQL service remains running (system service)
echo To stop PostgreSQL: net stop postgresql-x64-17
echo.
pause