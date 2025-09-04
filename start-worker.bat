echo 🚀 Starting Evidence Processing Worker...
echo =========================================

cd workers

:: Set NODE_ENV for development
set NODE_ENV=development

:: Start the worker with proper error handling
echo Starting worker process...
echo.
echo 📋 Worker Information:
echo • Process: Evidence Processing Worker
echo • Environment: Development
echo • Log Level: Info
echo • Queue: evidence.process.queue
echo.
echo 🔧 Worker will handle:
echo • OCR: Text extraction from documents
echo • Embeddings: Vector generation for similarity search
echo • RAG: AI-powered analysis and insights
echo • WebSocket: Real-time progress updates
echo.
echo ⚠️ Press Ctrl+C to stop the worker gracefully
echo.

:: Run the worker
node evidenceProcessor.js

if %errorlevel% neq 0 (
    echo.
    echo ❌ Worker exited with error code %errorlevel%
    echo.
    echo 🔧 Troubleshooting:
    echo • Check that all services are running: ..\start-all-services.bat
    echo • Verify database setup: ..\setup-database.bat
    echo • Check .env configuration file
    echo • Review worker logs above for specific errors
    echo.
    echo 💡 Common issues:
    echo • Database connection failed - check PostgreSQL
    echo • RabbitMQ connection failed - check RabbitMQ service
    echo • Redis connection failed - check Redis service
    echo • Permission errors - try running as Administrator
    echo.
) else (
    echo.
    echo ✅ Worker stopped gracefully
)

echo.
echo 📊 Final Status Check:
echo =====================

:: Check if services are still running
echo 🔍 Service Status:
netstat -an | findstr ":6379" >nul 2>&1 && echo ✅ Redis still running || echo ⚠️ Redis stopped
netstat -an | findstr ":5432" >nul 2>&1 && echo ✅ PostgreSQL still running || echo ⚠️ PostgreSQL stopped
netstat -an | findstr ":5672" >nul 2>&1 && echo ✅ RabbitMQ still running || echo ⚠️ RabbitMQ stopped
netstat -an | findstr ":6333" >nul 2>&1 && echo ✅ Qdrant still running || echo ⚠️ Qdrant stopped

echo.
echo 💡 To restart:
echo • Worker: start-worker.bat
echo • All services: start-all-services.bat
echo • System test: test-system.bat

cd ..
pause
