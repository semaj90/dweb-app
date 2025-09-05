@echo off
echo 🔄 Restarting Services - Clean Port Management
echo ================================================

echo 🛑 Stopping any existing services...
powershell -Command "Get-Process | Where-Object {$_.ProcessName -match 'enhanced-rag|recommendation|gpu-parser|embedder|indexer|error-processor'} | Stop-Process -Force -ErrorAction SilentlyContinue"

echo ⏳ Waiting for ports to clear...
timeout /t 3 /nobreak >nul

echo 🧹 Checking port availability...
netstat -an | findstr ":8094" && echo "⚠️ Port 8094 still in use" || echo "✅ Port 8094 available"
netstat -an | findstr ":8096" && echo "⚠️ Port 8096 still in use" || echo "✅ Port 8096 available"
netstat -an | findstr ":8097" && echo "⚠️ Port 8097 still in use" || echo "✅ Port 8097 available"

echo.
echo 🚀 Starting services on unique ports...

echo 📡 Starting Enhanced RAG V2 (Port 8097)...
start /B go run go-microservice/cmd/enhanced-rag-v2-local/main.go

echo ⏳ Waiting for Enhanced RAG to start...
timeout /t 5 /nobreak >nul

echo 🤖 Starting Recommendation Service (Port 8096)...
start /B go run go-microservice/recommendation-service.go

echo ⏳ Waiting for Recommendation Service to start...
timeout /t 3 /nobreak >nul

echo 🔧 Starting Context7 Workers...
start /B node scripts/context7-worker.js

echo 📊 Checking service health...
echo.
echo 🌐 Testing Enhanced RAG V2...
curl -s http://localhost:8097/health || echo "❌ Enhanced RAG V2 not responding"

echo.
echo 🤖 Testing Recommendation Service...
curl -s http://localhost:8096/health || echo "❌ Recommendation Service not responding"

echo.
echo 🧪 Testing comprehensive integration...
curl -s http://localhost:5173/api/comprehensive-integration || echo "❌ Integration API not responding"

echo.
echo ✅ Service restart complete!
echo 📊 Use 'npm run metrics:all' to monitor services
echo 🧪 Use 'npm run orchestrator:health' to check system health
echo 🔧 Use 'curl http://localhost:5173/api/system-integration-test' for full test

pause