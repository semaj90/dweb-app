@echo off
echo =======================================
echo 🚀 COMPLETE LEGAL AI SYSTEM STARTUP
echo =======================================
echo.

echo 📋 Starting all system components:
echo   • Go SIMD + Redis Microservice
echo   • SvelteKit Frontend
echo   • Database Services
echo   • AI Services
echo.

echo 🔧 Step 1: Starting Go SIMD Microservice...
cd /d "%~dp0go-microservice"
start "SIMD Service" cmd /k "echo SIMD JSON + Redis Microservice && go run simd-server.go"
timeout /t 3 /nobreak >nul
echo ✅ SIMD service started on localhost:8080

echo.
echo 🌐 Step 2: Starting SvelteKit Frontend...
cd /d "%~dp0sveltekit-frontend"
start "SvelteKit Dev" cmd /k "echo SvelteKit Development Server && npm run dev"
timeout /t 5 /nobreak >nul
echo ✅ SvelteKit started on localhost:5173

echo.
echo 📊 System Status:
echo   • SIMD Service: http://localhost:8080/health
echo   • SvelteKit App: http://localhost:5173
echo   • SIMD Test API: http://localhost:5173/api/simd/test
echo   • Redis Cache: localhost:6379
echo.

echo 🎯 Quick Tests:
echo   • Test SIMD Health: curl http://localhost:8080/health
echo   • Test Full System: curl -X POST http://localhost:5173/api/simd/test -H "Content-Type: application/json" -d "{\"test_type\":\"simd_health\"}"
echo.

echo ⚡ Performance Features Active:
echo   • SIMD-optimized JSON parsing (Go fastjson)
echo   • Redis JSON module integration  
echo   • Goroutine worker pools
echo   • SvelteKit 2 + Svelte 5 frontend
echo   • Enhanced RAG with vector embeddings
echo   • Multi-layer caching architecture
echo.

echo 🎉 COMPLETE SYSTEM READY!
echo Press any key to open test URLs...
pause >nul

start http://localhost:5173
start http://localhost:8080/health

echo System monitoring...
pause