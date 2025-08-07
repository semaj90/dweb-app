@echo off
start /min redis-server
timeout /t 2 >nul
cd go-microservice
start simd-redis-vite.exe
cd ..
npm run dev