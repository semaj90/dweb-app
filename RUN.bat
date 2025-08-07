@echo off
start redis-windows\redis-server.exe
cd go-microservice
start simd-redis-vite.exe
cd ..
npm run dev