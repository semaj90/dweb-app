@echo off
REM PRODUCTION-STARTUP.bat - Full production system launcher

cls
echo [PRODUCTION MODE] Starting SIMD Redis Vite System...

REM Set production environment
set NODE_ENV=production
set GIN_MODE=release
set SIMD_WORKERS=32
set REDIS_MAXMEMORY=2gb

REM Check services
where redis-server >nul 2>&1 || (echo Redis not found! & exit /b 1)
where go >nul 2>&1 || (echo Go not found! & exit /b 1)
where node >nul 2>&1 || (echo Node not found! & exit /b 1)
where pm2 >nul 2>&1 || (echo Installing PM2... & npm install -g pm2 pm2-windows-startup)

REM Build production
echo Building production assets...
cd go-microservice
go build -ldflags="-s -w" -tags production -o simd-server-prod.exe simd-redis-vite-server.go
cd ..
npm run build

REM Configure Redis for production
echo Configuring Redis...
(
echo maxmemory 2gb
echo maxmemory-policy allkeys-lru
echo save 900 1
echo save 300 10
echo save 60 10000
echo appendonly yes
echo appendfsync everysec
) > redis.production.conf

REM Start Redis with production config
start "Redis-Production" /min redis-server redis.production.conf

REM Start services with PM2
echo Starting PM2 services...
pm2 kill >nul 2>&1
pm2 start ecosystem.config.js
pm2 save
pm2 startup

REM Configure Windows Firewall
echo Configuring firewall...
netsh advfirewall firewall add rule name="SIMD-Server" dir=in action=allow protocol=TCP localport=8080 >nul 2>&1
netsh advfirewall firewall add rule name="Vite-Production" dir=in action=allow protocol=TCP localport=4173 >nul 2>&1

REM Health check
timeout /t 5 /nobreak >nul
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% neq 0 (echo SIMD server failed! & pm2 logs simd-server & exit /b 1)

echo.
echo ===== PRODUCTION SYSTEM READY =====
echo SIMD Server: http://localhost:8080
echo Vite Preview: http://localhost:4173
echo PM2 Dashboard: pm2 monit
echo Logs: pm2 logs
echo ===================================
echo.
pm2 status
