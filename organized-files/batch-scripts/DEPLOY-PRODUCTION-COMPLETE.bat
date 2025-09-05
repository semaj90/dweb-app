@echo off
:: Production Redis Windows Setup & Legal Processor Deployment

setlocal EnableDelayedExpansion

echo 🚀 Legal AI Production Deployment
echo ===================================

:: Check prerequisites
call :check_prerequisites || exit /b 1

:: Setup Redis Windows native
call :setup_redis || exit /b 1

:: Build Go microservices
call :build_go_services || exit /b 1

:: Setup Node.js services
call :setup_node_services || exit /b 1

:: Deploy SvelteKit frontend
call :deploy_frontend || exit /b 1

:: Start production services
call :start_services || exit /b 1

:: Verify deployment
call :verify_deployment || exit /b 1

echo ✅ Production deployment complete
echo 📊 Monitoring: http://localhost:8080/metrics
echo 🔍 Health: http://localhost:8080/health
echo 🌐 Frontend: http://localhost:5173

goto :eof

:check_prerequisites
echo 📋 Checking prerequisites...

:: Check CUDA
nvcc --version >nul 2>&1 || (echo ❌ CUDA required & exit /b 1)

:: Check Node.js
node --version >nul 2>&1 || (echo ❌ Node.js required & exit /b 1)

:: Check Go
go version >nul 2>&1 || (echo ❌ Go required & exit /b 1)

:: Check Git
git --version >nul 2>&1 || (echo ❌ Git required & exit /b 1)

echo ✅ Prerequisites verified
exit /b 0

:setup_redis
echo 🔧 Setting up Redis Windows...

:: Download Redis Windows if not exists
if not exist "redis-windows" (
    echo Downloading Redis Windows...
    curl -L -o redis-windows.zip https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip
    powershell -Command "Expand-Archive -Path redis-windows.zip -DestinationPath redis-windows -Force"
    del redis-windows.zip
)

:: Configure Redis
echo Creating Redis configuration...
(
echo port 6379
echo bind 127.0.0.1
echo maxmemory 1gb
echo maxmemory-policy allkeys-lru
echo save 900 1
echo save 300 10
echo save 60 10000
echo dir ./redis-data
echo logfile redis.log
echo databases 16
echo timeout 300
echo tcp-keepalive 300
) > redis-windows\redis.conf

:: Create Redis data directory
mkdir redis-windows\redis-data 2>nul

:: Install Redis as Windows service
echo Installing Redis service...
sc create "Redis" binPath= "%cd%\redis-windows\redis-server.exe %cd%\redis-windows\redis.conf" start= auto
sc description "Redis" "Redis in-memory data store for Legal AI"

:: Start Redis
net start Redis || (
    echo Warning: Service installation failed, starting manually...
    start /B redis-windows\redis-server.exe redis-windows\redis.conf
)

echo ✅ Redis setup complete
exit /b 0

:build_go_services
echo 🔨 Building Go microservices...

cd go-microservice

:: Set environment
set CGO_ENABLED=1
set CC=clang
set CXX=clang++
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CGO_CFLAGS=-I"%CUDA_PATH%\include" -mavx2 -mfma"
set "CGO_LDFLAGS=-L"%CUDA_PATH%\lib\x64" -lcudart -lcublas"

:: Update dependencies
go mod tidy

:: Build enhanced processor
go build -tags=cgo -ldflags="-s -w" -o legal-processor-enhanced.exe enhanced_legal_processor.go

:: Build GPU-SIMD processor
go build -tags=cgo -ldflags="-s -w" -o legal-processor-gpu.exe legal_processor_gpu_simd.go

if not exist legal-processor-enhanced.exe (
    echo ❌ Enhanced processor build failed
    exit /b 1
)

if not exist legal-processor-gpu.exe (
    echo ❌ GPU processor build failed
    exit /b 1
)

cd ..
echo ✅ Go services built
exit /b 0

:setup_node_services
echo 🔧 Setting up Node.js services...

:: Install production dependencies
npm install --production
npm install bullmq ioredis xstate @xstate/svelte

:: Build BullMQ producer
if not exist backend mkdir backend
copy /Y bullmq-producer.mjs backend\ 2>nul

:: Setup PM2 for process management
npm install -g pm2

:: Create PM2 ecosystem
(
echo module.exports = {
echo   apps: [
echo     {
echo       name: "bullmq-producer",
echo       script: "./backend/bullmq-producer.mjs",
echo       instances: 2,
echo       exec_mode: "cluster",
echo       env: {
echo         NODE_ENV: "production",
echo         REDIS_HOST: "localhost",
echo         REDIS_PORT: 6379
echo       },
echo       error_file: "./logs/bullmq-error.log",
echo       out_file: "./logs/bullmq-out.log",
echo       log_file: "./logs/bullmq-combined.log"
echo     }
echo   ]
echo };
) > ecosystem.config.js

mkdir logs 2>nul

echo ✅ Node.js services configured
exit /b 0

:deploy_frontend
echo 🌐 Deploying SvelteKit frontend...

:: Build SvelteKit
npm run build

:: Create production adapter config if needed
if not exist svelte.config.js.backup (
    copy svelte.config.js svelte.config.js.backup
)

echo ✅ Frontend built
exit /b 0

:start_services
echo 🚀 Starting production services...

:: Start Redis (if not running)
net start Redis 2>nul || echo Redis already running

:: Start Go microservices
echo Starting enhanced legal processor...
start /B go-microservice\legal-processor-enhanced.exe

:: Wait for service to start
timeout /t 5 /nobreak

:: Start BullMQ producer
echo Starting BullMQ services...
pm2 start ecosystem.config.js

:: Start SvelteKit (production)
echo Starting SvelteKit frontend...
start /B npm run preview

echo ✅ All services started
exit /b 0

:verify_deployment
echo 🔍 Verifying deployment...

:: Check Go service health
timeout /t 3 /nobreak >nul
curl -s http://localhost:8080/health | findstr "gpu_enabled" >nul && (
    echo ✅ Go microservice healthy
) || (
    echo ❌ Go microservice health check failed
    exit /b 1
)

:: Check Redis
redis-windows\redis-cli.exe ping >nul 2>&1 && (
    echo ✅ Redis responsive
) || (
    echo ❌ Redis not responding
    exit /b 1
)

:: Check BullMQ
pm2 list | findstr "bullmq-producer" | findstr "online" >nul && (
    echo ✅ BullMQ services running
) || (
    echo ❌ BullMQ services not running
    exit /b 1
)

:: Check SvelteKit
curl -s http://localhost:4173 >nul 2>&1 && (
    echo ✅ SvelteKit frontend serving
) || (
    echo ⚠️  SvelteKit check failed - may still be starting
)

echo ✅ Deployment verification complete
exit /b 0

endlocal