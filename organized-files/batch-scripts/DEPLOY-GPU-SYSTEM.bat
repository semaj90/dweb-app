@echo off
:: GPU System Deployment - After Auth Fix

setlocal EnableDelayedExpansion

echo 🚀 GPU Legal Processor - Complete Deployment
echo ============================================

:: Set PostgreSQL credentials
set PGPASSWORD=123456
set PGUSER=legal_admin
set PGDATABASE=legal_ai_db
set PGHOST=localhost
set PGPORT=5432

:: Test database connection first
echo Testing database connection...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U %PGUSER% -h %PGHOST% -d %PGDATABASE% -c "SELECT version();" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Database connection failed
    echo Run FIX-POSTGRES-ADMIN.bat first (as Administrator)
    pause
    exit /b 1
)

echo ✅ Database connected

:: Apply GPU schema migration
echo 📊 Applying GPU schema migration...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U %PGUSER% -h %PGHOST% -d %PGDATABASE% -f database\gpu-schema-migration.sql

if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Schema migration failed - may already be applied
)

:: Start Redis
echo 🔧 Starting Redis...
net start Redis 2>nul || (
    echo Starting Redis manually...
    start /B redis-windows\redis-server.exe redis-windows\redis.conf
)

:: Build Go services
echo 🔨 Building GPU services...
cd go-microservice

:: Install dependencies
go get -u github.com/gin-gonic/gin
go get -u github.com/redis/go-redis/v9
go get -u github.com/bytedance/sonic
go get -u github.com/tidwall/gjson
go get -u github.com/jackc/pgx/v5/pgxpool
go get -u github.com/neo4j/neo4j-go-driver/v5
go get -u github.com/fsnotify/fsnotify
go get -u github.com/minio/simdjson-go
go mod tidy

:: Build with proper CGO settings
set CGO_ENABLED=1
set CC=C:\Progra~1\LLVM\bin\clang.exe
set CXX=C:\Progra~1\LLVM\bin\clang++.exe
set CGO_CFLAGS=-IC:\Progra~1\NVIDIA~1\CUDA\v13.0\include -mavx2 -mfma
set CGO_LDFLAGS=-LC:\Progra~1\NVIDIA~1\CUDA\v13.0\lib\x64 -lcudart -lcublas

echo Building enhanced legal processor...
go build -tags=cgo -ldflags="-s -w" -o legal-processor-enhanced.exe enhanced_legal_processor.go

echo Building GPU+SIMD processor...
go build -tags=cgo -ldflags="-s -w" -o legal-processor-gpu.exe legal_processor_gpu_simd.go

echo Building auto-indexer...
go build -tags=cgo -o auto-indexer.exe auto-indexer-service.go

cd ..

:: Start Node.js services
echo 🔧 Starting Node services...
npm install --save bullmq ioredis xstate @xstate/svelte

:: Start services
echo 🚀 Launching services...

:: Start GPU processor
start "GPU Processor" /B go-microservice\legal-processor-enhanced.exe

:: Start auto-indexer
start "Auto Indexer" /B go-microservice\auto-indexer.exe

:: Start BullMQ producer
start "BullMQ" /B node backend\bullmq-producer.mjs

:: Wait for services to start
timeout /t 10 /nobreak >nul

:: Health check
echo.
echo 🔍 Running health checks...
echo ========================================

curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ GPU Service: ONLINE (port 8080)
    curl -s http://localhost:8080/health | findstr gpu_enabled
) else (
    echo ❌ GPU Service: OFFLINE
)

curl -s http://localhost:8081/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Auto-Indexer: ONLINE (port 8081)
) else (
    echo ⚠️  Auto-Indexer: May still be starting
)

redis-windows\redis-cli.exe ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Redis: ONLINE
) else (
    echo ❌ Redis: OFFLINE
)

echo.
echo 📊 System Status
echo ========================================
echo GPU Processor: http://localhost:8080
echo Auto-Indexer: http://localhost:8081
echo SvelteKit: http://localhost:5173 (run 'npm run dev' to start)
echo.
echo 🎯 Next Steps:
echo 1. Run system check: node check-system-integration.mjs
echo 2. Start frontend: npm run dev
echo 3. Test GPU acceleration: curl -X POST http://localhost:8080/similarity-search

endlocal
pause
