@echo off
:: Complete PostgreSQL + GPU System Setup

echo 🚀 Legal AI PostgreSQL + GPU Setup
echo =====================================

:: Navigate to sveltekit-frontend
cd sveltekit-frontend

:: 1. Test database connection
echo.
echo 1️⃣ Testing database connection...
node test-connection-simple.mjs

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Database connection failed
    echo Please ensure PostgreSQL is running and credentials are correct
    echo.
    echo To fix authentication:
    echo   1. Run ..\FIX-POSTGRES-ADMIN.bat as Administrator
    echo   2. Ensure password is set to: 123456
    pause
    exit /b 1
)

:: 2. Setup database schema
echo.
echo 2️⃣ Setting up database schema...
node setup-postgres-gpu.mjs --seed

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Database setup failed
    pause
    exit /b 1
)

:: 3. Run comprehensive tests
echo.
echo 3️⃣ Running comprehensive tests...
node test-postgres-drizzle.mjs

if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Some tests failed but continuing...
)

:: 4. Build GPU services
echo.
echo 4️⃣ Building GPU services...
cd ..\go-microservice

:: Install Go dependencies
go get -u github.com/gin-gonic/gin
go get -u github.com/redis/go-redis/v9
go get -u github.com/bytedance/sonic
go get -u github.com/tidwall/gjson
go get -u github.com/jackc/pgx/v5/pgxpool
go get -u github.com/fsnotify/fsnotify
go get -u github.com/minio/simdjson-go
go mod tidy

:: Build services
set CGO_ENABLED=1
set CC=C:\Progra~1\LLVM\bin\clang.exe
set CXX=C:\Progra~1\LLVM\bin\clang++.exe

echo Building GPU processor...
go build -tags=cgo -o legal-processor-gpu.exe legal_processor_gpu_simd.go

echo Building auto-indexer...
go build -tags=cgo -o auto-indexer.exe auto-indexer-service.go

cd ..

:: 5. Start services
echo.
echo 5️⃣ Starting services...

:: Start Redis
net start Redis 2>nul || (
    start /B redis-windows\redis-server.exe redis-windows\redis.conf
)

:: Start GPU processor
start "GPU Processor" /B go-microservice\legal-processor-gpu.exe

:: Start auto-indexer
start "Auto Indexer" /B go-microservice\auto-indexer.exe

:: Wait for services
timeout /t 5 /nobreak >nul

:: 6. Health check
echo.
echo 6️⃣ Running health checks...
echo =====================================

curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ GPU Service: ONLINE
) else (
    echo ❌ GPU Service: OFFLINE
)

redis-cli ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Redis: ONLINE
) else (
    echo ❌ Redis: OFFLINE
)

echo.
echo 📊 System Ready!
echo =====================================
echo PostgreSQL: legal_ai_db (legal_admin/123456)
echo GPU Service: http://localhost:8080
echo Auto-Indexer: http://localhost:8081
echo.
echo Next steps:
echo   1. Start frontend: cd sveltekit-frontend && npm run dev
echo   2. Access UI: http://localhost:5173
echo.
pause
