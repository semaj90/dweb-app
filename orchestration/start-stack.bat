@echo off
setlocal enabledelayedexpansion

echo 🚀 Starting Legal AI Native Windows Stack...
echo.

REM Check prerequisites
echo Checking prerequisites...
where /q go || (echo ❌ Go not found! Install from https://golang.org && exit /b 1)
where /q rustc || (echo ❌ Rust not found! Install from https://rustup.rs && exit /b 1)
where /q node || (echo ❌ Node.js not found! Install from https://nodejs.org && exit /b 1)
where /q npm || (echo ❌ npm not found! && exit /b 1)

REM Check if PM2 is installed
npm list -g pm2 >nul 2>&1 || (
    echo 📦 Installing PM2 globally...
    npm install -g pm2
)

REM Start PostgreSQL if not running
echo 📊 Starting PostgreSQL...
sc query "postgresql-x64-17" | findstr "RUNNING" >nul || (
    net start postgresql-x64-17 || echo ⚠️  PostgreSQL service not found - install PostgreSQL 17 with pgvector
)

REM Start Ollama if not running
echo 🤖 Starting Ollama...
tasklist | findstr "ollama.exe" >nul || (
    start /B ollama serve
    timeout /t 3 /nobreak >nul
)

REM Ensure Ollama models are available
echo 📥 Checking Ollama models...
ollama list | findstr "nomic-embed-text" >nul || (
    echo Pulling nomic-embed-text model...
    ollama pull nomic-embed-text
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "models" mkdir models
if not exist "documents" mkdir documents
if not exist "logs" mkdir logs
if not exist "certs" mkdir certs

REM Generate self-signed certificates for QUIC
if not exist "certs\server.crt" (
    echo 🔐 Generating self-signed certificates for QUIC...
    openssl req -x509 -newkey rsa:4096 -keyout certs\server.key -out certs\server.crt -days 365 -nodes -subj "/C=US/ST=State/L=City/O=LegalAI/CN=localhost" 2>nul || (
        echo ⚠️  OpenSSL not found - using dummy certificates
        echo dummy > certs\server.crt
        echo dummy > certs\server.key
    )
)

REM Build Go services if executables don't exist
echo 🔨 Building Go services...

if not exist "go-services\legal-bert-onnx\legal-bert-service.exe" (
    echo Building Legal-BERT ONNX service...
    if exist "go-services\legal-bert-onnx" (
        cd go-services\legal-bert-onnx
        go mod tidy
        go build -ldflags="-s -w" -o legal-bert-service.exe .
        cd ..\..
    ) else (
        echo ⚠️  go-services\legal-bert-onnx directory not found
    )
)

if not exist "go-services\simd-operations\simd-service.exe" (
    echo Building SIMD operations service...
    if exist "go-services\simd-operations" (
        cd go-services\simd-operations
        go mod tidy
        go build -ldflags="-s -w" -tags=avx512 -o simd-service.exe .
        cd ..\..
    ) else (
        echo ⚠️  go-services\simd-operations directory not found
    )
)

if not exist "go-services\quic-server\quic-service.exe" (
    echo Building QUIC coordination service...
    if exist "go-services\quic-server" (
        cd go-services\quic-server
        go mod tidy
        go build -ldflags="-s -w" -o quic-service.exe .
        cd ..\..
    ) else (
        echo ⚠️  go-services\quic-server directory not found
    )
)

REM Build Rust services if executables don't exist
echo 🦀 Building Rust services...

if not exist "rust-services\qdrant-vector\target\release\qdrant-service.exe" (
    echo Building Qdrant vector service...
    if exist "rust-services\qdrant-vector" (
        cd rust-services\qdrant-vector
        cargo build --release
        cd ..\..
    ) else (
        echo ⚠️  rust-services\qdrant-vector directory not found
    )
)

REM Build WASM bridge
if exist "rust-services\webasm-bridge" (
    echo Building WASM filesystem bridge...
    cd rust-services\webasm-bridge
    wasm-pack build --target web --out-dir ../../sveltekit-frontend/src/lib/webasm/pkg 2>nul || echo ⚠️  wasm-pack not found
    cd ..\..
)

REM Install SvelteKit dependencies
if exist "sveltekit-frontend\package.json" (
    echo 📦 Installing SvelteKit dependencies...
    cd sveltekit-frontend
    npm install
    cd ..
) else (
    echo ⚠️  sveltekit-frontend directory not found
)

REM Install MCP dependencies
if exist "mcp\package.json" (
    echo 📦 Installing MCP dependencies...
    cd mcp
    npm install
    cd ..
)

REM Start services with PM2
echo 🚀 Starting services with PM2...
pm2 delete all 2>nul
pm2 start orchestration\ecosystem.config.js

REM Wait for services to initialize
echo ⏳ Waiting for services to initialize...
timeout /t 15 /nobreak >nul

REM Health check services
echo 🏥 Health checking services...
set /a healthy=0

curl -s -o nul -w "%%{http_code}" http://localhost:8081/health | findstr "200" >nul && (
    echo ✅ Legal-BERT service: healthy
    set /a healthy+=1
) || echo ❌ Legal-BERT service: not responding

curl -s -o nul -w "%%{http_code}" http://localhost:6334/health | findstr "200" >nul && (
    echo ✅ Qdrant service: healthy  
    set /a healthy+=1
) || echo ❌ Qdrant service: not responding

curl -s -o nul -w "%%{http_code}" http://localhost:8082/health | findstr "200" >nul && (
    echo ✅ SIMD service: healthy
    set /a healthy+=1
) || echo ❌ SIMD service: not responding

curl -s -o nul -w "%%{http_code}" http://localhost:3000/health | findstr "200" >nul && (
    echo ✅ Context7 MCP service: healthy
    set /a healthy+=1
) || echo ❌ Context7 MCP service: not responding

curl -s -o nul -w "%%{http_code}" http://localhost:5173/ | findstr "200" >nul && (
    echo ✅ SvelteKit frontend: healthy
    set /a healthy+=1
) || echo ❌ SvelteKit frontend: not responding

echo.
echo 📊 Health Summary: !healthy!/5 services healthy
echo.

if !healthy! GEQ 3 (
    echo ✅ Legal AI Stack Started Successfully!
    echo.
    echo 🎯 Access Points:
    echo   Web UI: http://localhost:5173
    echo   Legal-BERT API: http://localhost:8081
    echo   Vector DB: http://localhost:6334  
    echo   SIMD API: http://localhost:8082
    echo   Context7 MCP: http://localhost:3000
    echo.
    echo 📊 Monitor: pm2 monit
    echo 🛑 Stop: pm2 delete all
    echo 📋 Logs: pm2 logs
    echo.
    
    REM Optional: Open browser
    choice /C YN /M "Open web browser now" /T 10 /D N
    if errorlevel 1 if not errorlevel 2 start http://localhost:5173
) else (
    echo ❌ Some services failed to start properly
    echo 📋 Check logs with: pm2 logs
    echo 🔍 Debug individual services:
    echo   pm2 logs legal-bert-onnx
    echo   pm2 logs qdrant-vector  
    echo   pm2 logs go-simd
    echo   pm2 logs context7-mcp
    echo   pm2 logs sveltekit-frontend
)

echo.
pause