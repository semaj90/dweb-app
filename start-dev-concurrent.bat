@echo off
echo 🚀 Starting Legal AI Development Environment
echo.

REM Set environment variables for professional PATH setup
set "PATH=%PATH%;C:\Program Files\LLVM\bin"
set "CC=clang"
set "CXX=clang++"
set "CGO_ENABLED=1"
set "CGO_LDFLAGS=-lkernel32"

echo ✅ Environment configured with professional PATH setup
echo.

REM Start SvelteKit dev server in background
echo 📦 Starting SvelteKit dev server on :5173...
start /B cmd /c "cd sveltekit-frontend && npm run dev"

REM Wait a moment for SvelteKit to start
timeout /t 3 /nobreak >nul

REM Build and start Go enhanced server
echo 🔨 Building Go enhanced server...
cd go-microservice

REM Build the enhanced server
go build -o dev-proxy-server.exe dev-proxy-server.go batch_embed.go main.go

if %errorlevel% neq 0 (
    echo ❌ Go build failed
    pause
    exit /b 1
)

echo ✅ Go build successful
echo.

echo 🚀 Starting Go-Enhanced Vite Server on :3000...
echo 📡 Proxying Vite requests to :5173
echo 🤖 Go API endpoints: /api/*, /ws, /health
echo.
echo 🌐 Access your app at: http://localhost:3000
echo.

REM Start the Go server (this will run in foreground)
dev-proxy-server.exe

pause