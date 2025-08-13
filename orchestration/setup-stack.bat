@echo off
setlocal enabledelayedexpansion

echo 🛠️  Legal AI Native Windows Stack - Complete Setup
echo.

REM Create all necessary directories
echo 📁 Creating project structure...

REM Go services structure
mkdir go-services\legal-bert-onnx 2>nul
mkdir go-services\simd-operations 2>nul  
mkdir go-services\quic-server 2>nul
mkdir go-services\grpc-gateway 2>nul

REM Rust services structure
mkdir rust-services\qdrant-vector 2>nul
mkdir rust-services\webasm-bridge 2>nul
mkdir rust-services\tensor-processing 2>nul

REM Desktop app structure  
mkdir desktop-app\cmd 2>nul
mkdir desktop-app\internal\services 2>nul
mkdir desktop-app\internal\webasm 2>nul
mkdir desktop-app\internal\coordination 2>nul
mkdir desktop-app\frontend 2>nul

REM Shared structure
mkdir shared\proto 2>nul
mkdir shared\types 2>nul
mkdir shared\config 2>nul

REM Runtime directories
mkdir models 2>nul
mkdir documents 2>nul
mkdir logs 2>nul
mkdir certs 2>nul

echo ✅ Directory structure created

REM Initialize Go modules
echo 🔧 Initializing Go modules...

cd go-services\legal-bert-onnx
echo module legal-ai/legal-bert-onnx > go.mod
echo. >> go.mod
echo go 1.21 >> go.mod
cd ..\..

cd go-services\simd-operations  
echo module legal-ai/simd-operations > go.mod
echo. >> go.mod
echo go 1.21 >> go.mod
cd ..\..

cd go-services\quic-server
echo module legal-ai/quic-server > go.mod
echo. >> go.mod  
echo go 1.21 >> go.mod
cd ..\..

cd desktop-app
echo module legal-ai/desktop-app > go.mod
echo. >> go.mod
echo go 1.21 >> go.mod
cd ..

REM Initialize Rust projects
echo 🦀 Initializing Rust projects...

cd rust-services\qdrant-vector
cargo init --name qdrant-service 2>nul
cd ..\..

cd rust-services\webasm-bridge
cargo init --lib --name webasm-bridge 2>nul
cd ..\..

REM Setup SvelteKit frontend enhancements
if exist "sveltekit-frontend\package.json" (
    echo 📦 Adding WebGPU and XState dependencies...
    cd sveltekit-frontend
    
    REM Add new dependencies to package.json
    npm install --save-dev @types/webgpu
    npm install xstate @xstate/svelte 
    npm install @tensorflow/tfjs @tensorflow/tfjs-backend-webgpu
    
    REM Create additional directories
    mkdir src\lib\webgpu 2>nul
    mkdir src\lib\xstate 2>nul  
    mkdir src\lib\webasm 2>nul
    mkdir src\lib\services\clients 2>nul
    
    cd ..
) else (
    echo ⚠️  SvelteKit frontend not found - run this from the project root
)

REM Download essential dependencies
echo 📥 Setting up development tools...

REM Install global tools if not present
npm list -g pm2 >nul 2>&1 || npm install -g pm2
npm list -g wasm-pack >nul 2>&1 || (
    echo Installing wasm-pack for Rust WASM builds...
    curl -sSf https://rustwasm.github.io/wasm-pack/installer/init.sh | sh 2>nul || echo ⚠️  Manual wasm-pack installation required
)

REM Setup environment template
echo 🔧 Creating environment configuration...

echo # Legal AI Native Windows Stack Configuration > .env.example
echo. >> .env.example
echo # Database >> .env.example  
echo DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db >> .env.example
echo POSTGRES_USER=legal_admin >> .env.example
echo POSTGRES_PASSWORD=123456 >> .env.example
echo. >> .env.example
echo # Ollama Configuration >> .env.example
echo OLLAMA_URL=http://localhost:11434 >> .env.example
echo OLLAMA_MODEL=gemma3-legal >> .env.example
echo OLLAMA_EMBED_MODEL=nomic-embed-text >> .env.example
echo. >> .env.example
echo # Service Ports >> .env.example
echo LEGAL_BERT_PORT=8081 >> .env.example
echo QDRANT_PORT=6334 >> .env.example
echo SIMD_PORT=8082 >> .env.example
echo QUIC_PORT=8083 >> .env.example
echo CONTEXT7_PORT=3000 >> .env.example
echo SVELTEKIT_PORT=5173 >> .env.example
echo. >> .env.example
echo # Performance Settings >> .env.example
echo ENABLE_WEBGPU=true >> .env.example
echo ENABLE_WASM=true >> .env.example
echo ENABLE_SIMD=true >> .env.example
echo TENSOR_CACHE_SIZE=2GB >> .env.example
echo WORKER_THREADS=8 >> .env.example

if not exist ".env" copy .env.example .env

echo ✅ Environment configuration created

REM Create VS Code workspace configuration
echo 🔧 Setting up VS Code workspace...

echo { > legal-ai-stack.code-workspace
echo   "folders": [ >> legal-ai-stack.code-workspace
echo     { "name": "Root", "path": "." }, >> legal-ai-stack.code-workspace
echo     { "name": "SvelteKit Frontend", "path": "./sveltekit-frontend" }, >> legal-ai-stack.code-workspace
echo     { "name": "Go Services", "path": "./go-services" }, >> legal-ai-stack.code-workspace
echo     { "name": "Rust Services", "path": "./rust-services" }, >> legal-ai-stack.code-workspace
echo     { "name": "Desktop App", "path": "./desktop-app" }, >> legal-ai-stack.code-workspace
echo     { "name": "MCP Server", "path": "./mcp" } >> legal-ai-stack.code-workspace
echo   ], >> legal-ai-stack.code-workspace
echo   "settings": { >> legal-ai-stack.code-workspace
echo     "go.gopath": "./go-services", >> legal-ai-stack.code-workspace
echo     "rust-analyzer.linkedProjects": ["./rust-services/*/Cargo.toml"], >> legal-ai-stack.code-workspace
echo     "typescript.preferences.includePackageJsonAutoImports": "on" >> legal-ai-stack.code-workspace
echo   } >> legal-ai-stack.code-workspace
echo } >> legal-ai-stack.code-workspace

echo ✅ VS Code workspace configured

REM Create quick start scripts
echo 📜 Creating utility scripts...

echo REM Quick development start > dev-start.bat
echo @echo off >> dev-start.bat
echo echo Starting development environment... >> dev-start.bat
echo pm2 start orchestration\ecosystem.config.js >> dev-start.bat
echo echo Development started! Monitor with: pm2 monit >> dev-start.bat

echo REM Quick stop all services > stop-all.bat
echo @echo off >> stop-all.bat
echo echo Stopping all services... >> stop-all.bat
echo pm2 delete all >> stop-all.bat
echo echo All services stopped >> stop-all.bat

echo REM Service status check > status.bat
echo @echo off >> status.bat
echo echo Legal AI Service Status: >> status.bat
echo pm2 status >> status.bat

echo ✅ Utility scripts created

echo.
echo 🎉 Legal AI Native Windows Stack setup complete!
echo.
echo 📋 Next steps:
echo   1. Run: .\orchestration\start-stack.bat
echo   2. Open VS Code: code legal-ai-stack.code-workspace
echo   3. Monitor services: pm2 monit
echo.
echo 🔗 Key files created:
echo   - orchestration\ecosystem.config.js (PM2 configuration)
echo   - orchestration\start-stack.bat (Main startup script)
echo   - .env.example (Environment template)
echo   - legal-ai-stack.code-workspace (VS Code workspace)
echo.
echo 📚 Documentation:
echo   - README.md (Updated with architecture)
echo   - Go to: http://localhost:5173 after starting
echo.

pause