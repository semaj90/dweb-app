@echo off
echo COMPLETE AI SYSTEM DEPLOYMENT

REM Kill existing processes
taskkill /F /IM ai-microservice.exe 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

REM Start PostgreSQL
net start postgresql-x64-16 2>nul
pg_isready >nul 2>&1 || echo PostgreSQL not running

REM Create database and schema
psql -U postgres -c "CREATE DATABASE IF NOT EXISTS docs_db;" 2>nul
psql -U postgres -d docs_db -f schema.sql 2>nul

REM Start Redis
start /B "C:\Program Files\Memurai\memurai.exe" 2>nul

REM Start Ollama
start /B ollama serve
timeout /t 2 >nul
ollama pull nomic-embed-text 2>nul
ollama pull llama3 2>nul

REM Build and start Go microservice
cd go-microservice
go mod init microservice 2>nul
go mod tidy
set CGO_ENABLED=1
set CC=clang
set "CGO_CFLAGS=-IC:/Progra~1/NVIDIA~2/CUDA/v12.9/include"
set "CGO_LDFLAGS=-LC:/Progra~1/NVIDIA~2/CUDA/v12.9/lib/x64 -lcudart -lcublas"
go build -o ai-microservice.exe doc_processor.go
start /B ai-microservice.exe
cd ..

timeout /t 3 >nul

REM Start SvelteKit with PM2 for multi-cluster
cd sveltekit-frontend
call npm install
call npm run build
call npm install -g pm2
pm2 start build/index.js -i max --name "sveltekit-app"
cd ..

echo.
echo SYSTEM READY:
echo   Go Microservice: http://localhost:8080
echo   SvelteKit App: http://localhost:5173
echo   PostgreSQL: localhost:5432/docs_db
echo   Redis: localhost:6379
echo   Ollama: http://localhost:11434
echo.
echo Test document processing:
curl -X POST http://localhost:8080/process-document -H "Content-Type: application/json" -d "{\"url\":\"https://example.com/doc.json\"}"
