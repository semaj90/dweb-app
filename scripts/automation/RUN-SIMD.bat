@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

set CGO_ENABLED=0
go build -ldflags "-s -w" -o simd-parser.exe simd_parser.go

taskkill /F /IM simd-parser.exe 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

start /B simd-parser.exe
timeout /t 1 >nul

echo Benchmarking SIMD performance...
curl -X POST http://localhost:8080/parse/simd -H "Content-Type: application/json" -d "{\"test\":\"data\",\"array\":[1,2,3,4,5]}"
