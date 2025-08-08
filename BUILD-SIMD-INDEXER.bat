@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

go build -o indexer.exe code_indexer_simd.go
if not exist indexer.exe exit /b 1

taskkill /F /IM indexer.exe 2>nul
start /B indexer.exe

timeout /t 1 >nul
curl -X POST http://localhost:8080/index
