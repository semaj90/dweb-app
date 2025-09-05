@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app

REM Skip PostgreSQL - use file-based indexing
cd go-microservice
go build -o indexer.exe code_indexer.go
if not exist indexer.exe (
    echo Build failed
    exit /b 1
)

taskkill /F /IM indexer.exe 2>nul
start /B indexer.exe
cd ..

timeout /t 1 >nul

REM Index
curl -X POST http://localhost:8080/index

REM Analyze errors
echo [{"error": "TS2322: Type mismatch"}] > errors.json
curl -X POST http://localhost:8080/analyze -H "Content-Type: application/json" -d @errors.json

echo.
echo Generated:
dir /B *.json *.txt *.md
