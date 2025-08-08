@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app

REM Database setup
psql -U postgres -c "CREATE DATABASE IF NOT EXISTS codebase_index;" 2>nul
psql -U postgres -d codebase_index -f schema_codebase.sql 2>nul

REM Build and run
cd go-microservice
set CGO_ENABLED=0
go build -o indexer.exe code_indexer.go
taskkill /F /IM indexer.exe 2>nul
start /B indexer.exe
cd ..

timeout /t 1 >nul

REM Index codebase
curl -X POST http://localhost:8080/index

REM Get TypeScript errors and analyze
npm run check 2>errors.txt
curl -X POST http://localhost:8080/analyze -H "Content-Type: application/json" -d @errors.txt

echo.
echo Files generated:
dir /B *.json *.txt *.md
