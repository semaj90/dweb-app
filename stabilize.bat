@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Clean module state
move go.mod go.mod.broken 2>nul
copy go.mod.clean go.mod
go mod tidy

REM Build without CGO
set CGO_ENABLED=0
go build -ldflags "-s -w" -o service.exe main.go

REM Service already running, no action needed
echo Service operational: http://localhost:8080
curl -s http://localhost:8080/health
