@echo off
set LOG=errors_%date:~10,4%%date:~4,2%%date:~7,2%.log
echo ERROR CHECK %date% %time% > %LOG%

echo [SERVICES] Checking ports...
netstat -an | findstr ":5432 :6379 :8080 :11434 :5173" >> %LOG%

echo [GO] Building...
cd go-microservice 2>>%LOG%
go build legal-ai-server.go 2>>%LOG%
cd ..

echo [TS] Checking...
cd sveltekit-frontend 2>>%LOG%
npx tsc --noEmit 2>>%LOG%
cd ..

type %LOG%
echo.
echo Log: %LOG%
timeout /t 5
