@echo off
set LOG=start_%date:~10,4%%date:~4,2%%date:~7,2%.log
echo START %date% %time% > %LOG%

net start postgresql-x64-17 2>>%LOG%
start /B ollama serve 2>>%LOG%

cd go-microservice
go build -o server.exe legal-ai-server.go 2>>%LOG%
start /B server.exe 2>>%LOG%
cd ..

cd sveltekit-frontend
if not exist node_modules npm install 2>>%LOG%
start cmd /k "npm run dev"
cd ..

type %LOG%
timeout /t 10
