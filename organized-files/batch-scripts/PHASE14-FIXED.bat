@echo off
set LOG=phase14_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%.log
echo PHASE14 START %date% %time% > %LOG%

echo [CHECK] PostgreSQL...
net start postgresql-x64-17 2>>%LOG%
if errorlevel 1 (
    echo PostgreSQL: Admin required - skipping
    echo PostgreSQL: Admin required >> %LOG%
) else (
    echo PostgreSQL: Started
)

echo [CHECK] Ollama...
start /B ollama serve 2>>%LOG%

echo [CHECK] Go server...
cd go-microservice
start /B legal-ai-server.exe 2>>%LOG%
cd ..

echo [CHECK] Frontend...
cd sveltekit-frontend
start cmd /k "npm run dev"
cd ..

timeout /t 3
curl -s http://localhost:8080/health && echo Go: OK
curl -s http://localhost:11434/api/tags >nul && echo Ollama: OK

echo System ready. Check %LOG%
timeout /t 60
