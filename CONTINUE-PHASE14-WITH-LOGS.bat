@echo off
set LOG=phase14_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%.log
echo PHASE14 START %date% %time% > %LOG%
echo PHASE14 START %date% %time%

echo [CHECK] Go microservice... 
echo [CHECK] Go microservice... >> %LOG%
cd go-microservice 2>>%LOG%
if exist enhanced-server-simple.go (
    echo [BUILD] enhanced-server-simple.go
    echo [BUILD] enhanced-server-simple.go >> %LOG%
    go build -o phase14-server.exe enhanced-server-simple.go 2>>%LOG%
    if exist phase14-server.exe (
        echo [START] Go server
        echo [START] Go server >> %LOG%
        start /B phase14-server.exe 2>>%LOG%
    )
) else (
    echo [FALLBACK] Using legal-ai-server.go
    echo [FALLBACK] Using legal-ai-server.go >> %LOG%
    go build -o legal-server.exe legal-ai-server.go 2>>%LOG%
    start /B legal-server.exe 2>>%LOG%
)
cd ..

echo [CHECK] PostgreSQL...
echo [CHECK] PostgreSQL... >> %LOG%
net start postgresql-x64-17 2>>%LOG%

echo [CHECK] Ollama...
echo [CHECK] Ollama... >> %LOG%
start /B ollama serve 2>>%LOG%

echo [CHECK] Frontend...
echo [CHECK] Frontend... >> %LOG%
cd sveltekit-frontend 2>>%LOG%
if not exist node_modules npm install 2>>%LOG%
start cmd /k "npm run dev"
cd ..

echo [STATUS] Services:
curl -s http://localhost:8080/health 2>>%LOG% && echo Go: OK || echo Go: ERROR
curl -s http://localhost:11434/api/tags 2>>%LOG% && echo Ollama: OK || echo Ollama: ERROR

echo [LOG] %LOG%
type %LOG%
echo.
echo Press any key to exit...
pause >nul
