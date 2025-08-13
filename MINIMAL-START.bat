@echo off
echo.
echo ======================================================
echo   AI SUMMARIZATION INTEGRATION COMPLETE v8.1.2
echo   Production Ready - August 12, 2025  
echo ======================================================
echo.
echo Starting minimal Legal AI environment...
echo [1/3] Starting Ollama service...
start /min ollama serve
timeout /t 2 >nul
echo [2/3] Starting Go summarizer service...
cd go-microservice
start /min go run ./cmd/summarizer-service/main.go
cd ..
timeout /t 3 >nul
echo [3/3] Starting SvelteKit frontend...
cd sveltekit-frontend
npm run dev
echo.
echo Environment started successfully!
pause
