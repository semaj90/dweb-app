@echo off
echo Starting Detective Evidence Synthesizer (CPU Version with Local GGUF)...
echo.
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
echo ?? Starting Docker services...
docker-compose up -d
echo.
echo ? Waiting for services to initialize...
timeout /t 10 /nobreak >nul
echo.
echo ?? Setting up local GGUF model...
cd ..
powershell -ExecutionPolicy Bypass -File setup-local-model.ps1
echo.
echo ? Services started on:
echo - PostgreSQL: localhost:5433
echo - Qdrant: localhost:6334
echo - Redis: localhost:6380
echo - RabbitMQ: localhost:5673 (Management: 15673)
echo - Ollama: localhost:11435 (Model: gemma3-detective)
echo.
echo ?? Starting SvelteKit development server...
cd sveltekit-frontend
npm run dev
pause
