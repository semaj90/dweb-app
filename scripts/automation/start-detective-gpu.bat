@echo off
echo Starting Detective Evidence Synthesizer (GPU Version with Local GGUF)...
echo.
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"
echo ?? Switching to GPU environment...
copy .env.gpu sveltekit-frontend\.env /Y
echo.
echo ?? Starting GPU Docker services...
docker-compose -f docker-compose.gpu.yml up -d
echo.
echo ? Waiting for GPU services to initialize...
timeout /t 15 /nobreak >nul
echo.
echo ?? Setting up local GGUF model for GPU...
powershell -ExecutionPolicy Bypass -File setup-local-model-gpu.ps1
echo.
echo ? GPU Services started on:
echo - PostgreSQL: localhost:5434
echo - Qdrant: localhost:6335
echo - Redis: localhost:6381
echo - Ollama GPU: localhost:11436 (Model: gemma3-detective)
echo.
echo ?? Starting SvelteKit development server...
cd sveltekit-frontend
npm run dev
pause
