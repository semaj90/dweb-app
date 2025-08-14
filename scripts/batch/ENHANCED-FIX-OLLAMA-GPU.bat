@echo off
title Enhanced Ollama GPU Fix - RTX 3060
echo ===================================================
echo    ENHANCED OLLAMA GPU FIX FOR RTX 3060
echo ===================================================
echo.

echo [1/6] Stopping any running containers...
docker-compose down
docker-compose -f docker-compose-gpu.yml down
timeout /t 3 > nul

echo [2/6] Creating enhanced .env file with CUDA optimization...
echo # Docker Environment Variables - Auto-generated > .env
echo NVIDIA_VISIBLE_DEVICES=all >> .env
echo NVIDIA_DRIVER_CAPABILITIES=compute,utility >> .env
echo OLLAMA_FLASH_ATTENTION=1 >> .env
echo OLLAMA_MODELS=/models >> .env
echo OLLAMA_KEEP_ALIVE=5m >> .env
echo OLLAMA_VERBOSE=1 >> .env
echo OLLAMA_NUM_GPU=1 >> .env
echo OLLAMA_GPU_LAYERS=50 >> .env
echo DATABASE_URL=postgresql://legal_admin:LegalSecure2024!@deeds-postgres:5432/prosecutor_db >> .env
echo OLLAMA_URL=http://localhost:11434 >> .env
echo PUBLIC_ORIGIN=http://localhost:5173 >> .env

echo [3/6] Creating enhanced Modelfile-Ampere with RTX 3060 optimizations...
echo # Optimized for NVIDIA RTX 3060 Ampere GPU > Modelfile-Ampere
echo FROM /tmp/mo16.gguf >> Modelfile-Ampere
echo. >> Modelfile-Ampere
echo TEMPLATE """^<start_of_turn^>user >> Modelfile-Ampere
echo {{ .Prompt }}^<end_of_turn^> >> Modelfile-Ampere
echo ^<start_of_turn^>model >> Modelfile-Ampere
echo """ >> Modelfile-Ampere
echo. >> Modelfile-Ampere
echo SYSTEM """Legal AI for prosecutors: evidence analysis, case timelines, legal research.""" >> Modelfile-Ampere
echo. >> Modelfile-Ampere
echo PARAMETER temperature 0.3 >> Modelfile-Ampere
echo PARAMETER top_k 20 >> Modelfile-Ampere
echo PARAMETER top_p 0.8 >> Modelfile-Ampere
echo PARAMETER repeat_penalty 1.1 >> Modelfile-Ampere
echo PARAMETER num_ctx 2048 >> Modelfile-Ampere
echo PARAMETER num_predict 256 >> Modelfile-Ampere
echo PARAMETER num_gpu 1 >> Modelfile-Ampere
echo PARAMETER num_thread 4 >> Modelfile-Ampere
echo PARAMETER use_flash_attn true >> Modelfile-Ampere
echo PARAMETER rope_freq_scale 0.5 >> Modelfile-Ampere
echo PARAMETER f16_kv true >> Modelfile-Ampere

echo [4/6] Starting GPU-optimized containers...
docker-compose -f docker-compose-gpu.yml up -d

timeout /t 10 > nul

echo [5/6] Verifying CUDA and GPU access...
docker exec deeds-ollama-gpu nvidia-smi || echo "ERROR: GPU not accessible!"
echo.
docker exec deeds-ollama-gpu ollama --version
echo.

echo [6/6] Recreating model with Ampere-optimized settings...
docker cp Modelfile-Ampere deeds-ollama-gpu:/tmp/
docker exec deeds-ollama-gpu ollama rm gemma3-legal
docker exec deeds-ollama-gpu ollama create gemma3-legal -f /tmp/Modelfile-Ampere

echo ===================================================
echo Testing optimized model...
echo ===================================================
docker exec deeds-ollama-gpu ollama list
echo.
echo Running test prompt...
docker exec deeds-ollama-gpu ollama run gemma3-legal "Provide a concise explanation of what makes evidence admissible in court."

echo.
echo ✅ Done! Ollama now has optimized GPU access for your RTX 3060.
echo    If you still have issues, try rebooting your computer and running this script again.
echo.

echo ===================================================
echo NEXT STEPS:
echo ===================================================
echo 1. Run your database migrations with: npm run db:migrate
echo 2. Seed your database with: npm run seed
echo 3. Start the SvelteKit frontend: npm run dev
echo 4. Visit http://localhost:5173/chat to test the AI chat interface
echo.
pause
