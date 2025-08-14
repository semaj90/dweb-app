@echo off
title Check App Directory Files
echo Scanning for issues in app files...

echo [1/6] Checking for duplicate files...
cd sveltekit-frontend\src\routes\api
if exist "chat\+server.js" if exist "chat\+server.ts" (
    echo ❌ Duplicate chat server files found
    del "chat\+server.js"
    echo ✅ Removed .js duplicate
)

echo [2/6] Checking model references...
cd ..\..\..\..
findstr /r /s "gemma3-legal" sveltekit-frontend\src\*.* 2>nul
if !errorlevel! equ 0 (
    echo ⚠️ Found gemma3-legal references -
)

echo [3/6] Checking TypeScript config...
cd sveltekit-frontend
if not exist ".svelte-kit\tsconfig.json" (
    echo ❌ Missing .svelte-kit/tsconfig.json - run sync
    npx svelte-kit sync
)

echo [4/6] Checking for 3GB model issues...
cd ..
docker exec deeds-ollama-gpu ollama list | findstr "3.0 GB" && echo ⚠️ Large model detected

echo [5/6] Verifying GPU container...
docker ps | findstr "deeds-ollama-gpu" && echo ✅ GPU container running || echo ❌ GPU container down

echo [6/6] Testing lightweight model...
docker exec deeds-ollama-gpu ollama run gemma-legal-2b "Test" 2>nul && echo ✅ 2B model works || echo ❌ 2B model failed

echo.
echo Scan complete - run FIX-BUILD-ERRORS.bat to resolve issues
pause
