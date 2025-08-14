@echo off
title Fix API Endpoint Conflict
echo Fixing duplicate API endpoint files...

cd sveltekit-frontend\src\routes\api\chat

echo Current files in chat directory:
dir /b

echo Removing old TypeScript server file...
if exist "+server.ts" (
    del "+server.ts"
    echo ✅ Deleted +server.ts
) else (
    echo ⚠️ +server.ts not found
)

echo Checking remaining files:
dir /b

cd ..\..\..\..

echo Testing Docker build...
docker-compose -f docker-compose-gpu.yml up --build -d

pause
