@echo off
title Fix API Conflict - Corrected Path
echo Fixed duplicate files, now running Docker build...

echo Current directory: %CD%
echo Running Docker build from correct location...

docker-compose -f docker-compose-gpu.yml up --build -d

if !errorlevel! equ 0 (
    echo ✅ Docker build successful!
    echo Testing the app...
    timeout /t 10 >nul
    echo App should be available at: http://localhost:5173
) else (
    echo ❌ Docker build failed
    echo Check the logs above for errors
)

pause
