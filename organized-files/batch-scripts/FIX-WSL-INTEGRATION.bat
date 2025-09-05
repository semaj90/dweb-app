@echo off
title Fix WSL Docker Integration
color 0A

echo Fixing WSL Docker Integration...
echo.

echo Current Docker containers running:
docker ps --format "table {{.Names}}\t{{.Status}}"

echo.
echo WSL Integration Check:
wsl docker version >nul 2>&1 && echo ✅ WSL can access Docker || echo ❌ WSL integration broken

echo.
echo To fix WSL integration:
echo 1. Open Docker Desktop
echo 2. Go to Settings → Resources → WSL Integration  
echo 3. Enable integration with your WSL distro
echo 4. Apply & Restart Docker Desktop

echo.
echo Alternative - Use Windows Docker directly:
echo Your containers are running fine in Windows
echo Skip WSL and continue with Windows commands

pause
