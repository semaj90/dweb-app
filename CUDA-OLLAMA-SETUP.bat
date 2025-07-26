@echo off
title CUDA & Ollama GPU Setup
echo Checking CUDA compatibility...

echo Detecting CUDA version...
set "CUDA_VERSION=unknown"
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*" (
    for /d %%i in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        set "CUDA_VERSION=%%~ni"
    )
)

echo CUDA Version: %CUDA_VERSION%

if "%CUDA_VERSION%"=="v12.0" goto CUDA12
if "%CUDA_VERSION%"=="v12.1" goto CUDA12
if "%CUDA_VERSION%"=="v12.2" goto CUDA12
if "%CUDA_VERSION%"=="v12.3" goto CUDA12
if "%CUDA_VERSION%"=="v12.4" goto CUDA12
if "%CUDA_VERSION%"=="v12.5" goto CUDA12
if "%CUDA_VERSION%"=="v12.6" goto CUDA12

:CUDA11
echo ⚠️ CUDA 11 detected - Ollama prefers CUDA 12
echo Recommendation: Update to CUDA 12.x
goto SETUP

:CUDA12
echo ✅ CUDA 12 detected - Perfect for Ollama
goto SETUP

:SETUP
echo Starting Ollama with GPU...
docker-compose -f docker-compose-gpu.yml up -d ollama

echo Testing GPU acceleration...
timeout /t 10 >nul
docker exec deeds-ollama ollama run gemma3-legal "GPU test"

pause
