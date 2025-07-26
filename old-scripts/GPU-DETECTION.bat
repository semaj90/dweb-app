@echo off
title GPU Detection & CUDA Check - Fixed
echo Detecting GPU and CUDA...

echo [1/4] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
) else (
    echo ❌ No NVIDIA GPU or drivers
)

echo [2/4] Checking CUDA version...
nvcc --version 2>nul | findstr "release"
if !errorlevel! equ 0 (
    echo ✅ CUDA detected
) else (
    echo ❌ CUDA not found
)

echo [3/4] Checking Ollama GPU support...
docker exec deeds-ollama nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Ollama has GPU access
) else (
    echo ❌ Ollama no GPU access - restart with GPU compose
)

echo [4/4] Testing model with GPU...
docker exec deeds-ollama ollama run gemma3-legal "Test response" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Model working
) else (
    echo ❌ Model error - run FIX-OLLAMA-MEMORY.bat
)

pause
