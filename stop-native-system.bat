@echo off
echo Stopping Native Windows Legal AI System...
echo.

echo [1/4] Stopping application processes...
echo   Stopping Node.js processes...
taskkill /f /im node.exe 2>nul
if %errorlevel% equ 0 (
    echo   ✅ Node.js processes stopped
) else (
    echo   ✅ No Node.js processes running
)

echo   Stopping Go processes...
taskkill /f /im main.exe 2>nul
taskkill /f /im go.exe 2>nul
if %errorlevel% equ 0 (
    echo   ✅ Go processes stopped
) else (
    echo   ✅ No Go processes running
)

echo   Stopping Ollama...
taskkill /f /im ollama.exe 2>nul
if %errorlevel% equ 0 (
    echo   ✅ Ollama stopped
) else (
    echo   ✅ Ollama not running
)

echo.
echo [2/4] Stopping Windows Services...
echo   Stopping Redis...
net stop Redis >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ Redis stopped
) else (
    echo   ✅ Redis already stopped
)

echo   Stopping Qdrant...
net stop Qdrant >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ Qdrant stopped
) else (
    echo   ✅ Qdrant already stopped
)

echo   Stopping MinIO...
net stop MinIO >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ MinIO stopped
) else (
    echo   ✅ MinIO already stopped
)

echo   Stopping PostgreSQL...
net stop legal-ai-postgres >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ PostgreSQL stopped
) else (
    echo   ✅ PostgreSQL already stopped
)

echo.
echo [3/4] Cleaning up temporary files...
if exist "%TEMP%\legal-ai-*" (
    del /q "%TEMP%\legal-ai-*" 2>nul
    echo   ✅ Temporary files cleaned
) else (
    echo   ✅ No temporary files to clean
)

echo.
echo [4/4] Verifying shutdown...
timeout /t 3 /nobreak >nul

echo   Checking for remaining processes...
tasklist /FI "IMAGENAME eq node.exe" 2>NUL | find /I /N "node.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo   ✅ No Node.js processes running
) else (
    echo   ⚠️ Some Node.js processes still running
)

tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo   ✅ Ollama stopped
) else (
    echo   ⚠️ Ollama still running
)

echo.
echo ✅ Native Windows Legal AI System Stopped!
echo.
echo 📋 All services have been shut down:
echo   • PostgreSQL service stopped
echo   • MinIO service stopped  
echo   • Qdrant service stopped
echo   • Redis service stopped
echo   • Ollama process terminated
echo   • Frontend/Backend processes terminated
echo.
echo 🔧 To restart the system, run: start-native-system.bat
echo.
pause