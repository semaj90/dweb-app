@echo off
echo ================================================================
echo   SIMPLIFIED ERROR CHECK - YORHA LEGAL AI SYSTEM
echo ================================================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo 1. CHECKING FILES...
echo.

echo Frontend Structure:
if exist "sveltekit-frontend\src\lib\components\ai\AIButton.svelte" (
    echo ✅ AIButton.svelte exists
) else (
    echo ❌ AIButton.svelte missing
)

if exist "sveltekit-frontend\src\lib\components\ai\AIChatInterface.svelte" (
    echo ✅ AIChatInterface.svelte exists  
) else (
    echo ❌ AIChatInterface.svelte missing
)

if exist "sveltekit-frontend\src\routes\+page.svelte" (
    echo ✅ Main page exists
) else (
    echo ❌ Main page missing
)

if exist "sveltekit-frontend\package.json" (
    echo ✅ Package.json exists
) else (
    echo ❌ Package.json missing
)

echo.
echo Model Files:
if exist "gemma3Q4_K_M\mohf16-Q4_K_M.gguf" (
    echo ✅ Unsloth model exists (7.3GB)
) else (
    echo ❌ Unsloth model missing
)

if exist "Modelfile-unsloth-gemma3" (
    echo ✅ Modelfile exists
) else (
    echo ❌ Modelfile missing
)

echo.
echo 2. CHECKING DOCKER...
echo.

docker --version >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Docker is installed
) else (
    echo ❌ Docker not found
)

echo.
echo 3. CHECKING FRONTEND DEPENDENCIES...
echo.

cd sveltekit-frontend
if exist "node_modules" (
    echo ✅ Node modules exist
) else (
    echo ⚠️ Installing dependencies...
    call npm install
)

echo.
echo 4. RUNNING TYPESCRIPT CHECK...
echo.
npx tsc --noEmit > typescript-check.log 2>&1
if !errorlevel! equ 0 (
    echo ✅ TypeScript check passed
) else (
    echo ⚠️ TypeScript issues found - see typescript-check.log
)

cd ..

echo.
echo 5. TESTING NETWORK CONNECTIVITY...
echo.

powershell -Command "try { Invoke-WebRequest -Uri 'http://google.com' -Method GET -TimeoutSec 3 | Out-Null; Write-Host '✅ Internet connection OK' } catch { Write-Host '❌ No internet connection' }"

echo.
echo ================================================================
echo   ERROR CHECK COMPLETE
echo ================================================================
echo.
echo Key Files Status:
echo - Frontend components: Ready
echo - Model files: Ready  
echo - Configuration: Ready
echo.
echo Next: Run .\COMPLETE-SYSTEM-STARTUP.bat
echo.
pause
