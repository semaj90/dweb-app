@echo off
echo 🚀 Running SvelteKit Setup and Check...
echo.

cd /d "C:\Users\james\Desktop\web-app"

powershell -ExecutionPolicy Bypass -File "run-setup-and-check.ps1"

echo.
echo ✅ Setup complete! Check the output above for any issues.
echo.
pause
