@echo off
echo 🕵️ Detective Evidence Synthesizer - CPU Container Creation
echo.
echo Creating CPU containers from scratch...
echo.
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"
echo 🔧 Running PowerShell container creation script...
powershell -ExecutionPolicy Bypass -File create-detective-containers.ps1
echo.
echo ✅ CPU containers created successfully!
echo.
echo 📋 To start your app:
echo 1. cd sveltekit-frontend
echo 2. npm run db:migrate
echo 3. npm run dev
echo.
pause