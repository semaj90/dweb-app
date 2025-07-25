@echo off
echo 🕵️ Detective Evidence Synthesizer - GPU Container Creation
echo.
echo Creating GPU-accelerated containers from scratch...
echo.
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"
echo 🔧 Running PowerShell GPU container creation script...
powershell -ExecutionPolicy Bypass -File create-detective-containers.ps1 -GPU
echo.
echo ✅ GPU containers created successfully!
echo.
echo 📋 To start your app:
echo 1. cd sveltekit-frontend
echo 2. npm run db:migrate
echo 3. npm run dev
echo.
pause