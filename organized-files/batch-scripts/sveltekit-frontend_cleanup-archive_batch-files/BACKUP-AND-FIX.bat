@echo off
REM COMPREHENSIVE BACKUP & ERROR FIX
REM Creates phase backups then applies all fixes

echo 📦 CREATING PHASE BACKUPS & FIXING ERRORS...

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

REM Run comprehensive backup and fix
node comprehensive-fix.js

echo.
echo ✅ BACKUP & FIX COMPLETE
echo 📁 Backups: src\lib\stores\phase-backups\
echo 🎯 Ready for Phase 3

pause
