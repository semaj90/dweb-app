@echo off
title Quick Fix Duplicate Files
echo Removing duplicate server files...

cd sveltekit-frontend\src\routes\api\chat
if exist "+server.js" del "+server.js"
echo ✅ Removed duplicate .js file

cd ..\..\..\..
echo Testing build...
cd sveltekit-frontend
npm run build

echo ✅ Build should work now
pause
