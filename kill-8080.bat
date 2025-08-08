@echo off
taskkill /F /IM ai-microservice.exe 2>nul
netstat -ano | findstr :8080
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul
echo Port 8080 cleared
