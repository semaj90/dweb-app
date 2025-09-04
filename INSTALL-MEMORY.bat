@echo off
REM Quick installer for Memory System
cls
echo =====================================
echo   Memory System Quick Installer
echo =====================================
echo.
echo This will install the complete memory monitoring system.
echo.
pause

REM Run PowerShell installer
powershell -ExecutionPolicy Bypass -File INSTALL-MEMORY-SYSTEM.ps1

exit /b