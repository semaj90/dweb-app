@echo off
REM build-go-server.bat
REM Build script for the enhanced Go GPU server

echo Building Enhanced Legal AI Server...
echo.

cd go-microservice

REM Install dependencies
echo Installing Go dependencies...
go mod tidy

REM Build the server
echo Building server executable...
go build -o enhanced-legal-server.exe enhanced-legal-server.go

if %errorlevel% == 0 (
    echo.
    echo Build successful!
    echo.
    echo To run the server:
    echo   cd go-microservice
    echo   enhanced-legal-server.exe
    echo.
    echo Or use the complete startup script:
    echo   COMPLETE-AI-SYSTEM-STARTUP.bat
) else (
    echo.
    echo Build failed!
    echo Please check for errors above.
)

cd ..
pause
