@echo off
REM Simple Native Windows Build - No Docker, No Protobuf Dependencies
echo ====================================
echo Simple Enterprise Vector Service Build
echo Native Windows - No Docker
echo ====================================

REM Check Go installation
go version
if %ERRORLEVEL% neq 0 (
    echo ERROR: Go not found
    exit /b 1
)

REM Create bin directory
if not exist "bin" mkdir bin

REM Simple Go build without complex dependencies
echo Building simple vector service...
go build -o bin\simple-vector-service.exe .\simple-main.go

if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed
    exit /b 1
)

echo ====================================
echo Build completed successfully!
echo Executable: bin\simple-vector-service.exe
echo ====================================
pause