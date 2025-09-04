@echo off
REM Enterprise Vector Consumer Service Build Script
REM Builds with all optimizations and enterprise features

echo Building Enterprise Vector Consumer Service v2.0...

REM Set build environment
set CGO_ENABLED=0
set GOOS=windows
set GOARCH=amd64

REM Generate protobuf code
echo Generating protobuf code...
if not exist "pb" mkdir pb
protoc --go_out=pb --go_opt=paths=source_relative ^
       --go-grpc_out=pb --go-grpc_opt=paths=source_relative ^
       proto/vector-service.proto

REM Generate database code with sqlc
echo Generating database code...
if exist "sqlc.exe" (
    sqlc.exe generate
) else (
    echo Warning: sqlc not found, skipping database code generation
)

REM Build optimized binary
echo Building optimized binary...
go build -mod=readonly ^
         -ldflags="-s -w -X main.Version=2.0.0 -X main.BuildTime=%date% %time%" ^
         -gcflags="-trimpath" ^
         -asmflags="-trimpath" ^
         -o bin/vector-consumer-enterprise.exe ^
         vector-consumer-service-v2.go

if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    echo Binary: bin/vector-consumer-enterprise.exe
    echo Size:
    dir bin\vector-consumer-enterprise.exe | find "vector-consumer-enterprise.exe"
) else (
    echo Build failed!
    exit /b 1
)

REM Create deployment package
echo Creating deployment package...
if not exist "deploy" mkdir deploy
copy bin\vector-consumer-enterprise.exe deploy\
copy db\migrations\*.sql deploy\migrations\
copy .env.production deploy\.env
copy README-enterprise.md deploy\

echo Deployment package ready in deploy/ directory
echo.
echo To run:
echo   cd deploy
echo   set DATABASE_URL=postgres://...
echo   set REDIS_URL=redis://...
echo   vector-consumer-enterprise.exe