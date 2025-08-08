@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

echo Benchmarking both parsers...
echo.

REM Run simple parser (guaranteed to work)
if exist simple_simd.exe (
    echo Simple SIMD (simdjson-go):
    start /B simple_simd.exe
    timeout /t 1 >nul
    curl -X POST http://localhost:8080/parse/simple -H "Content-Type: application/json" -d @benchmark.json
    taskkill /F /IM simple_simd.exe 2>nul
)

echo.
echo -------------------
echo.

REM Try homemade parser (might fail on Windows)
if exist homemade_simd.exe (
    echo Homemade SIMD (AVX2):
    start /B homemade_simd.exe
    timeout /t 1 >nul
    curl -X POST http://localhost:8080/parse/homemade -H "Content-Type: application/json" -d @benchmark.json
    taskkill /F /IM homemade_simd.exe 2>nul
) else (
    echo Homemade parser not available (CGO compilation failed)
)
