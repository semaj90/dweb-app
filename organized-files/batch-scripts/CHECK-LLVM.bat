@echo off
echo LLVM/Clang Environment Check
echo =============================
echo.

echo [1] Checking PATH for LLVM...
where clang >nul 2>&1
if %errorlevel%==0 (
    echo [OK] clang found in PATH
    where clang
    clang --version | findstr version
) else (
    echo [FAIL] clang not in PATH
)

echo.
echo [2] Checking common LLVM locations...
if exist "C:\Program Files\LLVM\bin\clang.exe" (
    echo [OK] Found at C:\Program Files\LLVM
    "C:\Program Files\LLVM\bin\clang.exe" --version | findstr version
) else if exist "C:\LLVM\bin\clang.exe" (
    echo [OK] Found at C:\LLVM
    "C:\LLVM\bin\clang.exe" --version | findstr version
) else (
    echo [FAIL] LLVM not found in standard locations
)

echo.
echo [3] Checking MSVC compatibility...
where cl >nul 2>&1
if %errorlevel%==0 (
    echo [OK] MSVC compiler found
    cl 2>&1 | findstr Version
) else (
    echo [INFO] MSVC not found (optional)
)

echo.
echo [4] Testing CGO with Clang...
echo package main > test_cgo.go
echo // #include ^<stdio.h^> >> test_cgo.go
echo import "C" >> test_cgo.go
echo func main() { C.printf(C.CString("CGO works\n")) } >> test_cgo.go

set CC=clang
set CGO_ENABLED=1
go build -x test_cgo.go 2>&1 | findstr "error fatal warning"
if exist test_cgo.exe (
    echo [OK] CGO compilation successful
    del test_cgo.exe
) else (
    echo [FAIL] CGO with Clang failed
)
del test_cgo.go 2>nul

echo.
echo [5] Checking AVX2 support...
powershell -Command "Get-WmiObject Win32_Processor | Select-Object -ExpandProperty Description"
wmic cpu get Name, Manufacturer, MaxClockSpeed /value | findstr "="

echo.
echo [6] Recommended setup:
if exist "C:\Program Files\LLVM\bin\clang.exe" (
    echo set PATH=C:\Program Files\LLVM\bin;%%PATH%%
    echo set CC=clang
    echo set CXX=clang++
) else (
    echo Install LLVM: winget install LLVM.LLVM
    echo Or download from: https://github.com/llvm/llvm-project/releases
)
