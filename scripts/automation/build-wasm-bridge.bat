@echo off
echo 🦀 Building Rust WASM Bridge for Legal AI System...

REM Check if wasm-pack is installed
where wasm-pack >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ wasm-pack not found. Installing...
    cargo install wasm-pack
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to install wasm-pack
        pause
        exit /b 1
    )
)

REM Check if Rust is installed
where rustc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Rust not found. Please install Rust from https://rustup.rs/
    pause
    exit /b 1
)

echo ✅ Prerequisites checked

REM Navigate to webasm-bridge directory
cd rust-services\webasm-bridge

REM Clean previous builds
if exist pkg rmdir /s /q pkg
if exist target rmdir /s /q target

echo 🔨 Building WASM package...

REM Build the WASM package for web target
wasm-pack build --target web --out-dir ../../sveltekit-frontend/src/lib/wasm/pkg --scope legal-ai

if %ERRORLEVEL% NEQ 0 (
    echo ❌ WASM build failed
    cd ..\..
    pause
    exit /b 1
)

cd ..\..

echo ✅ WASM package built successfully

REM Copy additional files if needed
if exist sveltekit-frontend\src\lib\wasm\pkg (
    echo 📦 WASM package available at: sveltekit-frontend\src\lib\wasm\pkg
    echo 🔧 Package contents:
    dir sveltekit-frontend\src\lib\wasm\pkg
) else (
    echo ❌ WASM package not found after build
    exit /b 1
)

echo 🚀 Integration ready! You can now:
echo 1. Import the Rust bridge in your SvelteKit components
echo 2. Use initRustBridge() to initialize the WASM module
echo 3. Access native Windows functionality through the bridge

echo 💡 Example usage:
echo   import { initRustBridge, getSystemInfo } from '$lib/wasm/rust-bridge';
echo   await initRustBridge();
echo   const info = getSystemInfo();

pause