@echo off
title Fix Duplicate Files & Build Issues
echo Fixing duplicate endpoint files...

cd sveltekit-frontend

echo [1/5] Removing duplicate server files...
if exist "src\routes\api\chat\+server.js" del "src\routes\api\chat\+server.js"
if exist "src\routes\api\chat\+server.ts" (
    echo Keeping TypeScript version
) else (
    echo Creating missing +server.ts...
    mkdir "src\routes\api\chat" 2>nul
    > src\routes\api\chat\+server.ts (
        echo import { json } from '@sveltejs/kit';
        echo.
        echo export async function POST({ request }^) {
        echo   const { message } = await request.json(^);
        echo.
        echo   try {
        echo     const response = await fetch('http://localhost:11434/api/generate', {
        echo       method: 'POST',
        echo       headers: { 'Content-Type': 'application/json' },
        echo       body: JSON.stringify({
        echo         model: 'gemma-legal-2b',
        echo         prompt: message,
        echo         stream: false
        echo       }^)
        echo     }^);
        echo.
        echo     const data = await response.json(^);
        echo     return json({ response: data.response }^);
        echo   } catch (error^) {
        echo     return json({ error: 'AI service unavailable' }, { status: 500 }^);
        echo   }
        echo }
    )
)

echo [2/5] Fixing tsconfig.json...
> tsconfig.json (
echo {
echo   "extends": "./.svelte-kit/tsconfig.json"
echo }
)

echo [3/5] Running SvelteKit sync...
npx svelte-kit sync

echo [4/5] Testing build...
npm run build

echo [5/5] If build succeeds, testing dev mode...
if !errorlevel! equ 0 (
    echo ✅ Build successful
    npm run dev
) else (
    echo ❌ Build failed - check errors above
)

cd ..
pause
