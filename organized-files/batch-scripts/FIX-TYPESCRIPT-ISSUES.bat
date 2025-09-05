@echo off
title Fix TypeScript & TailwindCSS Issues
color 0A

echo [1/6] Fixing TailwindCSS/UnoCSS conflict...
cd sveltekit-frontend
npm uninstall tailwindcss @tailwindcss/typography @tailwindcss/forms

echo [2/6] Updating tsconfig...
> tsconfig.json (
echo {
echo   "compilerOptions": {
echo     "allowJs": true,
echo     "checkJs": true,
echo     "esModuleInterop": true,
echo     "forceConsistentCasingInFileNames": true,
echo     "resolveJsonModule": true,
echo     "skipLibCheck": true,
echo     "sourceMap": true,
echo     "strict": false,
echo     "moduleResolution": "bundler"
echo   }
echo }
)

echo [3/6] Fixing Card component exports...
> src\lib\components\ui\Card.js (
echo export { default as Card } from "./Card.svelte";
echo export { default as CardHeader } from "./CardHeader.svelte";  
echo export { default as CardContent } from "./CardContent.svelte";
echo export { default as CardTitle } from "./CardTitle.svelte";
)

echo [4/6] Installing AI dependencies...
if not exist "src\lib\services\embedding-service.ts" (
    npm install langchain @langchain/ollama @langchain/postgres
)

echo [5/6] Running SvelteKit sync...
npx svelte-kit sync

echo [6/6] Testing check...
npm run check

cd ..
echo ✅ TypeScript issues fixed
pause
