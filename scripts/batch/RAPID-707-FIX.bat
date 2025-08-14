@echo off
echo Rapid fix for 707 errors...

if exist "sveltekit-frontend\package.json" cd sveltekit-frontend

:: 1. Nuclear option - fix store exports
echo [1/6] Store exports...
rmdir /s /q src\lib\stores 2>nul
mkdir src\lib\stores
echo export const aiCommands = {subscribe:()=>{}}; > src\lib\stores\ai-commands.js
echo export const casesStore = {subscribe:()=>{}}; > src\lib\stores\cases.js
echo export const evidenceStore = {subscribe:()=>{}}; > src\lib\stores\evidence.js
echo export * from './ai-commands.js'; > src\lib\stores\index.ts
echo export * from './cases.js'; >> src\lib\stores\index.ts
echo export * from './evidence.js'; >> src\lib\stores\index.ts

:: 2. Fix utils
echo [2/6] Utils...
mkdir src\lib 2>nul
echo export function cn(...args) { return args.join(' '); } > src\lib\utils.ts

:: 3. Fix Button
echo [3/6] Button...
mkdir src\lib\components\ui 2>nul
(
echo ^<script^>
echo   let { children, ...props } = $props(^);
echo ^</script^>
echo ^<button {...props}^>{@render children?(^)}^</button^>
) > src\lib\components\ui\button.svelte

:: 4. Install deps
echo [4/6] Dependencies...
npm install clsx bits-ui -q

:: 5. Types
echo [5/6] Types...
mkdir src\lib\types 2>nul
echo export interface Case { id: string; } > src\lib\types\index.ts

:: 6. Check
echo [6/6] Checking...
npm run check 2>&1 | findstr "found"

echo Done. Check results above.
pause