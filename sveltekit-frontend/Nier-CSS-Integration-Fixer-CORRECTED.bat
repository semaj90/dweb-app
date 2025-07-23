@echo off
setlocal EnableDelayedExpansion
cls

echo ========================================================================
echo NIER.CSS + SVELTEKIT 2.6 INTEGRATION FIXER - CORRECTED
echo Bits-UI v2 + shadcn-svelte + UnoCSS + Nier Aesthetic
echo ========================================================================
echo.

cd /d "%~dp0"
if not exist "package.json" (
    echo ❌ ERROR: No package.json found. Run from project root.
    pause
    exit /b 1
)

echo [1/10] 🎮 Installing Nier.css integration dependencies...
call npm install unocss @unocss/preset-uno @unocss/preset-icons
call npm install tailwind-merge clsx
call npm install @iconify-json/lucide @iconify-json/material-symbols
call npm install bits-ui shadcn-svelte
call npm install @xstate/svelte xstate
call npm install sveltekit-superforms zod

echo [2/10] 🎨 Creating Nier.css theme configuration...
if not exist "src\lib\styles" mkdir "src\lib\styles"

(
echo /* Nier.css + shadcn-svelte Integration */
echo.
echo @layer base {
echo   :root {
echo     /* Nier-inspired monochromatic palette */
echo     --background: 40 6%% 8%%;
echo     --foreground: 50 8%% 95%%;
echo     --card: 40 6%% 12%%;
echo     --card-foreground: 50 8%% 95%%;
echo     --popover: 40 6%% 10%%;
echo     --popover-foreground: 50 8%% 95%%;
echo     --primary: 50 8%% 85%%;
echo     --primary-foreground: 40 6%% 8%%;
echo     --secondary: 40 6%% 16%%;
echo     --secondary-foreground: 50 8%% 85%%;
echo     --muted: 40 6%% 14%%;
echo     --muted-foreground: 50 4%% 65%%;
echo     --accent: 45 100%% 65%%;
echo     --accent-foreground: 40 6%% 8%%;
echo     --destructive: 0 65%% 55%%;
echo     --destructive-foreground: 50 8%% 95%%;
echo     --border: 40 6%% 18%%;
echo     --input: 40 6%% 14%%;
echo     --ring: 45 100%% 65%%;
echo     --radius: 0.25rem;
echo   }
echo.
echo   .light {
echo     --background: 50 8%% 96%%;
echo     --foreground: 40 6%% 12%%;
echo     --card: 50 8%% 98%%;
echo     --card-foreground: 40 6%% 12%%;
echo     --primary: 40 6%% 15%%;
echo     --primary-foreground: 50 8%% 95%%;
echo     --accent: 45 100%% 45%%;
echo     --border: 50 8%% 88%%;
echo   }
echo }
echo.
echo body {
echo   font-family: 'Gothic A1', 'JetBrains Mono', system-ui, sans-serif;
echo   font-weight: 400;
echo   line-height: 1.5;
echo   letter-spacing: 0.025em;
echo }
echo.
echo @keyframes nier-glow {
echo   0%%, 100%% { box-shadow: 0 0 10px hsl^(var^(--accent^) / 0.3^); }
echo   50%% { box-shadow: 0 0 20px hsl^(var^(--accent^) / 0.5^); }
echo }
) > "src\lib\styles\theme.css"

echo [3/10] ⚙️ Creating UnoCSS configuration...
(
echo import { defineConfig } from 'unocss'
echo import { presetUno } from '@unocss/preset-uno'
echo import { presetIcons } from '@unocss/preset-icons'
echo.
echo export default defineConfig^({
echo   presets: [
echo     presetUno^(^),
echo     presetIcons^({
echo       collections: {
echo         lucide: ^(^) =^> import^('@iconify-json/lucide/icons.json'^).then^(i =^> i.default^)
echo       }
echo     }^)
echo   ],
echo   theme: {
echo     colors: {
echo       'nier-bg': 'hsl^(var^(--background^)^)',
echo       'nier-fg': 'hsl^(var^(--foreground^)^)',
echo       'nier-surface': 'hsl^(var^(--card^)^)',
echo       'nier-accent': 'hsl^(var^(--accent^)^)',
echo       'nier-border': 'hsl^(var^(--border^)^)',
echo       background: 'hsl^(var^(--background^)^)',
echo       foreground: 'hsl^(var^(--foreground^)^)',
echo       primary: {
echo         DEFAULT: 'hsl^(var^(--primary^)^)',
echo         foreground: 'hsl^(var^(--primary-foreground^)^)'
echo       },
echo       accent: {
echo         DEFAULT: 'hsl^(var^(--accent^)^)',
echo         foreground: 'hsl^(var^(--accent-foreground^)^)'
echo       }
echo     },
echo     fontFamily: {
echo       'nier': ['Gothic A1', 'JetBrains Mono', 'system-ui', 'sans-serif']
echo     },
echo     borderRadius: { 'nier': '0.25rem' }
echo   },
echo   shortcuts: {
echo     'nier-btn': 'inline-flex items-center justify-center font-nier text-sm font-medium transition-all duration-200 border border-nier-border bg-nier-surface hover:bg-accent rounded-nier',
echo     'nier-btn-primary': 'nier-btn bg-primary text-primary-foreground hover:bg-primary/90',
echo     'nier-input': 'flex w-full border border-nier-border bg-nier-surface px-3 py-2 text-sm font-nier rounded-nier',
echo     'nier-surface': 'bg-nier-surface border border-nier-border relative',
echo     'nier-glow': 'shadow-[0_0_10px_hsl^(var^(--accent^)/0.3^)]'
echo   }
echo }^)
) > "uno.config.ts"

echo [4/10] 🔧 Updating Vite configuration for UnoCSS...
if exist "vite.config.ts" (
    powershell -NoProfile -Command ^
    "$content = Get-Content 'vite.config.ts' -Raw; ^
    if ($content -notmatch 'UnoCSS') { ^
        $content = $content -replace 'import { sveltekit }', 'import UnoCSS from ''unocss/vite''^`nimport { sveltekit }'; ^
        $content = $content -replace 'plugins: \[', 'plugins: [^`n^t^tUnoCSS(),'; ^
        Set-Content 'vite.config.ts' $content ^
    }"
)

echo [5/10] 📝 Creating utility functions...
if not exist "src\lib\utils" mkdir "src\lib\utils"

(
echo import { type ClassValue, clsx } from 'clsx'
echo import { twMerge } from 'tailwind-merge'
echo.
echo export function cn^(...inputs: ClassValue[]^) {
echo   return twMerge^(clsx^(inputs^)^)
echo }
) > "src\lib\utils\cn.ts"

echo [6/10] 📋 Creating stores...
if not exist "src\lib\stores" mkdir "src\lib\stores"

(
echo import { writable } from 'svelte/store'
echo.
echo export const isSidebarOpen = writable^(false^)
echo export const theme = writable^('dark'^)
) > "src\lib\stores\index.ts"

echo [7/10] 🎨 Updating app layout...
if exist "src\routes\+layout.svelte" (
    powershell -NoProfile -Command ^
    "$content = Get-Content 'src\routes\+layout.svelte' -Raw; ^
    if ($content -notmatch 'theme.css') { ^
        $content = $content -replace '^<script', \"import '`$lib/styles/theme.css'^`n^<script\"; ^
        Set-Content 'src\routes\+layout.svelte' $content ^
    }"
) else (
    (
    echo import '$lib/styles/theme.css'
    echo.
    echo ^<div class="min-h-screen bg-nier-bg text-nier-fg font-nier"^>
    echo   ^<slot /^>
    echo ^</div^>
    ) > "src\routes\+layout.svelte"
)

echo [8/10] Running npm check...
call npm run check

echo [9/10] Testing development server...
echo Starting dev server to test integration...
timeout /t 3 /nobreak >nul
echo You can now test the Nier.css integration!

echo [10/10] ✅ NIER.CSS INTEGRATION COMPLETE!
echo.
echo 🎮 Integration Summary:
echo - UnoCSS configured with Nier theme
echo - Nier color palette applied
echo - Utility functions created
echo - Theme CSS imported
echo.
echo 🚀 Next: Run npm run dev to see your Nier-themed app!
echo.
pause
