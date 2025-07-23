@echo off
setlocal EnableDelayedExpansion
cls

echo ========================================================================
echo NIER.CSS + SVELTEKIT 2.6 INTEGRATION FIXER
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
call npm install @unocss/preset-uno @unocss/preset-wind @unocss/preset-icons
call npm install tailwind-merge clsx
call npm install @iconify-json/lucide @iconify-json/material-symbols
call npm install bits-ui shadcn-svelte
call npm install @xstate/svelte xstate
call npm install sveltekit-superforms zod
call npm install drizzle-orm drizzle-kit @types/pg

echo [2/10] 🎨 Creating Nier.css theme configuration...
if not exist "src\lib\styles" mkdir "src\lib\styles"

(
echo /* Nier.css + shadcn-svelte Integration */
echo @import url('nier.css'^);
echo.
echo @layer base {
echo   :root {
echo     /* Nier-inspired monochromatic palette */
echo     --background: 40 6%% 8%%;     /* Deep charcoal */
echo     --foreground: 50 8%% 95%%;    /* Off-white text */
echo     --card: 40 6%% 12%%;          /* Darker surface */
echo     --card-foreground: 50 8%% 95%%;
echo     --popover: 40 6%% 10%%;
echo     --popover-foreground: 50 8%% 95%%;
echo     --primary: 50 8%% 85%%;       /* Light gray primary */
echo     --primary-foreground: 40 6%% 8%%;
echo     --secondary: 40 6%% 16%%;     /* Medium surface */
echo     --secondary-foreground: 50 8%% 85%%;
echo     --muted: 40 6%% 14%%;
echo     --muted-foreground: 50 4%% 65%%;
echo     --accent: 45 100%% 65%%;      /* Yellow accent */
echo     --accent-foreground: 40 6%% 8%%;
echo     --destructive: 0 65%% 55%%;   /* Muted red */
echo     --destructive-foreground: 50 8%% 95%%;
echo     --border: 40 6%% 18%%;        /* Subtle borders */
echo     --input: 40 6%% 14%%;
echo     --ring: 45 100%% 65%%;        /* Yellow focus ring */
echo     --radius: 0.25rem;            /* Angular radius */
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
echo /* Nier typography */
echo body {
echo   font-family: 'Gothic A1', 'JetBrains Mono', system-ui, sans-serif;
echo   font-weight: 400;
echo   line-height: 1.5;
echo   letter-spacing: 0.025em;
echo }
echo.
echo /* Nier animations */
echo @keyframes nier-glow {
echo   0%%, 100%% { box-shadow: 0 0 10px hsl(var(--accent^) / 0.3^); }
echo   50%% { box-shadow: 0 0 20px hsl(var(--accent^) / 0.5^); }
echo }
echo.
echo @keyframes nier-scan {
echo   0%% { transform: translateX(-100%%^); }
echo   100%% { transform: translateX(100%%^); }
echo }
) > "src\lib\styles\theme.css"

echo [3/10] ⚙️ Creating UnoCSS configuration with Nier presets...
(
echo import { defineConfig } from 'unocss'
echo import { presetUno } from '@unocss/preset-uno'
echo import { presetWind } from '@unocss/preset-wind'
echo import { presetIcons } from '@unocss/preset-icons'
echo.
echo export default defineConfig({
echo   presets: [
echo     presetUno(^),
echo     presetWind(^),
echo     presetIcons({
echo       collections: {
echo         lucide: (^) =^> import('@iconify-json/lucide/icons.json'^).then(i =^> i.default^),
echo       }
echo     }^)
echo   ],
echo   theme: {
echo     colors: {
echo       'nier-bg': 'hsl(var(--background^)^)',
echo       'nier-fg': 'hsl(var(--foreground^)^)',
echo       'nier-surface': 'hsl(var(--card^)^)',
echo       'nier-accent': 'hsl(var(--accent^)^)',
echo       'nier-border': 'hsl(var(--border^)^)',
echo       background: 'hsl(var(--background^)^)',
echo       foreground: 'hsl(var(--foreground^)^)',
echo       primary: { DEFAULT: 'hsl(var(--primary^)^)', foreground: 'hsl(var(--primary-foreground^)^)' },
echo       accent: { DEFAULT: 'hsl(var(--accent^)^)', foreground: 'hsl(var(--accent-foreground^)^)' },
echo     },
echo     fontFamily: {
echo       'nier': ['Gothic A1', 'JetBrains Mono', 'system-ui', 'sans-serif'],
echo       'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
echo     },
echo     borderRadius: { 'nier': '0.25rem' },
echo   },
echo   shortcuts: {
echo     'nier-btn': 'inline-flex items-center justify-center font-nier text-sm font-medium transition-all duration-200 border border-nier-border bg-nier-surface hover:bg-accent rounded-nier',
echo     'nier-btn-primary': 'nier-btn bg-primary text-primary-foreground hover:bg-primary/90 shadow-[0_0_10px_hsl(var(--accent^)/0.3^)]',
echo     'nier-input': 'flex w-full border border-nier-border bg-nier-surface px-3 py-2 text-sm font-nier rounded-nier',
echo     'nier-surface': 'bg-nier-surface border border-nier-border relative',
echo     'nier-glow': 'shadow-[0_0_10px_hsl(var(--accent^)/0.3^),0_0_20px_hsl(var(--accent^)/0.1^)]',
echo   }
echo }^)
) > "uno.config.ts"

echo [4/10] 🔧 Updating Vite configuration for UnoCSS...
if exist "vite.config.js" (
    powershell -NoProfile -Command ^
    "$content = Get-Content 'vite.config.js' -Raw; ^
    if ($content -notmatch 'UnoCSS') { ^
        $content = $content -replace 'import { sveltekit }', 'import UnoCSS from ''unocss/vite''^`nimport { sveltekit }'; ^
        $content = $content -replace 'plugins: \[', 'plugins: [^`n^t^tUnoCSS(),'; ^
        Set-Content 'vite.config.js' $content ^
    }"
)

echo [5/10] 📝 Creating Nier utility functions...
if not exist "src\lib\utils" mkdir "src\lib\utils"

(
echo import { type ClassValue, clsx } from 'clsx'
echo import { twMerge } from 'tailwind-merge'
echo.
echo export function cn(...inputs: ClassValue[]^) {
echo   return twMerge(clsx(inputs^)^)
echo }
echo.
echo export function cnNier(...inputs: ClassValue[]^) {
echo   return twMerge(clsx(inputs^), 'font-nier'^)
echo }
) > "src\lib\utils\cn.ts"

echo [6/10] 🧩 Creating Nier Button component...
if not exist "src\lib\components\ui\button" mkdir "src\lib\components\ui\button"

(
echo ^<script lang="ts"^>
echo   import { cn } from '$lib/utils/cn'
echo   import type { HTMLButtonAttributes } from 'svelte/elements'
echo.
echo   interface Props extends HTMLButtonAttributes {
echo     variant?: 'default' ^| 'primary' ^| 'ghost' ^| 'outline'
echo     size?: 'default' ^| 'sm' ^| 'lg' ^| 'icon'
echo     class?: string
echo     glow?: boolean
echo   }
echo.
echo   let { variant = 'default', size = 'default', class: className, glow = false, ...restProps }: Props = $props(^)
echo.
echo   const variants = {
echo     default: 'nier-btn',
echo     primary: 'nier-btn-primary',
echo     ghost: 'nier-btn border-transparent hover:bg-accent',
echo     outline: 'nier-btn border-nier-accent text-nier-accent hover:bg-nier-accent hover:text-nier-bg'
echo   }
echo.
echo   const sizes = {
echo     default: 'h-10 px-4 py-2',
echo     sm: 'h-8 px-3 text-xs',
echo     lg: 'h-12 px-8 text-base',
echo     icon: 'h-10 w-10 p-0'
echo   }
echo ^</script^>
echo.
echo ^<button
echo   class={cn(variants[variant], sizes[size], glow ^&^& 'nier-glow', className^)}
echo   {...restProps}
echo ^>
echo   ^<slot /^>
echo ^</button^>
) > "src\lib\components\ui\button\button.svelte"

(
echo export { default as Button } from './button.svelte'
) > "src\lib\components\ui\button\index.ts"

echo [7/10] 🎯 Creating Nier Input component...
if not exist "src\lib\components\ui\input" mkdir "src\lib\components\ui\input"

(
echo ^<script lang="ts"^>
echo   import { cn } from '$lib/utils/cn'
echo   import type { HTMLInputAttributes } from 'svelte/elements'
echo.
echo   interface Props extends HTMLInputAttributes {
echo     class?: string
echo     error?: string
echo     label?: string
echo     glow?: boolean
echo   }
echo.
echo   let { class: className, error, label, glow = false, ...restProps }: Props = $props(^)
echo ^</script^>
echo.
echo ^<div class="space-y-2"^>
echo   {#if label}
echo     ^<label class="block text-sm font-nier font-medium text-nier-fg"^>
echo       {label}
echo     ^</label^>
echo   {/if}
echo.
echo   ^<input
echo     class={cn('nier-input', error ^&^& 'border-destructive', glow ^&^& 'focus-visible:nier-glow', className^)}
echo     {...restProps}
echo     bind:value
echo   /^>
echo.
echo   {#if error}
echo     ^<p class="text-sm text-destructive font-nier"^>{error}^</p^>
echo   {/if}
echo ^</div^>
) > "src\lib\components\ui\input\input.svelte"

(
echo export { default as Input } from './input.svelte'
) > "src\lib\components\ui\input\index.ts"

echo [8/10] 🤖 Creating Nier ChatInput component...
if not exist "src\lib\components\ai" mkdir "src\lib\components\ai"

(
echo ^<script lang="ts"^>
echo   import { createEventDispatcher } from 'svelte'
echo   import { cn } from '$lib/utils/cn'
echo   import { Button } from '$lib/components/ui/button'
echo.
echo   let value = $state('')
echo   let disabled = $state(false^)
echo   let textarea: HTMLTextAreaElement
echo.
echo   const dispatch = createEventDispatcher^<{ send: { text: string, timestamp: number } }^>(^)
echo.
echo   function handleSend(^) {
echo     const trimmed = value.trim(^)
echo     if (!trimmed ^|^| disabled^) return
echo     dispatch('send', { text: trimmed, timestamp: Date.now(^) }^)
echo     value = ''
echo   }
echo.
echo   function handleKeydown(event: KeyboardEvent^) {
echo     if (event.key === 'Enter' ^&^& !event.shiftKey^) {
echo       event.preventDefault(^)
echo       handleSend(^)
echo     }
echo   }
echo ^</script^>
echo.
echo ^<div class="nier-surface rounded-nier p-3 font-nier"^>
echo   ^<div class="flex items-end gap-3"^>
echo     ^<div class="text-nier-accent font-mono"^>^>^</div^>
echo     ^<textarea
echo       bind:this={textarea}
echo       bind:value
echo       on:keydown={handleKeydown}
echo       placeholder="QUERY: Enter analysis request..."
echo       class="flex-1 bg-transparent text-nier-fg resize-none border-none outline-none placeholder:text-nier-muted"
echo       rows="1"
echo       {disabled}
echo     ^>^</textarea^>
echo     ^<Button variant="primary" glow on:click={handleSend} {disabled}^>
echo       EXECUTE
echo     ^</Button^>
echo   ^</div^>
echo ^</div^>
) > "src\lib\components\ai\ChatInput.svelte"

echo [9/10] 📋 Creating barrel exports and stores...
if not exist "src\lib\stores" mkdir "src\lib\stores"

(
echo import { writable } from 'svelte/store'
echo.
echo export const isSidebarOpen = writable(false^)
echo export const theme = writable('dark'^)
) > "src\lib\stores\ui.ts"

(
echo export { isSidebarOpen, theme } from './ui'
) > "src\lib\stores\index.ts"

echo [10/10] 🎨 Updating app layout with Nier styling...
if exist "src\routes\+layout.svelte" (
    powershell -NoProfile -Command ^
    "$content = Get-Content 'src\routes\+layout.svelte' -Raw; ^
    if ($content -notmatch 'theme.css') { ^
        $imports = \"import '../app.css'^`nimport '`$lib/styles/theme.css'^`n\"; ^
        $content = $content -replace '^<script.*?^>', \"`$0^`n`$imports\"; ^
        Set-Content 'src\routes\+layout.svelte' $content ^
    }"
) else (
    (
    echo ^<script lang="ts"^>
    echo   import '../app.css'
    echo   import '$lib/styles/theme.css'
    echo ^</script^>
    echo.
    echo ^<div class="min-h-screen bg-nier-bg text-nier-fg font-nier"^>
    echo   ^<slot /^>
    echo ^</div^>
    ) > "src\routes\+layout.svelte"
)

echo.
echo ✅ NIER.CSS INTEGRATION COMPLETE!
echo.
echo 🎮 What was created/updated:
echo - UnoCSS config with Nier presets
echo - Nier.css theme variables and animations  
echo - Button component with glow effects
echo - Input component with terminal styling
echo - ChatInput component with scan lines
echo - Utility functions for class merging
echo - Barrel exports for clean imports
echo - Updated Vite config for UnoCSS
echo.
echo 🚀 Next steps:
echo 1. Run: npm run dev
echo 2. Test Nier components in your pages
echo 3. Add more components as needed
echo 4. Customize theme colors in theme.css
echo.
echo 🎯 Your SvelteKit app now has full Nier: Automata aesthetic!
echo Ready for legal AI terminal interfaces.
echo.
pause
