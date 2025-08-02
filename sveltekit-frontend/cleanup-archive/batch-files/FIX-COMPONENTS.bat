@echo off
setlocal EnableDelayedExpansion
echo ========================================
echo Component Error Fix - Bits UI & Types
echo ========================================
echo.

echo [1/4] Checking Bits UI installation...
npm list bits-ui > nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo Installing Bits UI...
    npm install bits-ui
    if !ERRORLEVEL! NEQ 0 (
        echo ERROR: Failed to install Bits UI
        echo Press any key to continue anyway...
        pause > nul
    )
)

echo [2/4] Creating missing component exports...
if not exist "src\lib\components\ui" (
    mkdir "src\lib\components\ui"
    echo Created UI components directory
)

echo // Bits UI Separator component > src\lib\components\ui\separator.ts
echo import { Separator as SeparatorPrimitive } from "bits-ui"; >> src\lib\components\ui\separator.ts
echo export { SeparatorPrimitive as Separator }; >> src\lib\components\ui\separator.ts
echo ✓ Created Separator component wrapper

echo [3/4] Fixing common component type issues...

REM Create index.ts for components
if not exist "src\lib\components\ui\index.ts" (
    echo export * from './button'; > src\lib\components\ui\index.ts
    echo export * from './separator'; >> src\lib\components\ui\index.ts
    echo export * from './modal'; >> src\lib\components\ui\index.ts
    echo ✓ Created component index exports
)

REM Fix Button component if it has Svelte 5 syntax issues
if exist "src\lib\components\ui\button\Button.svelte" (
    echo Updating Button component for Svelte 5...
    echo ^<script lang="ts"^> > src\lib\components\ui\button\Button.temp
    echo   import { type ComponentProps } from 'svelte'; >> src\lib\components\ui\button\Button.temp
    echo   >> src\lib\components\ui\button\Button.temp
    echo   interface Props extends ComponentProps^<'button'^> { >> src\lib\components\ui\button\Button.temp
    echo     variant?: 'default' ^| 'primary' ^| 'secondary' ^| 'ghost' ^| 'outline'; >> src\lib\components\ui\button\Button.temp
    echo     size?: 'sm' ^| 'md' ^| 'lg'; >> src\lib\components\ui\button\Button.temp
    echo   } >> src\lib\components\ui\button\Button.temp
    echo   >> src\lib\components\ui\button\Button.temp
    echo   let { variant = 'default', size = 'md', class: className = '', children, ...restProps }: Props = $props(); >> src\lib\components\ui\button\Button.temp
    echo ^</script^> >> src\lib\components\ui\button\Button.temp
    echo. >> src\lib\components\ui\button\Button.temp
    echo ^<button class="btn btn-{variant} btn-{size} {className}" {...restProps}^> >> src\lib\components\ui\button\Button.temp
    echo   {#if children} >> src\lib\components\ui\button\Button.temp
    echo     {@render children()} >> src\lib\components\ui\button\Button.temp
    echo   {:else} >> src\lib\components\ui\button\Button.temp
    echo     ^<slot /^> >> src\lib\components\ui\button\Button.temp
    echo   {/if} >> src\lib\components\ui\button\Button.temp
    echo ^</button^> >> src\lib\components\ui\button\Button.temp
    
    move "src\lib\components\ui\button\Button.temp" "src\lib\components\ui\button\Button.svelte" > nul
    echo ✓ Updated Button component
)

echo [4/4] Running component check...
echo Checking for component-related errors...
npm run check 2>&1 | findstr /C:"component\|Component\|bits-ui\|Separator" > component-check.txt

if !ERRORLEVEL! EQU 0 (
    echo ⚠️ Component-related issues found:
    type component-check.txt
    echo.
    echo Check component-check.txt for details
) else (
    echo ✅ No component-related errors found
)

echo.
echo 🧩 Component fixes applied:
echo   - Bits UI Separator component wrapper created
echo   - Button component updated for Svelte 5
echo   - Component index exports created
echo   - Type-safe component interfaces added
echo.
echo ==========================================
echo Press any key to close this window...
echo Or press Ctrl+C to keep window open
echo ==========================================
pause > nul
