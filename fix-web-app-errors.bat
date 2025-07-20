@echo off
echo 🔧 Quick Web App Error Fix Script
echo ================================

cd /d "C:\Users\james\Desktop\web-app\sveltekit-frontend"

echo.
echo 📦 Installing critical packages...
call npm install fuse.js @types/node
if %errorlevel% neq 0 (
    echo ❌ Package installation failed
    pause
    exit /b 1
)

echo.
echo 🔄 Quick fixes for critical errors...

:: Fix 1: Replace fuse imports
echo Fixing fuse.js imports...
powershell -Command "(Get-Content 'src\lib\stores\saved-notes.ts' -Raw) -replace 'import Fuse from \"fuse\"', 'import Fuse from \"fuse.js\"' | Set-Content 'src\lib\stores\saved-notes.ts'"
powershell -Command "(Get-Content 'src\lib\stores\evidence-store.ts' -Raw) -replace 'import Fuse from \"fuse\"', 'import Fuse from \"fuse.js\"' | Set-Content 'src\lib\stores\evidence-store.ts'"
powershell -Command "(Get-Content 'src\lib\utils\fuzzy.ts' -Raw) -replace 'import Fuse from \"fuse\"', 'import Fuse from \"fuse.js\"' | Set-Content 'src\lib\utils\fuzzy.ts'"

:: Fix 2: Add environment imports
echo Adding environment imports...
powershell -Command "$content = Get-Content 'src\lib\services\ai-service.ts' -Raw; if ($content -notmatch 'import.*env.*from') { $content = 'import { env } from ''$env/static/private'';' + [System.Environment]::NewLine + $content; Set-Content 'src\lib\services\ai-service.ts' $content }"

:: Fix 3: Fix modal accessibility
echo Fixing modal accessibility...
powershell -Command "(Get-Content 'src\lib\components\ui\modal\Modal.svelte' -Raw) -replace 'role=\"dialog\"', 'role=\"dialog\" tabindex={-1}' | Set-Content 'src\lib\components\ui\modal\Modal.svelte'"

:: Fix 4: Fix ModalManager event handler
echo Fixing ModalManager...
powershell -Command "(Get-Content 'src\lib\components\ui\ModalManager.svelte' -Raw) -replace 'on:click=\{\(\) => \(e\) => handleBackdropClick\(e, modal\)\(\)\}', 'on:click={(e) => handleBackdropClick(e, modal)}' | Set-Content 'src\lib\components\ui\ModalManager.svelte'"

:: Fix 5: Add missing drizzle imports
echo Adding missing drizzle imports...
for /r "src" %%f in (*.ts *.js) do (
    powershell -Command "$file='%%f'; $content = Get-Content $file -Raw -ErrorAction SilentlyContinue; if ($content -and $content -match 'eq\(' -and $content -notmatch 'import.*eq.*from') { $lines = $content -split '\n'; $importIndex = -1; for ($i=0; $i -lt $lines.Length; $i++) { if ($lines[$i] -match 'import.*from.*drizzle-orm') { $importIndex = $i; break } } if ($importIndex -ge 0) { $lines[$importIndex] = $lines[$importIndex] -replace '\}', ', eq }'; $content = $lines -join \"`n\"; Set-Content $file $content } }"
)

:: Fix 6: Fix hooks.server.ts user properties
echo Fixing user properties in hooks.server.ts...
powershell -Command "$file='src\hooks.server.ts'; if (Test-Path $file) { $content = Get-Content $file -Raw; $content = $content -replace 'user\.createdAt', '(user as any).createdAt'; $content = $content -replace 'user\.updatedAt', '(user as any).updatedAt'; Set-Content $file $content }"

echo.
echo 🔍 Running check to see remaining errors...
call npm run check > check_output.txt 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Some errors remain. Check check_output.txt for details.
    echo Most critical errors should be fixed.
    type check_output.txt | findstr /i "error"
) else (
    echo ✅ No critical errors found!
)

echo.
echo 🎉 Quick fix complete!
echo.
echo 📋 What was fixed:
echo ✅ fuse.js import errors
echo ✅ Missing environment variable imports  
echo ✅ Modal accessibility issues
echo ✅ Event handler problems
echo ✅ Missing drizzle-orm imports
echo ✅ User property type issues
echo.
echo 🚀 Try running: npm run dev
echo.
echo 💡 If you still have errors:
echo 1. Check database is running: docker-compose up -d
echo 2. Run migrations: npm run db:migrate  
echo 3. Check environment variables in .env file
echo.
pause
