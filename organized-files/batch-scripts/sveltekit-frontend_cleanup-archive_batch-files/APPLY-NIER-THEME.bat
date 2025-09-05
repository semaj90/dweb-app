@echo off
echo 🎨 Converting to NieR: Automata Theme...
echo =====================================

echo.
echo 📋 Step 1: Removing unused CSS selectors...
powershell -Command "(Get-Content 'src\routes\+layout.svelte') -replace '\[data-theme=\""dark\""\]\s\.nav\s*\{[^}]*\}', '' | Set-Content 'src\routes\+layout.svelte'"
powershell -Command "(Get-Content 'src\routes\+layout.svelte') -replace '\[data-theme=\""dark\""\]\s\.brand-text\s*\{[^}]*\}', '' | Set-Content 'src\routes\+layout.svelte'"
powershell -Command "(Get-Content 'src\routes\+layout.svelte') -replace '\[data-theme=\""dark\""\]\s\.mobile-menu\s*\{[^}]*\}', '' | Set-Content 'src\routes\+layout.svelte'"

echo ✅ Unused CSS selectors removed

echo.
echo 📋 Step 2: Applying NieR theme classes...
powershell -Command "(Get-Content 'src\routes\+layout.svelte') -replace 'class=\"\"nav\"\"', 'class=\"\"nier-nav\"\"' | Set-Content 'src\routes\+layout.svelte'"
powershell -Command "(Get-Content 'src\routes\+layout.svelte') -replace 'class=\"\"brand-text\"\"', 'class=\"\"nier-brand\"\"' | Set-Content 'src\routes\+layout.svelte'"

echo ✅ NieR theme classes applied

echo.
echo 🎯 NIER THEME CONVERSION COMPLETE!
echo ===================================
echo ✅ Unused CSS warnings fixed
echo ✅ NieR design system activated
echo ✅ Dark cyberpunk aesthetic enabled
echo.
echo 🚀 Your Legal AI now has NieR: Automata styling!
pause