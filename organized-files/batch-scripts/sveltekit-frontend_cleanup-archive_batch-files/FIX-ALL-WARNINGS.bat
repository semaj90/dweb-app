@echo off
echo 🔧 Fixing All CSS and Import Errors...
echo =====================================

echo.
echo 📋 Step 1: Fixing package.json duplicates...
node fix-duplicates.mjs

echo.
echo 📋 Step 2: Fixing KeyboardShortcuts import...
powershell -Command "(Get-Content 'src\lib\components\keyboard\KeyboardShortcuts.svelte') -replace 'import type \{ User \} from ''(\$lib/types/user)'';', '' -replace '(<script[^>]*>)', '$1`nimport type { User } from ''$lib/types/user'';' | Set-Content 'src\lib\components\keyboard\KeyboardShortcuts.svelte'"

echo.
echo 📋 Step 3: Removing unused CSS selectors...
powershell -Command "
$files = @(
  'src\routes\+layout.svelte',
  'src\lib\components\keyboard\KeyboardShortcuts.svelte'
)

$selectors = @(
  '\[data-theme=""dark""\]\s\.nav\s*\{[^}]*\}',
  '\[data-theme=""dark""\]\s\.brand-text\s*\{[^}]*\}',
  '\[data-theme=""dark""\]\s\.mobile-menu\s*\{[^}]*\}',
  '\.close-button\s*\{[^}]*\}',
  '\.commands-list\s*\{[^}]*\}',
  '\.no-results\s*\{[^}]*\}',
  '\.no-results p\s*\{[^}]*\}',
  '\.shortcuts-hint\s*\{[^}]*\}',
  '\.shortcuts-hint kbd\s*\{[^}]*\}',
  '\.shortcuts-help\.hidden\s*\{[^}]*\}'
)

foreach ($file in $files) {
  if (Test-Path $file) {
    $content = Get-Content $file -Raw
    foreach ($selector in $selectors) {
      $content = $content -replace $selector, ''
    }
    Set-Content $file $content
    Write-Host \"✅ Cleaned $file\"
  }
}
"

echo.
echo 📋 Step 4: Fixing ErrorBoundary export...
powershell -Command "(Get-Content 'src\lib\components\ErrorBoundary.svelte') -replace 'export let maxWidth', 'export const maxWidth' | Set-Content 'src\lib\components\ErrorBoundary.svelte'"

echo.
echo ✅ ALL FIXES APPLIED!
echo ====================
echo ✅ Package.json duplicates removed
echo ✅ KeyboardShortcuts import fixed
echo ✅ Unused CSS selectors removed
echo ✅ ErrorBoundary export fixed
echo.
echo 🚀 Restart dev server: npm run dev
pause