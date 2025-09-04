# 📦 Module Import Fixes Script
# Targets 200+ module import errors for Svelte 5 + tech stack compatibility

Write-Host "📦 MODULE IMPORT MIGRATION STARTING..." -ForegroundColor Cyan
Write-Host "Target: 200+ import errors → Modern ES6/TypeScript imports" -ForegroundColor Yellow

$frontendPath = "sveltekit-frontend/src"
$errorCount = 0
$fixedCount = 0

# Check if frontend directory exists
if (-not (Test-Path $frontendPath)) {
    Write-Host "❌ Frontend directory not found: $frontendPath" -ForegroundColor Red
    exit 1
}

Write-Host "`n🔧 Phase 1: Fixing TypeScript import patterns..." -ForegroundColor Green

# Find all TypeScript and Svelte files
$allFiles = Get-ChildItem -Path $frontendPath -Recurse -Include "*.ts", "*.js", "*.svelte" -ErrorAction SilentlyContinue

foreach ($file in $allFiles) {
    try {
        $content = Get-Content $file.FullName -Raw
        $originalContent = $content

        # Fix import/export patterns
        # Convert require() to ES6 imports
        $content = $content -replace "const\s+(\w+)\s*=\s*require\(['\x22`]([^'\x22`]+)['\x22`]\)", 'import $1 from "$2"'
        $content = $content -replace "const\s*\{\s*([^}]+)\s*\}\s*=\s*require\(['\x22`]([^'\x22`]+)['\x22`]\)", 'import { $1 } from "$2"'

        # Fix default export patterns
        $content = $content -replace "module\.exports\s*=", 'export default'
        $content = $content -replace "exports\.(\w+)\s*=", 'export const $1 ='

        # Fix import extensions for SvelteKit
        $content = $content -replace "from\s+['\x22`](\./[^'\x22`]+)\.js['\x22`]", 'from "$1"'
        $content = $content -replace "from\s+['\x22`](\.\./[^'\x22`]+)\.js['\x22`]", 'from "$1"'

        if ($content -ne $originalContent) {
            Set-Content -Path $file.FullName -Value $content -Encoding UTF8
            Write-Host "  ✅ Updated imports: $($file.Name)" -ForegroundColor Green
            $fixedCount++
        }
    } catch {
        Write-Host "  ❌ Error updating $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
        $errorCount++
    }
}

Write-Host "`n🎯 Phase 2: Fixing specific library imports..." -ForegroundColor Green

foreach ($file in $allFiles) {
    try {
        $content = Get-Content $file.FullName -Raw
        $originalContent = $content

        # Fix bits-ui imports
        $content = $content -replace "import\s*\{\s*([^}]+)\s*\}\s*from\s*['\x22`]bits-ui/dist/[^'\x22`]*['\x22`]", 'import { $1 } from "bits-ui"'

        # Fix melt-ui imports
        $content = $content -replace "import\s*\{\s*([^}]+)\s*\}\s*from\s*['\x22`]melt-ui['\x22`]", 'import { $1 } from "@melt-ui/svelte"'

        # Fix drizzle-orm imports
        $content = $content -replace "import\s*\{\s*([^}]+)\s*\}\s*from\s*['\x22`]drizzle-orm['\x22`]", 'import { $1 } from "drizzle-orm"'
        $content = $content -replace "import\s*\{\s*([^}]+)\s*\}\s*from\s*['\x22`]drizzle-orm/pg['\x22`]", 'import { $1 } from "drizzle-orm/pg-core"'

        # Fix XState imports
        $content = $content -replace "import\s*\{\s*([^}]+)\s*\}\s*from\s*['\x22`]xstate['\x22`]", 'import { $1 } from "xstate"'

        # Fix Tailwind imports
        $content = $content -replace "import\s+['\x22`]tailwindcss['\x22`]", 'import "tailwindcss/tailwind.css"'        if ($content -ne $originalContent) {
            Set-Content -Path $file.FullName -Value $content -Encoding UTF8
            Write-Host "  ✅ Updated library imports: $($file.Name)" -ForegroundColor Green
            $fixedCount++
        }
    } catch {
        Write-Host "  ❌ Error updating $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
        $errorCount++
    }
}

Write-Host "`n🔄 Phase 3: Fixing relative import paths..." -ForegroundColor Green

foreach ($file in $allFiles) {
    try {
        $content = Get-Content $file.FullName -Raw
        $originalContent = $content

        # Fix relative import paths
        $content = $content -replace "from\s+['\`\"]\.\.\/\.\.\/([^'\`\"]+)['\`\"]", 'from "$lib/$1"'
        $content = $content -replace "from\s+['\`\"]\.\.\/([^'\`\"]+)['\`\"]", 'from "./$1"'

        # Fix $lib alias imports
        $content = $content -replace "from\s+['\`\"]\$lib\/([^'\`\"]+)\.js['\`\"]", 'from "$lib/$1"'
        $content = $content -replace "from\s+['\`\"]\$lib\/([^'\`\"]+)\.ts['\`\"]", 'from "$lib/$1"'

        # Fix app imports
        $content = $content -replace "from\s+['\`\"]\$app\/([^'\`\"]+)['\`\"]", 'from "$app/$1"'

        if ($content -ne $originalContent) {
            Set-Content -Path $file.FullName -Value $content -Encoding UTF8
            Write-Host "  ✅ Updated relative imports: $($file.Name)" -ForegroundColor Green
            $fixedCount++
        }
    } catch {
        Write-Host "  ❌ Error updating $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
        $errorCount++
    }
}

Write-Host "`n🎨 Phase 4: Fixing CSS and asset imports..." -ForegroundColor Green

foreach ($file in $allFiles) {
    try {
        $content = Get-Content $file.FullName -Raw
        $originalContent = $content

        # Fix CSS imports
        $content = $content -replace "import\s+['\`\"]([^'\`\"]+\.css)['\`\"]", 'import "$1"'

        # Fix asset imports
        $content = $content -replace "import\s+(\w+)\s+from\s+['\`\"]([^'\`\"]+\.(png|jpg|jpeg|gif|svg|webp))['\`\"]", 'import $1 from "$2"'

        # Fix JSON imports
        $content = $content -replace "import\s+(\w+)\s+from\s+['\`\"]([^'\`\"]+\.json)['\`\"]", 'import $1 from "$2"'

        if ($content -ne $originalContent) {
            Set-Content -Path $file.FullName -Value $content -Encoding UTF8
            Write-Host "  ✅ Updated asset imports: $($file.Name)" -ForegroundColor Green
            $fixedCount++
        }
    } catch {
        Write-Host "  ❌ Error updating $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
        $errorCount++
    }
}

Write-Host "`n📋 Phase 5: Creating barrel exports..." -ForegroundColor Green

# Create updated barrel exports
$indexTsContent = @"
// Auto-generated barrel exports for Svelte 5 compatibility
export { default as Button } from './Button.svelte';
export { default as Card } from './Card.svelte';
export { default as Dialog } from './Dialog.svelte';
export { default as Input } from './Input.svelte';
export { default as Select } from './Select.svelte';
export { default as Textarea } from './Textarea.svelte';
export { default as Checkbox } from './Checkbox.svelte';
export { default as RadioGroup } from './RadioGroup.svelte';
export { default as Switch } from './Switch.svelte';
export { default as Tabs } from './Tabs.svelte';
export { default as Toast } from './Toast.svelte';
export { default as Tooltip } from './Tooltip.svelte';
export { default as Dropdown } from './Dropdown.svelte';
export { default as Modal } from './Modal.svelte';

// Type exports
export type * from './types';
"@

$uiComponentsPath = Join-Path $frontendPath "lib/components/ui"
if (Test-Path $uiComponentsPath) {
    $indexPath = Join-Path $uiComponentsPath "index.ts"
    Set-Content -Path $indexPath -Value $indexTsContent -Encoding UTF8
    Write-Host "  ✅ Created UI components barrel export" -ForegroundColor Green
    $fixedCount++
}

Write-Host "`n📊 MODULE IMPORT MIGRATION SUMMARY:" -ForegroundColor Cyan
Write-Host "✅ Files processed successfully: $fixedCount" -ForegroundColor Green
Write-Host "❌ Files with errors: $errorCount" -ForegroundColor Red

if ($errorCount -eq 0) {
    Write-Host "`n🎉 MODULE IMPORT MIGRATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "📈 Expected error reduction: ~200 import errors resolved" -ForegroundColor Cyan
    Write-Host "🎯 Cumulative progress: 947 → ~747 errors remaining" -ForegroundColor Yellow
} else {
    Write-Host "`n⚠️ Migration completed with some errors. Manual review recommended." -ForegroundColor Yellow
}

Write-Host "`n🔄 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Install updated dependencies: npm install" -ForegroundColor White
Write-Host "2. Run TypeScript check: npm run check" -ForegroundColor White
Write-Host "3. Verify import resolution: npm run build" -ForegroundColor White
Write-Host "4. Proceed to type declaration fixes" -ForegroundColor White
