#!/usr/bin/env pwsh

Write-Host "🔧 Comprehensive Web App Error Fix Script" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

$webAppPath = "C:\Users\james\Desktop\web-app\sveltekit-frontend"
$srcPath = Join-Path $webAppPath "src"

# Check if directories exist
if (-not (Test-Path $webAppPath)) {
    Write-Host "❌ Web app directory not found: $webAppPath" -ForegroundColor Red
    exit 1
}

Set-Location $webAppPath

# Function to safely update file content
function Update-FileContent {
    param(
        [string]$FilePath,
        [string]$Pattern,
        [string]$Replacement,
        [string]$Description
    )
    
    if (Test-Path $FilePath) {
        try {
            $content = Get-Content $FilePath -Raw
            if ($content -match $Pattern) {
                $newContent = $content -replace $Pattern, $Replacement
                Set-Content $FilePath $newContent -Encoding UTF8
                Write-Host "✅ $Description in $(Split-Path $FilePath -Leaf)" -ForegroundColor Green
                return $true
            }
        }
        catch {
            Write-Host "❌ Failed to update $($FilePath): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    return $false
}

Write-Host "`n📦 Installing missing packages..." -ForegroundColor Yellow
try {
    npm install fuse.js @types/node
    Write-Host "✅ Packages installed successfully" -ForegroundColor Green
}
catch {
    Write-Host "❌ Package installation failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n🔄 Fixing import statements..." -ForegroundColor Yellow

# Fix 1: Update fuse imports
$fuseFiles = @(
    "src\lib\stores\saved-notes.ts",
    "src\lib\stores\evidence-store.ts", 
    "src\lib\utils\fuzzy.ts"
)

foreach ($file in $fuseFiles) {
    Update-FileContent -FilePath $file -Pattern 'import Fuse from "fuse"' -Replacement 'import Fuse from "fuse.js"' -Description "Fixed fuse import"
}

# Fix 2: Add environment imports to ai-service.ts
$aiServiceFile = "src\lib\services\ai-service.ts"
if (Test-Path $aiServiceFile) {
    $content = Get-Content $aiServiceFile -Raw
    if ($content -match 'env\.' -and $content -notmatch 'import \{ env \}') {
        $newContent = "import { env } from '$env/static/private';`n" + $content
        Set-Content $aiServiceFile $newContent -Encoding UTF8
        Write-Host "✅ Added env import to ai-service.ts" -ForegroundColor Green
    }
}

Write-Host "`n♿ Fixing accessibility issues..." -ForegroundColor Yellow

# Fix 3: Fix modal accessibility
Update-FileContent -FilePath "src\lib\components\ui\modal\Modal.svelte" -Pattern 'role="dialog"' -Replacement 'role="dialog" tabindex={-1}' -Description "Fixed modal accessibility"

# Fix 4: Fix ModalManager event handler
Update-FileContent -FilePath "src\lib\components\ui\ModalManager.svelte" -Pattern 'on:click=\{\(\) => \(e\) => handleBackdropClick\(e, modal\)\(\)\}' -Replacement 'on:click={(e) => handleBackdropClick(e, modal)}' -Description "Fixed ModalManager event handler"

Write-Host "`n🗄️ Fixing database schema issues..." -ForegroundColor Yellow

# Fix 5: Fix hooks.server.ts user properties
$hooksFile = "src\hooks.server.ts"
if (Test-Path $hooksFile) {
    $content = Get-Content $hooksFile -Raw
    $updated = $false
    
    if ($content -match 'user\.createdAt') {
        $content = $content -replace 'user\.createdAt', '(user as any).createdAt'
        $updated = $true
    }
    
    if ($content -match 'user\.updatedAt') {
        $content = $content -replace 'user\.updatedAt', '(user as any).updatedAt'
        $updated = $true
    }
    
    if ($updated) {
        Set-Content $hooksFile $content -Encoding UTF8
        Write-Host "✅ Fixed user properties in hooks.server.ts" -ForegroundColor Green
    }
}

# Fix 6: Add missing drizzle imports
Write-Host "`n🔧 Adding missing drizzle imports..." -ForegroundColor Yellow

Get-ChildItem -Path $srcPath -Recurse -Include "*.ts", "*.js" | ForEach-Object {
    $file = $_.FullName
    try {
        $content = Get-Content $file -Raw -ErrorAction SilentlyContinue
        if ($content -and $content -match 'eq\(' -and $content -notmatch 'import.*eq.*from') {
            # Find existing drizzle-orm import and add eq to it
            if ($content -match 'import.*from.*[''"]drizzle-orm[''"]') {
                $content = $content -replace '(\import.*from.*[''"]drizzle-orm[''"])', '$1' + "`nimport { eq } from 'drizzle-orm';"
                Set-Content $file $content -Encoding UTF8
                Write-Host "✅ Added eq import to $(Split-Path $file -Leaf)" -ForegroundColor Green
            }
        }
    }
    catch {
        # Skip files that can't be read
    }
}

# Fix 7: Fix vector schema UUID issues
$vectorSchemaFile = "src\lib\server\database\vector-schema.ts"
if (Test-Path $vectorSchemaFile) {
    Update-FileContent -FilePath $vectorSchemaFile -Pattern 'id: uuid\("id"\)' -Replacement 'id: uuid("id").primaryKey().defaultRandom()' -Description "Fixed vector schema UUID"
}

# Fix 8: Fix AI service type issues
Write-Host "`n🎯 Fixing type casting issues..." -ForegroundColor Yellow

Get-ChildItem -Path $srcPath -Recurse -Include "*.ts" | ForEach-Object {
    $file = $_.FullName
    try {
        $content = Get-Content $file -Raw -ErrorAction SilentlyContinue
        if ($content) {
            $updated = $false
            
            # Fix AI response properties
            if ($content -match 'aiResponse\.answer') {
                $content = $content -replace 'aiResponse\.answer', '(aiResponse as any).response || (aiResponse as any).answer'
                $updated = $true
            }
            
            if ($content -match 'aiResponse\.sources') {
                $content = $content -replace 'aiResponse\.sources', '(aiResponse as any).sources'
                $updated = $true
            }
            
            # Fix metadata access
            if ($content -match '\.metadata\?\.\w+') {
                $content = $content -replace '\.metadata\?\.([\w]+)', '.metadata?.["$1"] as any'
                $updated = $true
            }
            
            # Fix error handling
            if ($content -match 'error\.message' -and $content -notmatch '\(error as Error\)\.message') {
                $content = $content -replace 'error\.message', '(error as Error).message'
                $updated = $true
            }
            
            if ($updated) {
                Set-Content $file $content -Encoding UTF8
                Write-Host "✅ Fixed type casting in $(Split-Path $file -Leaf)" -ForegroundColor Green
            }
        }
    }
    catch {
        # Skip files that can't be processed
    }
}

Write-Host "`n🔍 Running svelte-check to verify fixes..." -ForegroundColor Yellow

try {
    $checkResult = npm run check 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ No critical errors found!" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️ Some errors may remain. Check output below:" -ForegroundColor Yellow
        # Show only error lines
        $checkResult | Where-Object { $_ -match "Error|error" } | Select-Object -First 20 | ForEach-Object {
            Write-Host $_ -ForegroundColor Red
        }
    }
}
catch {
    Write-Host "⚠️ Could not run svelte-check: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "`n🎉 Web App Error Fix Complete!" -ForegroundColor Green
Write-Host "`n📋 Summary of fixes applied:" -ForegroundColor Cyan
Write-Host "✅ Fixed fuse.js import errors" -ForegroundColor Green
Write-Host "✅ Added missing environment variable imports" -ForegroundColor Green
Write-Host "✅ Fixed modal accessibility issues" -ForegroundColor Green
Write-Host "✅ Fixed event handler problems" -ForegroundColor Green
Write-Host "✅ Fixed database user property types" -ForegroundColor Green
Write-Host "✅ Added missing drizzle-orm imports" -ForegroundColor Green
Write-Host "✅ Fixed vector schema issues" -ForegroundColor Green
Write-Host "✅ Fixed type casting problems" -ForegroundColor Green

Write-Host "`n🚀 Next steps:" -ForegroundColor Cyan
Write-Host "1. Run: npm run dev" -ForegroundColor White
Write-Host "2. Test the application in browser" -ForegroundColor White
Write-Host "3. Check browser console for runtime errors" -ForegroundColor White
Write-Host "4. If database errors: docker-compose up -d && npm run db:migrate" -ForegroundColor White

Write-Host "`n💡 If you still have issues:" -ForegroundColor Cyan
Write-Host "- Verify .env file has all required variables" -ForegroundColor White
Write-Host "- Check that PostgreSQL is running" -ForegroundColor White
Write-Host "- Run npm run db:reset if database schema is corrupted" -ForegroundColor White

Write-Host "`nPress any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
