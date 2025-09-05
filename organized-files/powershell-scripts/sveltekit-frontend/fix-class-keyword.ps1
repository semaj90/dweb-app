# PowerShell script to fix the class: class JavaScript parse error in Svelte files
# This replaces class: class with class: className = '' and updates template usage

$files = Get-ChildItem -Path "src" -Name "*.svelte" -Recurse | ForEach-Object { "src\$_" }
$fixedCount = 0
$errorFiles = @()

Write-Host "Starting to fix class: class pattern in Svelte files..." -ForegroundColor Green
Write-Host "Found $($files.Count) Svelte files to process" -ForegroundColor Cyan

foreach ($file in $files) {
    try {
        $content = Get-Content $file -Raw
        $originalContent = $content
        
        # Fix the destructuring pattern: class: class -> class: className = ''
        $content = $content -replace 'class:\s*class\b', "class: className = ''"
        
        # Fix template usage: {class || ''} -> {className}
        $content = $content -replace '\{class\s*\|\|\s*["'']?["'']?\}', '{className}'
        
        # Fix template usage: {class} -> {className}
        $content = $content -replace '\{class\}', '{className}'
        
        # Fix concatenation patterns: " + class" -> " + className"
        $content = $content -replace '"\s*\+\s*class\b', '" + className'
        
        # Fix other common class variable usage patterns in template literals
        $content = $content -replace '\$\{class\}', '${className}'
        
        if ($content -ne $originalContent) {
            Set-Content $file -Value $content -NoNewline
            $fixedCount++
            Write-Host "Fixed: $file" -ForegroundColor Yellow
        }
    }
    catch {
        $errorFiles += $file
        Write-Host "Error processing ${file}: $_" -ForegroundColor Red
    }
}

Write-Host "`nProcessing complete!" -ForegroundColor Green
Write-Host "Files processed: $($files.Count)" -ForegroundColor Cyan
Write-Host "Files fixed: $fixedCount" -ForegroundColor Green

if ($errorFiles.Count -gt 0) {
    Write-Host "Files with errors: $($errorFiles.Count)" -ForegroundColor Red
    $errorFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
}

Write-Host "`nRunning npm run check to verify fixes..." -ForegroundColor Cyan
npm run check