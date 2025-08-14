# Batch Fix Event Handlers - Systematic Svelte 5 Modernization
# Created: 2025-08-13 17:45:00
# Purpose: Fix remaining event handler deprecations across codebase

Write-Host "🚀 Starting Batch Event Handler Fixes..." -ForegroundColor Green
Write-Host "Target: Convert on:event → onevent patterns systematically" -ForegroundColor Yellow

$rootPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app"
$fixCount = 0

# Define event handler mappings
$eventMappings = @{
    "on:click" = "onclick"
    "on:change" = "onchange" 
    "on:input" = "oninput"
    "on:keydown" = "onkeydown"
    "on:keyup" = "onkeyup"
    "on:keypress" = "onkeypress"
    "on:submit" = "onsubmit"
    "on:focus" = "onfocus"
    "on:blur" = "onblur"
    "on:dragover" = "ondragover"
    "on:dragleave" = "ondragleave"
    "on:drop" = "ondrop"
    "on:contextmenu" = "oncontextmenu"
    "on:mousedown" = "onmousedown"
    "on:mouseup" = "onmouseup"
    "on:mouseover" = "onmouseover"
    "on:mouseout" = "onmouseout"
    "on:scroll" = "onscroll"
}

# Target file patterns (prioritize by impact)
$targetPatterns = @(
    "sveltekit-frontend\src\routes\**\*.svelte"
    "sveltekit-frontend\src\lib\components\**\*.svelte" 
    "src\lib\components\**\*.svelte"
    "src\routes\**\*.svelte"
    "*.svelte"
)

Write-Host "📁 Scanning file patterns..." -ForegroundColor Cyan

foreach ($pattern in $targetPatterns) {
    $fullPattern = Join-Path $rootPath $pattern
    Write-Host "  Checking: $pattern" -ForegroundColor Gray
    
    try {
        $files = Get-ChildItem -Path $fullPattern -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Extension -eq ".svelte" }
        
        foreach ($file in $files) {
            $content = Get-Content -Path $file.FullName -Raw -ErrorAction SilentlyContinue
            if (-not $content) { continue }
            
            $originalContent = $content
            $fileFixed = $false
            
            # Apply each event handler mapping
            foreach ($oldEvent in $eventMappings.Keys) {
                $newEvent = $eventMappings[$oldEvent]
                if ($content.Contains($oldEvent)) {
                    $content = $content -replace [regex]::Escape($oldEvent), $newEvent
                    $fileFixed = $true
                }
            }
            
            # Write back if changes were made
            if ($fileFixed -and $content -ne $originalContent) {
                Set-Content -Path $file.FullName -Value $content -NoNewline
                $fixCount++
                
                $relativePath = $file.FullName.Substring($rootPath.Length + 1)
                Write-Host "  ✅ Fixed: $relativePath" -ForegroundColor Green
            }
        }
    }
    catch {
        Write-Host "  ⚠️  Pattern error: $pattern" -ForegroundColor Yellow
    }
}

Write-Host "`n🎯 Batch Fix Summary:" -ForegroundColor Green
Write-Host "  Files modified: $fixCount" -ForegroundColor White
Write-Host "  Event handlers modernized: $(($eventMappings.Keys).Count) types" -ForegroundColor White

# Run a quick validation check
Write-Host "`n🔍 Running validation check..." -ForegroundColor Cyan
try {
    Set-Location $rootPath
    $checkResult = & timeout 30 npx svelte-check --threshold error --output human 2>&1 | Select-String "svelte-check found"
    if ($checkResult) {
        Write-Host "  $($checkResult.Line)" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "  ⚠️  Validation check failed" -ForegroundColor Red
}

Write-Host "`n✅ Batch event handler fixes completed!" -ForegroundColor Green
Write-Host "Next steps: Run 'npm run check' for full validation" -ForegroundColor Gray