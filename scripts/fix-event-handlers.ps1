# PowerShell Script: Automated Event Handler Migration
# Purpose: Batch fix on: event handlers to modern onclick/onchange patterns
# Usage: Run from VS Code task "Batch Fix: Event Handler Deprecations"

param(
    [string]$ProjectPath = "sveltekit-frontend",
    [switch]$DryRun = $false
)

Write-Host "Starting Event Handler Migration..." -ForegroundColor Green
Write-Host "Project Path: $ProjectPath" -ForegroundColor Cyan
Write-Host "Dry Run Mode: $DryRun" -ForegroundColor Cyan

# Change to project directory
Set-Location $ProjectPath

# Define event handler mappings
$eventMappings = @{
    'on:click=' = 'onclick='
    'on:change=' = 'onchange='
    'on:input=' = 'oninput='
    'on:submit=' = 'onsubmit='
    'on:focus=' = 'onfocus='
    'on:blur=' = 'onblur='
    'on:mouseenter=' = 'onmouseenter='
    'on:mouseleave=' = 'onmouseleave='
    'on:mousedown=' = 'onmousedown='
    'on:mouseup=' = 'onmouseup='
    'on:keydown=' = 'onkeydown='
    'on:keyup=' = 'onkeyup='
    'on:keypress=' = 'onkeypress='
    'on:load=' = 'onload='
}

# Find all Svelte files
$svelteFiles = Get-ChildItem -Recurse -Filter "*.svelte" | Where-Object { $_.FullName -notmatch "node_modules" }

Write-Host "Found $($svelteFiles.Count) Svelte files to process..." -ForegroundColor Yellow

$totalReplacements = 0
$processedFiles = 0

foreach ($file in $svelteFiles) {
    $fileReplacements = 0
    
    # Skip if file doesn't exist
    if (-not (Test-Path $file.FullName)) {
        continue
    }
    
    $content = (Get-Content -Path $file.FullName) -join "`n"
    $originalContent = $content
    
    # Skip if content is empty or null
    if ([string]::IsNullOrEmpty($content)) {
        continue
    }
    
    # Apply each event handler mapping
    foreach ($oldPattern in $eventMappings.Keys) {
        $newPattern = $eventMappings[$oldPattern]
        $beforeCount = ([regex]::Matches($content, [regex]::Escape($oldPattern))).Count
        
        if ($beforeCount -gt 0) {
            $content = $content -replace [regex]::Escape($oldPattern), $newPattern
            $afterCount = ([regex]::Matches($content, [regex]::Escape($oldPattern))).Count
            $replacements = $beforeCount - $afterCount
            
            if ($replacements -gt 0) {
                Write-Host "  $($file.Name): $oldPattern to $newPattern ($replacements replacements)" -ForegroundColor Green
                $fileReplacements += $replacements
            }
        }
    }
    
    # Handle special cases that need more complex replacements
    
    # Fix on:submit|preventDefault pattern
    if ($content -match 'on:submit\|preventDefault=') {
        $content = $content -replace 'on:submit\|preventDefault=\{([^}]+)\}', 'onsubmit={(e) => { e.preventDefault(); $1(); }}'
        Write-Host "  $($file.Name): Fixed on:submit|preventDefault pattern" -ForegroundColor Green
        $fileReplacements++
    }
    
    # Fix on:click|stopPropagation pattern  
    if ($content -match 'on:click\|stopPropagation=') {
        $content = $content -replace 'on:click\|stopPropagation=\{([^}]+)\}', 'onclick={(e) => { e.stopPropagation(); $1(); }}'
        Write-Host "  $($file.Name): Fixed on:click|stopPropagation pattern" -ForegroundColor Green
        $fileReplacements++
    }
    
    # Update svelte:window event handlers
    $content = $content -replace '<svelte:window on:keydown=', '<svelte:window onkeydown='
    $content = $content -replace '<svelte:window on:resize=', '<svelte:window onresize='
    $content = $content -replace '<svelte:window on:scroll=', '<svelte:window onscroll='
    
    # Only update file if changes were made and not in dry run mode
    if ($content -ne $originalContent) {
        $processedFiles++
        $totalReplacements += $fileReplacements
        
        if (-not $DryRun) {
            Set-Content -Path $file.FullName -Value $content -NoNewline
            Write-Host "Updated: $($file.Name) ($fileReplacements changes)" -ForegroundColor Green
        } else {
            Write-Host "Would update: $($file.Name) ($fileReplacements changes)" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Migration Summary:" -ForegroundColor Cyan
Write-Host "  Files processed: $processedFiles" -ForegroundColor White
Write-Host "  Total replacements: $totalReplacements" -ForegroundColor White
Write-Host "  Dry run mode: $DryRun" -ForegroundColor White

if ($DryRun) {
    Write-Host ""
    Write-Host "This was a dry run. Re-run without -DryRun to apply changes." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Event handler migration completed!" -ForegroundColor Green
    Write-Host "Run 'npm run check:ultra-fast' to validate changes" -ForegroundColor Cyan
}

# Return to original directory
Set-Location ..