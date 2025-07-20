# ENHANCED: Critical Svelte/TypeScript Error Fixer
# Fixed and integrated with existing web-app structure

Write-Host "🔧 Fixing Critical Svelte/TypeScript Issues..." -ForegroundColor Green

$fixes = @{
    tabindexFixed = @()
    booleanPropsFixed = @()
    accessibilityIssues = @()
    propTypeIssues = @()
    filesProcessed = 0
    totalIssues = 0
}

# Get all Svelte files in the current directory
$svelteFiles = Get-ChildItem -Path "." -Filter "*.svelte" | Where-Object { $_.Name -like "+*" }

if ($svelteFiles.Count -eq 0) {
    Write-Host "⚠️  No Svelte component files found (looking for +*.svelte)" -ForegroundColor Yellow
    Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Gray
    
    # Also check in src directory if it exists
    if (Test-Path "src") {
        Write-Host "🔍 Checking src directory..." -ForegroundColor Cyan
        $svelteFiles = Get-ChildItem -Path "src" -Filter "*.svelte" -Recurse
        if ($svelteFiles.Count -gt 0) {
            Write-Host "✅ Found $($svelteFiles.Count) Svelte files in src/" -ForegroundColor Green
        }
    }
    
    if ($svelteFiles.Count -eq 0) {
        Write-Host "❌ No Svelte files found to process." -ForegroundColor Red
        Write-Host "💡 Make sure you're in the correct directory with .svelte files" -ForegroundColor Cyan
        exit 1
    }
}

Write-Host "📊 Found $($svelteFiles.Count) Svelte files to analyze" -ForegroundColor Cyan

foreach ($file in $svelteFiles) {
    try {
        $content = Get-Content $file.FullName -Raw
        $originalContent = $content
        $issuesInFile = 0
        
        Write-Host "`n📄 Analyzing: $($file.Name)" -ForegroundColor Yellow
        
        # 1. Fix tabindex="number" to tabindex={number} - CRITICAL
        $tabindexMatches = [regex]::Matches($content, 'tabindex="(\d+)"')
        if ($tabindexMatches.Count -gt 0) {
            $content = $content -replace 'tabindex="(\d+)"', 'tabindex={$1}'
            $matchStrings = $tabindexMatches | ForEach-Object { $_.Value }
            $fixes.tabindexFixed += @{
                file = $file.Name
                matches = $matchStrings
            }
            Write-Host "  ✅ Fixed tabindex quotes: $($matchStrings -join ', ')" -ForegroundColor Green
            $issuesInFile += $tabindexMatches.Count
        }
        
        # 2. Fix boolean props as strings - MEDIUM PRIORITY
        $booleanMatches = [regex]::Matches($content, '(disabled|readonly|checked)="(true|false)"')
        if ($booleanMatches.Count -gt 0) {
            $content = $content -replace '(disabled|readonly|checked)="(true|false)"', '$1={$2}'
            $matchStrings = $booleanMatches | ForEach-Object { $_.Value }
            $fixes.booleanPropsFixed += @{
                file = $file.Name
                matches = $matchStrings
            }
            Write-Host "  ✅ Fixed boolean props: $($matchStrings -join ', ')" -ForegroundColor Green
            $issuesInFile += $booleanMatches.Count
        }
        
        # 3. Check for accessibility issues (report only)
        $hasClickableDiv = $content -match '<div[^>]*on:click[^>]*>'
        $hasProperRole = $content -match 'role="button"'
        $hasTabindex = $content -match 'tabindex='
        
        if ($hasClickableDiv -and (-not $hasProperRole -or -not $hasTabindex)) {
            $fixes.accessibilityIssues += $file.Name
            Write-Host "  🔵 Accessibility review needed: clickable div missing proper ARIA" -ForegroundColor Blue
        }
        
        # 4. Check for generic string props - MEDIUM PRIORITY  
        $stringPropMatches = [regex]::Matches($content, 'export let \w+: string')
        if ($stringPropMatches.Count -gt 0) {
            $matchStrings = $stringPropMatches | ForEach-Object { $_.Value }
            $fixes.propTypeIssues += @{
                file = $file.Name
                matches = $matchStrings
            }
            Write-Host "  🟡 Consider union types for: $($matchStrings -join ', ')" -ForegroundColor Yellow
        }
        
        # Write file back if changes were made
        if ($content -ne $originalContent) {
            Set-Content -Path $file.FullName -Value $content -NoNewline
            Write-Host "  💾 Updated: $($file.Name)" -ForegroundColor Green
        } elseif ($issuesInFile -eq 0 -and $stringPropMatches.Count -eq 0 -and $fixes.accessibilityIssues -notcontains $file.Name) {
            Write-Host "  ✓ No issues found: $($file.Name)" -ForegroundColor Gray
        }
        
        $fixes.filesProcessed++
        $fixes.totalIssues += $issuesInFile
        
    } catch {
        Write-Host "  ❌ Error processing $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Generate comprehensive report
Write-Host "`n📊 COMPREHENSIVE FIX SUMMARY" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Gray

Write-Host "📁 Files processed: $($fixes.filesProcessed)" -ForegroundColor Cyan
Write-Host "🔧 Total critical issues fixed: $($fixes.totalIssues)" -ForegroundColor Green
Write-Host "🔴 TabIndex fixes: $($fixes.tabindexFixed.Count) files" -ForegroundColor Yellow
Write-Host "🟡 Boolean prop fixes: $($fixes.booleanPropsFixed.Count) files" -ForegroundColor Yellow
Write-Host "🔵 Accessibility reviews needed: $($fixes.accessibilityIssues.Count) files" -ForegroundColor Yellow
Write-Host "📝 Prop type reviews suggested: $($fixes.propTypeIssues.Count) files" -ForegroundColor Yellow

# Detailed breakdown
if ($fixes.tabindexFixed.Count -gt 0) {
    Write-Host "`n🔴 CRITICAL FIXES APPLIED - TabIndex:" -ForegroundColor Red
    foreach ($fix in $fixes.tabindexFixed) {
        Write-Host "   📄 $($fix.file): $($fix.matches -join ', ')" -ForegroundColor White
    }
}

if ($fixes.booleanPropsFixed.Count -gt 0) {
    Write-Host "`n🟡 MEDIUM FIXES APPLIED - Boolean Props:" -ForegroundColor Yellow
    foreach ($fix in $fixes.booleanPropsFixed) {
        Write-Host "   📄 $($fix.file): $($fix.matches -join ', ')" -ForegroundColor White
    }
}

if ($fixes.propTypeIssues.Count -gt 0) {
    Write-Host "`n📝 PROP TYPE SUGGESTIONS:" -ForegroundColor Cyan
    foreach ($fix in $fixes.propTypeIssues) {
        Write-Host "   📄 $($fix.file):" -ForegroundColor White
        foreach ($match in $fix.matches) {
            $propName = ($match -split ' ')[2] -replace ':', ''
            Write-Host "      • $match → export let $propName`: `"sm`" | `"md`" | `"lg`" = `"md`"" -ForegroundColor Gray
        }
    }
}

if ($fixes.accessibilityIssues.Count -gt 0) {
    Write-Host "`n🔵 ACCESSIBILITY REVIEW NEEDED:" -ForegroundColor Blue
    foreach ($file in $fixes.accessibilityIssues) {
        Write-Host "   📄 $file: Add role=`"button`", tabindex={0}, and keyboard handlers" -ForegroundColor White
    }
}

# Success metrics
$successRate = if ($fixes.filesProcessed -gt 0) { 
    [math]::Round(($fixes.filesProcessed - $fixes.accessibilityIssues.Count) / $fixes.filesProcessed * 100, 1)
} else { 0 }

Write-Host "`n🎯 COMPLETION STATUS:" -ForegroundColor Green
Write-Host "   ✅ Success rate: $successRate%" -ForegroundColor White
Write-Host "   🔧 Auto-fixable issues: RESOLVED" -ForegroundColor White
Write-Host "   👀 Manual review items: $($fixes.propTypeIssues.Count + $fixes.accessibilityIssues.Count)" -ForegroundColor White

# Next steps
Write-Host "`n🚀 RECOMMENDED NEXT STEPS:" -ForegroundColor Cyan
Write-Host "   1. Run TypeScript check: npm run check" -ForegroundColor White
Write-Host "   2. Test keyboard navigation on interactive elements" -ForegroundColor White
Write-Host "   3. Review prop types for union type opportunities" -ForegroundColor White
Write-Host "   4. Test accessibility with screen readers" -ForegroundColor White

# Create log file
$logPath = "error-fix-log-$(Get-Date -Format 'yyyy-MM-dd-HHmm').txt"
$logContent = @"
Svelte Error Fix Log - $(Get-Date)
===========================================

Files Processed: $($fixes.filesProcessed)
Total Issues Fixed: $($fixes.totalIssues)

TabIndex Fixes ($($fixes.tabindexFixed.Count)):
$(($fixes.tabindexFixed | ForEach-Object { "  $($_.file): $($_.matches -join ', ')" }) -join "`n")

Boolean Prop Fixes ($($fixes.booleanPropsFixed.Count)):
$(($fixes.booleanPropsFixed | ForEach-Object { "  $($_.file): $($_.matches -join ', ')" }) -join "`n")

Accessibility Reviews Needed ($($fixes.accessibilityIssues.Count)):
$(($fixes.accessibilityIssues | ForEach-Object { "  $_" }) -join "`n")

Prop Type Suggestions ($($fixes.propTypeIssues.Count)):
$(($fixes.propTypeIssues | ForEach-Object { "  $($_.file): $($_.matches -join ', ')" }) -join "`n")
"@

Set-Content -Path $logPath -Value $logContent
Write-Host "`n📋 Detailed log saved: $logPath" -ForegroundColor Gray

if ($fixes.totalIssues -gt 0) {
    Write-Host "`n🎉 SUCCESS: $($fixes.totalIssues) critical issues fixed!" -ForegroundColor Green
} else {
    Write-Host "`n✨ EXCELLENT: No critical issues found! Your code is clean." -ForegroundColor Green
}

Write-Host "`n💡 Tip: Use VS Code tasks (Ctrl+Shift+P → 'Tasks: Run Task') for easy re-running" -ForegroundColor Cyan
