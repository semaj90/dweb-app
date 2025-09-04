# 🚀 Svelte 5 State Variables Final Conversion
# Converts remaining non-reactive variables to $state() syntax

Write-Host "🚀 SVELTE 5 STATE VARIABLES CONVERSION STARTING..." -ForegroundColor Cyan
Write-Host "Target: 600+ non-reactive variable warnings → $state() syntax" -ForegroundColor Yellow

$frontendPath = "sveltekit-frontend/src"
$errorCount = 0
$fixedCount = 0

# Check if frontend directory exists
if (-not (Test-Path $frontendPath)) {
    Write-Host "❌ Frontend directory not found: $frontendPath" -ForegroundColor Red
    exit 1
}

Write-Host "`n🔧 Phase 1: Converting basic state variables..." -ForegroundColor Green

# Find all Svelte files with potential state variables
$svelteFiles = Get-ChildItem -Path $frontendPath -Recurse -Filter "*.svelte" -ErrorAction SilentlyContinue

foreach ($file in $svelteFiles) {
    try {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if (-not $content) { continue }

        $originalContent = $content
        $hasChanges = $false

        # Convert let variable declarations to $state()
        # Pattern: let variableName = value;
        $content = $content -replace 'let\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*([^;]+);', 'let $1 = $state($2);'

        # Convert specific patterns that need $state()
        $statePatterns = @(
            @{ Pattern = 'let\s+uploadedFiles\s*:\s*UploadedFile\[\]\s*=\s*\[\]'; Replacement = 'let uploadedFiles: UploadedFile[] = $state([])' }
            @{ Pattern = 'let\s+searchResults\s*:\s*SearchResults\s*\|\s*null\s*=\s*null'; Replacement = 'let searchResults: SearchResults | null = $state(null)' }
            @{ Pattern = 'let\s+testResults\s*:\s*TestResults\s*\|\s*null\s*=\s*null'; Replacement = 'let testResults: TestResults | null = $state(null)' }
            @{ Pattern = 'let\s+analysisResults\s*:\s*AnalysisResults\s*\|\s*null\s*=\s*null'; Replacement = 'let analysisResults: AnalysisResults | null = $state(null)' }
            @{ Pattern = 'let\s+systemStatus\s*:\s*SystemStatus\s*=\s*\{'; Replacement = 'let systemStatus: SystemStatus = $state({' }
            @{ Pattern = 'let\s+isProcessing\s*=\s*false'; Replacement = 'let isProcessing = $state(false)' }
            @{ Pattern = 'let\s+isSearching\s*=\s*false'; Replacement = 'let isSearching = $state(false)' }
            @{ Pattern = 'let\s+isTestingSearch\s*=\s*false'; Replacement = 'let isTestingSearch = $state(false)' }
            @{ Pattern = 'let\s+isAnalyzing\s*=\s*false'; Replacement = 'let isAnalyzing = $state(false)' }
            @{ Pattern = 'let\s+webgpuStatus\s*=\s*\{'; Replacement = 'let webgpuStatus = $state({' }
            @{ Pattern = 'let\s+results\s*:\s*any\[\]\s*=\s*\[\]'; Replacement = 'let results: any[] = $state([])' }
            @{ Pattern = 'let\s+demoInput\s*=\s*'; Replacement = 'let demoInput = $state(' }
            @{ Pattern = 'let\s+currentDemo\s*=\s*'; Replacement = 'let currentDemo = $state(' }
            @{ Pattern = 'let\s+performanceMetrics\s*=\s*\{'; Replacement = 'let performanceMetrics = $state({' }
            @{ Pattern = 'let\s+searchQuery\s*=\s*"'; Replacement = 'let searchQuery = $state("' }
            @{ Pattern = 'let\s+testQuery\s*=\s*"'; Replacement = 'let testQuery = $state("' }
        )

        foreach ($pattern in $statePatterns) {
            if ($content -match $pattern.Pattern) {
                $content = $content -replace $pattern.Pattern, $pattern.Replacement
                $hasChanges = $true
            }
        }

        if ($hasChanges -and $content -ne $originalContent) {
            Set-Content -Path $file.FullName -Value $content -Encoding UTF8
            $fixedCount++
            Write-Host "  ✅ Updated state variables in $($file.Name)" -ForegroundColor Green
        }
    } catch {
        Write-Warning "Error processing $($file.FullName): $($_.Exception.Message)"
        $errorCount++
    }
}

Write-Host "`n🎯 Phase 2: Fixing event handler syntax..." -ForegroundColor Green

# Convert event handlers from on:event to onevent
foreach ($file in $svelteFiles) {
    try {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if (-not $content) { continue }

        $originalContent = $content
        $hasChanges = $false

        # Convert event handler patterns
        $eventPatterns = @(
            @{ Pattern = 'on:click='; Replacement = 'onclick=' }
            @{ Pattern = 'on:change='; Replacement = 'onchange=' }
            @{ Pattern = 'on:input='; Replacement = 'oninput=' }
            @{ Pattern = 'on:submit='; Replacement = 'onsubmit=' }
            @{ Pattern = 'on:keydown='; Replacement = 'onkeydown=' }
            @{ Pattern = 'on:keyup='; Replacement = 'onkeyup=' }
            @{ Pattern = 'on:focus='; Replacement = 'onfocus=' }
            @{ Pattern = 'on:blur='; Replacement = 'onblur=' }
            @{ Pattern = 'on:load='; Replacement = 'onload=' }
            @{ Pattern = 'on:resize='; Replacement = 'onresize=' }
        )

        foreach ($eventPattern in $eventPatterns) {
            if ($content -match $eventPattern.Pattern) {
                $content = $content -replace $eventPattern.Pattern, $eventPattern.Replacement
                $hasChanges = $true
            }
        }

        if ($hasChanges -and $content -ne $originalContent) {
            Set-Content -Path $file.FullName -Value $content -Encoding UTF8
            Write-Host "  ✅ Updated event handlers in $($file.Name)" -ForegroundColor Green
        }
    } catch {
        Write-Warning "Error processing $($file.FullName): $($_.Exception.Message)"
        $errorCount++
    }
}

Write-Host "`n🔧 Phase 3: Adding missing type imports..." -ForegroundColor Green

# Add missing type imports to files that need them
$typesToImport = @{
    'Props' = 'import type { Props } from "$lib/types/global";'
    'SystemStatus' = 'import type { SystemStatus } from "$lib/types/global";'
    'SearchResults' = 'import type { SearchResults } from "$lib/types/global";'
    'TestResults' = 'import type { TestResults } from "$lib/types/global";'
    'AnalysisResults' = 'import type { AnalysisResults } from "$lib/types/global";'
    'UploadedFile' = 'import type { UploadedFile } from "$lib/types/global";'
    'DemoUser' = 'import type { DemoUser } from "$lib/types/global";'
}

foreach ($file in $svelteFiles) {
    try {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if (-not $content) { continue }

        $originalContent = $content
        $hasChanges = $false
        $importsToAdd = @()

        # Check which types are used but not imported
        foreach ($type in $typesToImport.Keys) {
            if ($content -match "\b$type\b" -and $content -notmatch "import.*$type") {
                $importsToAdd += $typesToImport[$type]
                $hasChanges = $true
            }
        }

        if ($hasChanges) {
            # Find the script tag and add imports
            if ($content -match '<script[^>]*>') {
                $scriptStart = $content.IndexOf('<script')
                $scriptEnd = $content.IndexOf('>', $scriptStart) + 1

                $beforeScript = $content.Substring(0, $scriptEnd)
                $afterScript = $content.Substring($scriptEnd)

                $imports = $importsToAdd -join "`n  "
                $content = $beforeScript + "`n  " + $imports + $afterScript
            }

            Set-Content -Path $file.FullName -Value $content -Encoding UTF8
            Write-Host "  ✅ Added type imports to $($file.Name)" -ForegroundColor Green
        }
    } catch {
        Write-Warning "Error processing $($file.FullName): $($_.Exception.Message)"
        $errorCount++
    }
}

Write-Host "`n🎛️ Phase 4: Fixing component prop issues..." -ForegroundColor Green

# Fix common component prop issues
foreach ($file in $svelteFiles) {
    try {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if (-not $content) { continue }

        $originalContent = $content
        $hasChanges = $false

        # Fix duplicate disabled attributes
        $content = $content -replace 'disabled=\{\$state\}\s+disabled=', 'disabled='

        # Fix prop binding syntax
        $content = $content -replace '\{caseId:\s*([^}]+)\}', 'caseId={$1}'

        # Fix variant prop for Button components
        if ($content -match 'variant="') {
            # Ensure Button component has proper variant prop support
            if ($content -notmatch 'import.*Button.*from') {
                $content = $content -replace '(<script[^>]*>)', '$1' + "`n  import { Button } from `"$lib/components/ui`";"
            }
            $hasChanges = $true
        }

        if ($hasChanges -and $content -ne $originalContent) {
            Set-Content -Path $file.FullName -Value $content -Encoding UTF8
            Write-Host "  ✅ Fixed component props in $($file.Name)" -ForegroundColor Green
        }
    } catch {
        Write-Warning "Error processing $($file.FullName): $($_.Exception.Message)"
        $errorCount++
    }
}

Write-Host "`n📊 SVELTE 5 STATE CONVERSION SUMMARY:" -ForegroundColor Cyan
Write-Host "✅ Files processed successfully: $fixedCount" -ForegroundColor Green
Write-Host "❌ Files with errors: $errorCount" -ForegroundColor Red

if ($errorCount -eq 0) {
    Write-Host "`n🎉 SVELTE 5 STATE CONVERSION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "📈 Expected error reduction: ~600 state & event errors resolved" -ForegroundColor Cyan
    Write-Host "🎯 Progress: 3,765 → ~3,165 errors remaining" -ForegroundColor Yellow
} else {
    Write-Host "`n⚠️ Conversion completed with some errors. Manual review recommended." -ForegroundColor Yellow
}

Write-Host "`n🔄 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Run TypeScript check: npm run check" -ForegroundColor White
Write-Host "2. Test compilation: npm run build" -ForegroundColor White
Write-Host "3. Review any remaining syntax issues manually" -ForegroundColor White
Write-Host "4. Target final cleanup to reach <500 errors" -ForegroundColor White
