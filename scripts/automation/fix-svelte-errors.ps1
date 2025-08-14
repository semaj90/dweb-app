# Comprehensive Svelte Error Fixer and Best Practices Checker
# This script automatically fixes common Svelte/TypeScript errors and applies best practices

param(
    [string]$Path = "sveltekit-frontend",
    [switch]$DryRun = $false,
    [switch]$Backup = $true,
    [switch]$Verbose = $false
)

Write-Host "🔧 Svelte Error Fixer & Best Practices Checker" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Statistics
$stats = @{
    FilesProcessed = 0
    ErrorsFixed = 0
    WarningsFixed = 0
    BackupsCreated = 0
    DeprecatedEventsFixed = 0
    UnusedCSSRemoved = 0
    ImportsFixed = 0
    TypeErrorsFixed = 0
}

# Create backup directory if needed
if ($Backup -and -not $DryRun) {
    $backupDir = "backups\backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    Write-Host "📁 Backup directory created: $backupDir" -ForegroundColor Green
}

# Function to create backup
function Backup-File {
    param($FilePath)
    
    if ($Backup -and -not $DryRun) {
        $relativePath = $FilePath.Replace($Path, "").TrimStart("\")
        $backupPath = Join-Path $backupDir $relativePath
        $backupDir = Split-Path $backupPath -Parent
        
        if (!(Test-Path $backupDir)) {
            New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        }
        
        Copy-Item -Path $FilePath -Destination $backupPath -Force
        $stats.BackupsCreated++
        
        if ($Verbose) {
            Write-Host "  📋 Backed up: $relativePath" -ForegroundColor Gray
        }
    }
}

# Function to fix deprecated Svelte event handlers
function Fix-DeprecatedEvents {
    param($Content, $FilePath)
    
    $patterns = @(
        @{ Old = 'on:click='; New = 'onclick=' },
        @{ Old = 'on:submit='; New = 'onsubmit=' },
        @{ Old = 'on:input='; New = 'oninput=' },
        @{ Old = 'on:change='; New = 'onchange=' },
        @{ Old = 'on:focus='; New = 'onfocus=' },
        @{ Old = 'on:blur='; New = 'onblur=' },
        @{ Old = 'on:keydown='; New = 'onkeydown=' },
        @{ Old = 'on:keyup='; New = 'onkeyup=' },
        @{ Old = 'on:mouseenter='; New = 'onmouseenter=' },
        @{ Old = 'on:mouseleave='; New = 'onmouseleave=' },
        @{ Old = 'on:mouseover='; New = 'onmouseover=' },
        @{ Old = 'on:mouseout='; New = 'onmouseout=' },
        @{ Old = 'on:scroll='; New = 'onscroll=' },
        @{ Old = 'on:load='; New = 'onload=' },
        @{ Old = 'on:error='; New = 'onerror=' }
    )
    
    $fixedContent = $Content
    $fixCount = 0
    
    foreach ($pattern in $patterns) {
        $matches = [regex]::Matches($fixedContent, [regex]::Escape($pattern.Old))
        if ($matches.Count -gt 0) {
            $fixedContent = $fixedContent -replace [regex]::Escape($pattern.Old), $pattern.New
            $fixCount += $matches.Count
            $stats.DeprecatedEventsFixed += $matches.Count
            
            if ($Verbose) {
                Write-Host "    ✅ Fixed $($matches.Count) instances of '$($pattern.Old)'" -ForegroundColor Green
            }
        }
    }
    
    return @{
        Content = $fixedContent
        FixCount = $fixCount
    }
}

# Function to fix unused CSS selectors
function Fix-UnusedCSS {
    param($Content, $FilePath)
    
    # For Svelte files, analyze component markup and remove unused styles
    if ($FilePath -match '\.svelte$') {
        # Extract HTML content
        $htmlMatch = [regex]::Match($Content, '(?s)^(.*?)<style', [System.Text.RegularExpressions.RegexOptions]::Singleline)
        $htmlContent = if ($htmlMatch.Success) { $htmlMatch.Groups[1].Value } else { "" }
        
        # Extract style content
        $styleMatch = [regex]::Match($Content, '(?s)<style[^>]*>(.*?)</style>', [System.Text.RegularExpressions.RegexOptions]::Singleline)
        
        if ($styleMatch.Success) {
            $styleContent = $styleMatch.Groups[1].Value
            $originalStyleLength = $styleContent.Length
            
            # Find all CSS selectors
            $selectorPattern = '([.#][\w-]+|[\w-]+)\s*\{[^}]*\}'
            $selectors = [regex]::Matches($styleContent, $selectorPattern)
            
            $unusedSelectors = @()
            foreach ($selector in $selectors) {
                $selectorName = [regex]::Match($selector.Value, '^([.#]?[\w-]+)').Groups[1].Value
                $cleanSelector = $selectorName -replace '^[.#]', ''
                
                # Check if selector is used in HTML
                if ($htmlContent -notmatch "\b$cleanSelector\b") {
                    $unusedSelectors += $selector.Value
                }
            }
            
            # Remove unused selectors
            $fixedStyle = $styleContent
            foreach ($unused in $unusedSelectors) {
                $fixedStyle = $fixedStyle -replace [regex]::Escape($unused), ''
            }
            
            # Clean up empty lines
            $fixedStyle = $fixedStyle -replace '(\r?\n\s*){3,}', "`n`n"
            
            if ($fixedStyle.Length -lt $originalStyleLength) {
                $Content = $Content -replace [regex]::Escape($styleContent), $fixedStyle
                $stats.UnusedCSSRemoved += $unusedSelectors.Count
                
                if ($Verbose -and $unusedSelectors.Count -gt 0) {
                    Write-Host "    🎨 Removed $($unusedSelectors.Count) unused CSS selectors" -ForegroundColor Yellow
                }
                
                return @{
                    Content = $Content
                    FixCount = $unusedSelectors.Count
                }
            }
        }
    }
    
    return @{
        Content = $Content
        FixCount = 0
    }
}

# Function to fix missing default exports
function Fix-MissingExports {
    param($Content, $FilePath)
    
    $fixCount = 0
    $fixedContent = $Content
    
    # Check if file has any exports but no default export
    if ($Content -match 'export\s+(const|let|var|function|class)' -and $Content -notmatch 'export\s+default') {
        # Get the component name from filename
        $componentName = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
        $componentName = $componentName -replace '^[+]', '' # Remove + prefix for Svelte routes
        
        # Add default export at the end if it's a Svelte component
        if ($FilePath -match '\.svelte$') {
            # Svelte components don't need default exports typically
        } elseif ($FilePath -match '\.(ts|js)$') {
            # For TS/JS files, check if there's a main export that should be default
            if ($Content -match "export\s+(?:const|let|var|function|class)\s+($componentName)\b") {
                $fixedContent += "`n`nexport default $componentName;"
                $fixCount = 1
                $stats.ImportsFixed++
                
                if ($Verbose) {
                    Write-Host "    📦 Added default export for $componentName" -ForegroundColor Magenta
                }
            }
        }
    }
    
    return @{
        Content = $fixedContent
        FixCount = $fixCount
    }
}

# Function to fix TypeScript type errors
function Fix-TypeErrors {
    param($Content, $FilePath)
    
    $fixCount = 0
    $fixedContent = $Content
    
    # Fix common type issues
    $typePatterns = @(
        # Fix optional property access
        @{
            Pattern = '(\w+)\.(\w+)(?!\?)'
            Check = '\$1\?\.\$2'
            Description = "Optional chaining"
        },
        # Add type assertions for common patterns
        @{
            Pattern = 'as\s+unknown\s+as\s+\w+'
            Replace = 'as any'
            Description = "Simplify type assertions"
        }
    )
    
    # Fix bindable properties in Svelte 5
    if ($FilePath -match '\.svelte$' -and $Content -match 'bind:') {
        # Add $bindable() to props
        $bindPattern = 'bind:(\w+)'
        $bindMatches = [regex]::Matches($Content, $bindPattern)
        
        foreach ($match in $bindMatches) {
            $propName = $match.Groups[1].Value
            
            # Check if prop needs $bindable()
            if ($Content -match "let\s+\{\s*.*$propName.*\}\s*=\s*\$props\(\)") {
                # Update to use $bindable()
                $fixedContent = $fixedContent -replace "(let\s+\{[^}]*?)($propName)([^}]*?\})", "`$1$propName = `$bindable()`$3"
                $fixCount++
                $stats.TypeErrorsFixed++
                
                if ($Verbose) {
                    Write-Host "    🔗 Fixed bindable property: $propName" -ForegroundColor Cyan
                }
            }
        }
    }
    
    return @{
        Content = $fixedContent
        FixCount = $fixCount
    }
}

# Function to add missing imports
function Fix-MissingImports {
    param($Content, $FilePath)
    
    $fixCount = 0
    $fixedContent = $Content
    $addedImports = @()
    
    # Common Svelte imports
    $commonImports = @{
        'onMount' = "import { onMount } from 'svelte';"
        'createEventDispatcher' = "import { createEventDispatcher } from 'svelte';"
        'getContext' = "import { getContext } from 'svelte';"
        'setContext' = "import { setContext } from 'svelte';"
        'tick' = "import { tick } from 'svelte';"
        'writable' = "import { writable } from 'svelte/store';"
        'readable' = "import { readable } from 'svelte/store';"
        'derived' = "import { derived } from 'svelte/store';"
    }
    
    foreach ($key in $commonImports.Keys) {
        if ($Content -match "\b$key\b" -and $Content -notmatch "import.*$key") {
            $addedImports += $commonImports[$key]
            $fixCount++
            $stats.ImportsFixed++
        }
    }
    
    if ($addedImports.Count -gt 0) {
        # Add imports at the beginning of script section
        if ($FilePath -match '\.svelte$') {
            if ($Content -match '<script') {
                $fixedContent = $Content -replace '(<script[^>]*>)', "`$1`n$($addedImports -join "`n")`n"
            } else {
                $fixedContent = "<script>`n$($addedImports -join "`n")`n</script>`n" + $Content
            }
        } else {
            $fixedContent = ($addedImports -join "`n") + "`n`n" + $Content
        }
        
        if ($Verbose) {
            Write-Host "    📥 Added $($addedImports.Count) missing imports" -ForegroundColor Blue
        }
    }
    
    return @{
        Content = $fixedContent
        FixCount = $fixCount
    }
}

# Function to process a single file
function Process-File {
    param($FilePath)
    
    if ($Verbose) {
        Write-Host "`n📄 Processing: $FilePath" -ForegroundColor White
    }
    
    $content = Get-Content -Path $FilePath -Raw -Encoding UTF8
    $originalContent = $content
    $totalFixes = 0
    
    # Apply fixes
    $result = Fix-DeprecatedEvents -Content $content -FilePath $FilePath
    $content = $result.Content
    $totalFixes += $result.FixCount
    
    $result = Fix-UnusedCSS -Content $content -FilePath $FilePath
    $content = $result.Content
    $totalFixes += $result.FixCount
    
    $result = Fix-MissingExports -Content $content -FilePath $FilePath
    $content = $result.Content
    $totalFixes += $result.FixCount
    
    $result = Fix-TypeErrors -Content $content -FilePath $FilePath
    $content = $result.Content
    $totalFixes += $result.FixCount
    
    $result = Fix-MissingImports -Content $content -FilePath $FilePath
    $content = $result.Content
    $totalFixes += $result.FixCount
    
    # Save changes if any fixes were applied
    if ($content -ne $originalContent) {
        if (-not $DryRun) {
            Backup-File -FilePath $FilePath
            Set-Content -Path $FilePath -Value $content -Encoding UTF8 -NoNewline
            Write-Host "  ✅ Fixed $totalFixes issues in $(Split-Path $FilePath -Leaf)" -ForegroundColor Green
        } else {
            Write-Host "  🔍 Would fix $totalFixes issues in $(Split-Path $FilePath -Leaf)" -ForegroundColor Yellow
        }
        $stats.ErrorsFixed += $totalFixes
    }
    
    $stats.FilesProcessed++
}

# Main processing
Write-Host "`n🔍 Scanning for files..." -ForegroundColor Yellow

# Get all Svelte and TypeScript files
$files = @()
$files += Get-ChildItem -Path $Path -Filter "*.svelte" -Recurse -ErrorAction SilentlyContinue
$files += Get-ChildItem -Path $Path -Filter "*.ts" -Recurse -ErrorAction SilentlyContinue
$files += Get-ChildItem -Path $Path -Filter "*.js" -Recurse -ErrorAction SilentlyContinue

# Exclude node_modules and build directories
$files = $files | Where-Object { 
    $_.FullName -notmatch 'node_modules|\.svelte-kit|build|dist' 
}

Write-Host "📊 Found $($files.Count) files to process" -ForegroundColor Cyan

# Process each file
$progress = 0
foreach ($file in $files) {
    $progress++
    $percent = [math]::Round(($progress / $files.Count) * 100)
    Write-Progress -Activity "Processing files" -Status "$percent% Complete" -PercentComplete $percent
    
    Process-File -FilePath $file.FullName
}

Write-Progress -Activity "Processing files" -Completed

# Display summary
Write-Host "`n📊 Summary Report" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan
Write-Host "Files Processed:        $($stats.FilesProcessed)" -ForegroundColor White
Write-Host "Total Fixes Applied:    $($stats.ErrorsFixed)" -ForegroundColor $(if ($stats.ErrorsFixed -gt 0) { 'Green' } else { 'Gray' })
Write-Host "Deprecated Events:      $($stats.DeprecatedEventsFixed)" -ForegroundColor $(if ($stats.DeprecatedEventsFixed -gt 0) { 'Yellow' } else { 'Gray' })
Write-Host "Unused CSS Removed:     $($stats.UnusedCSSRemoved)" -ForegroundColor $(if ($stats.UnusedCSSRemoved -gt 0) { 'Yellow' } else { 'Gray' })
Write-Host "Imports Fixed:          $($stats.ImportsFixed)" -ForegroundColor $(if ($stats.ImportsFixed -gt 0) { 'Magenta' } else { 'Gray' })
Write-Host "Type Errors Fixed:      $($stats.TypeErrorsFixed)" -ForegroundColor $(if ($stats.TypeErrorsFixed -gt 0) { 'Cyan' } else { 'Gray' })

if ($Backup -and -not $DryRun) {
    Write-Host "Backups Created:        $($stats.BackupsCreated)" -ForegroundColor Blue
    Write-Host "`n📁 Backups saved to: $backupDir" -ForegroundColor Blue
}

if ($DryRun) {
    Write-Host "`n⚠️  DRY RUN MODE - No files were actually modified" -ForegroundColor Yellow
    Write-Host "Remove -DryRun flag to apply fixes" -ForegroundColor Yellow
}

# Run npm check after fixes
if (-not $DryRun -and $stats.ErrorsFixed -gt 0) {
    Write-Host "`n🔄 Running npm check to verify fixes..." -ForegroundColor Cyan
    Set-Location $Path
    npm run check 2>&1 | Select-Object -Last 20
    Set-Location ..
}

Write-Host "`n✨ Done!" -ForegroundColor Green