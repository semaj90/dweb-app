# File Organization Script - Organize files by type while preserving originals
# Created: September 4, 2025
# Purpose: Organize 17,584 project files into type-specific directories

param(
    [switch]$DryRun,
    [switch]$Copy,  # Copy mode (preserve originals)
    [switch]$Move   # Move mode (relocate files)
)

$basePath = "C:\Users\james\Desktop\deeds-web\deeds-web-app"
$organizedPath = Join-Path $basePath "organized-files"

# File type mappings
$fileTypes = @{
    "markdown-docs" = @("*.md")
    "batch-scripts" = @("*.bat", "*.cmd")
    "java-archives" = @("*.jar")
    "json-configs" = @("*.json")
    "powershell-scripts" = @("*.ps1")
    "text-files" = @("*.txt")
    "go-source" = @("*.go")
    "backup-files" = @("*.backup*")
    "javascript-modules" = @("*.js", "*.mjs")
    "typescript-source" = @("*.ts")
    "svelte-components" = @("*.svelte")
}

Write-Host "🚀 Starting File Organization Process" -ForegroundColor Cyan
Write-Host "📂 Base path: $basePath" -ForegroundColor Yellow
Write-Host "📁 Organized path: $organizedPath" -ForegroundColor Yellow
Write-Host "🔧 Mode: $(if ($Move) { 'MOVE' } else { 'COPY' }) $(if ($DryRun) { '(DRY RUN)' })" -ForegroundColor Green
Write-Host ""

$totalFiles = 0
$organizedFiles = 0

foreach ($category in $fileTypes.Keys) {
    $categoryPath = Join-Path $organizedPath $category

    Write-Host "📂 Processing category: $category" -ForegroundColor Magenta

    # Ensure category directory exists
    if (-not (Test-Path $categoryPath) -and -not $DryRun) {
        New-Item -ItemType Directory -Path $categoryPath -Force | Out-Null
    }

    $categoryCount = 0

    foreach ($pattern in $fileTypes[$category]) {
        Write-Host "  🔍 Searching for: $pattern" -ForegroundColor Gray

        # Find files matching pattern (exclude node_modules, .git, organized-files)
        $files = Get-ChildItem -Path $basePath -Recurse -Filter $pattern | Where-Object {
            $_.FullName -notlike "*node_modules*" -and
            $_.FullName -notlike "*.git*" -and
            $_.FullName -notlike "*organized-files*" -and
            $_.FullName -notlike "*build*" -and
            $_.FullName -notlike "*dist*"
        }

        foreach ($file in $files) {
            $categoryCount++
            $totalFiles++

            # Create subdirectory structure based on original location
            $relativePath = $file.FullName.Replace($basePath, "").TrimStart('\')
            $relativeDir = Split-Path $relativePath -Parent
            $fileName = $file.Name

            # Clean directory name for organization
            $cleanDir = $relativeDir -replace '[\\/:*?"<>|]', '_'
            $targetDir = Join-Path $categoryPath $cleanDir
            $targetFile = Join-Path $targetDir $fileName

            if ($DryRun) {
                Write-Host "    📋 Would $(if ($Move) { 'move' } else { 'copy' }): $relativePath -> $category\$cleanDir\$fileName" -ForegroundColor DarkGray
            } else {
                try {
                    # Ensure target directory exists
                    $targetDirPath = Split-Path $targetFile -Parent
                    if (-not (Test-Path $targetDirPath)) {
                        New-Item -ItemType Directory -Path $targetDirPath -Force | Out-Null
                    }

                    if ($Move) {
                        Move-Item -Path $file.FullName -Destination $targetFile -Force
                        Write-Host "    📦 Moved: $fileName" -ForegroundColor Green
                    } else {
                        Copy-Item -Path $file.FullName -Destination $targetFile -Force
                        Write-Host "    📄 Copied: $fileName" -ForegroundColor Blue
                    }

                    $organizedFiles++
                } catch {
                    Write-Host "    ❌ Error with $fileName : $($_.Exception.Message)" -ForegroundColor Red
                }
            }
        }
    }

    Write-Host "  ✅ ${category}: $categoryCount files" -ForegroundColor Green
    Write-Host ""
}

Write-Host "🎉 Organization Complete!" -ForegroundColor Cyan
Write-Host "📊 Total files found: $totalFiles" -ForegroundColor Yellow
Write-Host "📁 Files organized: $organizedFiles" -ForegroundColor Green
Write-Host "🔧 Mode used: $(if ($Move) { 'MOVED' } else { 'COPIED' })" -ForegroundColor Magenta

if ($DryRun) {
    Write-Host ""
    Write-Host "🔍 This was a DRY RUN - no files were actually moved/copied" -ForegroundColor Yellow
    Write-Host "💡 Run without -DryRun to perform actual organization" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "📂 Organized files location: $organizedPath" -ForegroundColor Green
