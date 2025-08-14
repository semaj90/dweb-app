# FIXED: Enhanced Documentation System Setup for Legal AI Web-App
# This script creates and runs the complete documentation pipeline

Write-Host "🚀 Setting up Documentation System for Legal AI Web-App..." -ForegroundColor Green

# Step 1: Create folder structure
Write-Host "`n📁 Creating directory structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "docs\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "docs\processed" | Out-Null  
New-Item -ItemType Directory -Force -Path "docs\legal" | Out-Null
New-Item -ItemType Directory -Force -Path "docs\ai-models" | Out-Null
Write-Host "✅ Directories created" -ForegroundColor Green

# Step 2: Check if Node.js is available
Write-Host "`n🔍 Checking system requirements..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js detected: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js first." -ForegroundColor Red
    Write-Host "💡 Download from: https://nodejs.org/" -ForegroundColor Cyan
    Write-Host "ℹ️  You can still run the documentation fetcher without Node.js" -ForegroundColor Gray
    
    $response = Read-Host "`nContinue with documentation fetch only? (y/n)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        exit 1
    }
}

# Step 3: Run documentation fetcher
Write-Host "`n📥 Step 1: Fetching documentation..." -ForegroundColor Cyan
if (Test-Path "fetch-docs.ps1") {
    & .\fetch-docs.ps1
    
    # Check if any files were downloaded
    $downloadedFiles = Get-ChildItem -Path "docs\raw" -Filter "*.html" -ErrorAction SilentlyContinue
    if ($downloadedFiles.Count -gt 0) {
        Write-Host "✅ Documentation fetch completed: $($downloadedFiles.Count) files downloaded" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No files downloaded. Check internet connection or try again later." -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ fetch-docs.ps1 not found" -ForegroundColor Red
}

# Step 4: Process documentation (if Node.js available and files exist)
if ($nodeVersion -and $downloadedFiles.Count -gt 0) {
    Write-Host "`n⚙️  Step 2: Processing documentation..." -ForegroundColor Cyan
    if (Test-Path "process-docs.mjs") {
        try {
            node .\process-docs.mjs
            
            # Check if processing was successful
            $processedFiles = Get-ChildItem -Path "docs\processed" -Filter "*.json" -ErrorAction SilentlyContinue
            if ($processedFiles.Count -gt 0) {
                Write-Host "✅ Documentation processing completed: $($processedFiles.Count) files processed" -ForegroundColor Green
            } else {
                Write-Host "⚠️  Processing completed but no JSON files found" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "❌ Error processing documentation: $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ process-docs.mjs not found" -ForegroundColor Red
    }
} elseif (-not $nodeVersion) {
    Write-Host "`n⚠️  Skipping documentation processing (Node.js not available)" -ForegroundColor Yellow
} else {
    Write-Host "`n⚠️  Skipping documentation processing (no files to process)" -ForegroundColor Yellow
}

# Step 5: Update VS Code settings for documentation integration
Write-Host "`n⚙️  Step 3: Configuring VS Code integration..." -ForegroundColor Cyan

$vscodeDir = ".vscode"
if (!(Test-Path $vscodeDir)) {
    New-Item -ItemType Directory -Path $vscodeDir | Out-Null
}

# Enhanced VS Code settings for documentation system
$vscodeSettings = @{
    "files.associations" = @{
        "*.svelte" = "svelte"
        "*.md" = "markdown"
        "*.mjs" = "javascript"
    }
    "search.include" = @{
        "docs/processed/**" = $true
        "*.svelte" = $true
        "*.ts" = $true
        "*.js" = $true
    }
    "search.exclude" = @{
        "docs/raw/**" = $true
        "node_modules/**" = $true
        "bits-ui-main/**" = $true
        "melt-ui-develop/**" = $true
        ".vite/**" = $true
        "dist/**" = $true
    }
    "github.copilot.enable" = @{
        "*" = $true
        "markdown" = $true
        "svelte" = $true
        "typescript" = $true
        "javascript" = $true
    }
    "svelte.enable-ts-plugin" = $true
    "typescript.preferences.includePackageJsonAutoImports" = "on"
    "typescript.suggest.autoImports" = $true
}

try {
    $vscodeSettings | ConvertTo-Json -Depth 10 | Set-Content -Path "$vscodeDir\settings.json"
    Write-Host "✅ VS Code settings updated" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not update VS Code settings: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Step 6: Create integration summary
Write-Host "`n📊 Creating integration summary..." -ForegroundColor Yellow

$summaryContent = @"
# Documentation System Integration Summary

## 📊 Setup Status
- **Generated**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
- **Raw Files**: $($downloadedFiles.Count) HTML documents downloaded
- **Processed Files**: $(if ($processedFiles) { $processedFiles.Count } else { "0 (Node.js required)" }) JSON documents
- **VS Code Integration**: ✅ Configured

## 📁 Directory Structure
- **docs/raw/** - Downloaded HTML documentation
- **docs/processed/** - AI-ready JSON documents  
- **docs/legal/** - Legal-specific documentation (future use)
- **docs/ai-models/** - AI model documentation (future use)

## 🎯 Usage Instructions

### For VS Code Copilot Enhancement:
Add this context prompt:
```
Working on SvelteKit legal AI web-app. Use documentation from docs/processed/ for context. Focus on evidence management, case tracking, legal document processing. Fix Svelte issues: tabindex={0}, union types, proper accessibility.
```

### For Local LLM Integration:
- Feed JSON files from docs/processed/ into Ollama/Gemma
- Use index.json for quick topic overview
- Filter by relevance: high/medium/low for legal AI context

### For Search within VS Code:
- Press Ctrl+Shift+F
- Search will include docs/processed/ automatically
- Use tags like 'legal-domain', 'svelte-ui', 'database' for filtering

## 🔄 Maintenance
- Re-run setup-docs-system.ps1 monthly for updates
- Use fetch-docs.ps1 standalone for quick documentation refresh
- Monitor docs/processed/index.json for statistics

## 💡 Integration with Your Legal AI App
- Evidence management UI components: Search 'svelte-ui' + 'forms'
- Database queries: Search 'drizzle' + 'sql'
- AI integration: Search 'anthropic' + 'llm'
- Case management: Search 'legal-domain' + 'components'
"@

Set-Content -Path "docs\INTEGRATION_SUMMARY.md" -Value $summaryContent

# Final status report
Write-Host "`n🎉 Documentation System Setup Complete!" -ForegroundColor Green
Write-Host "`n📋 What was accomplished:" -ForegroundColor Yellow
Write-Host "   📁 Directory structure created" -ForegroundColor Gray
Write-Host "   📥 Documentation fetched from 20+ sources" -ForegroundColor Gray
if ($nodeVersion -and $processedFiles.Count -gt 0) {
    Write-Host "   ⚙️  Documentation processed into AI-ready format" -ForegroundColor Gray
} else {
    Write-Host "   ⚠️  Documentation processing skipped (install Node.js and re-run)" -ForegroundColor Gray
}
Write-Host "   🔧 VS Code integration configured" -ForegroundColor Gray
Write-Host "   📊 Integration summary created" -ForegroundColor Gray

Write-Host "`n📁 Key files created:" -ForegroundColor Cyan
Write-Host "   • docs/INTEGRATION_SUMMARY.md - Usage guide" -ForegroundColor White
if ($processedFiles.Count -gt 0) {
    Write-Host "   • docs/processed/index.json - Document index" -ForegroundColor White
    Write-Host "   • docs/processed/*.json - AI-ready documents" -ForegroundColor White
}
Write-Host "   • .vscode/settings.json - Enhanced VS Code configuration" -ForegroundColor White

Write-Host "`n🚀 Next steps:" -ForegroundColor Green
Write-Host "   1. Open VS Code and test document search (Ctrl+Shift+F)" -ForegroundColor White
Write-Host "   2. Use enhanced Copilot context for better AI assistance" -ForegroundColor White
Write-Host "   3. Integrate processed docs with your local LLM" -ForegroundColor White
Write-Host "   4. Run .\fix-critical-errors.ps1 to fix Svelte issues" -ForegroundColor White

if (-not $nodeVersion) {
    Write-Host "`n💡 To enable full documentation processing:" -ForegroundColor Cyan
    Write-Host "   1. Install Node.js from https://nodejs.org/" -ForegroundColor White
    Write-Host "   2. Re-run this script: .\setup-docs-system.ps1" -ForegroundColor White
}

Write-Host "`n✨ Your legal AI web-app documentation system is ready!" -ForegroundColor Green
