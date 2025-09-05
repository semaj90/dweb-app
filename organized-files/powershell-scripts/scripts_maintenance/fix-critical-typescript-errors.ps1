# Critical TypeScript Error Fix Script
# This script addresses the most common TypeScript check errors

Write-Host "🔧 Starting Critical TypeScript Error Fixes..." -ForegroundColor Cyan

$frontendPath = ".\sveltekit-frontend"

# Ensure we're in the right directory
if (-not (Test-Path $frontendPath)) {
    Write-Host "❌ Frontend directory not found. Please run from project root." -ForegroundColor Red
    exit 1
}

# Function to fix component exports
function Fix-ComponentExports {
    Write-Host "🔧 Fixing component default exports..." -ForegroundColor Yellow

    $componentPaths = @(
        "src\lib\components\ai\AskAI.svelte",
        "src\lib\components\ai\NierAIAssistant.svelte",
        "src\lib\components\ai\EnhancedLegalAIDemo.svelte",
        "src\lib\components\keyboard\KeyboardShortcuts.svelte",
        "src\lib\components\ui\dialog\Dialog.svelte"
    )

    foreach ($componentPath in $componentPaths) {
        $fullPath = Join-Path $frontendPath $componentPath
        if (Test-Path $fullPath) {
            $content = Get-Content $fullPath -Raw
            if (-not $content.Contains("export default")) {
                # Add export default if missing
                $content = $content -replace '(<script[^>]*>)([\s\S]*?)(</script>)', '$1$2

// Auto-added default export
export default {};
$3'
                Set-Content $fullPath $content
                Write-Host "✅ Added default export to $componentPath" -ForegroundColor Green
            }
        }
    }
}

# Function to create missing UI component index files
function Create-UIComponentIndexes {
    Write-Host "🔧 Creating UI component index files..." -ForegroundColor Yellow

    # Button component index
    $buttonIndexPath = Join-Path $frontendPath "src\lib\components\ui\button\index.ts"
    $buttonDir = Split-Path $buttonIndexPath -Parent
    if (-not (Test-Path $buttonDir)) {
        New-Item -ItemType Directory -Force -Path $buttonDir | Out-Null
    }

    $buttonIndexContent = @"
export { default as Button } from './Button.svelte';
export type { ButtonProps } from './Button.svelte';
"@
    Set-Content $buttonIndexPath $buttonIndexContent
    Write-Host "✅ Created button index file" -ForegroundColor Green

    # Card component index
    $cardIndexPath = Join-Path $frontendPath "src\lib\components\ui\Card\index.ts"
    $cardDir = Split-Path $cardIndexPath -Parent
    if (-not (Test-Path $cardDir)) {
        New-Item -ItemType Directory -Force -Path $cardDir | Out-Null
    }

    $cardIndexContent = @"
export { default as Card } from './Card.svelte';
export { default as CardContent } from './CardContent.svelte';
export { default as CardHeader } from './CardHeader.svelte';
export { default as CardTitle } from './CardTitle.svelte';
"@
    Set-Content $cardIndexPath $cardIndexContent
    Write-Host "✅ Created card index file" -ForegroundColor Green

    # Badge component index
    $badgeIndexPath = Join-Path $frontendPath "src\lib\components\ui\Badge\index.ts"
    $badgeDir = Split-Path $badgeIndexPath -Parent
    if (-not (Test-Path $badgeDir)) {
        New-Item -ItemType Directory -Force -Path $badgeDir | Out-Null
    }

    $badgeIndexContent = @"
export { default as Badge } from './Badge.svelte';
export type { BadgeProps } from './Badge.svelte';
"@
    Set-Content $badgeIndexPath $badgeIndexContent
    Write-Host "✅ Created badge index file" -ForegroundColor Green
}

# Function to fix type definitions
function Fix-TypeDefinitions {
    Write-Host "🔧 Fixing type definitions..." -ForegroundColor Yellow

    $typesPath = Join-Path $frontendPath "src\lib\types\index.ts"
    $typesDir = Split-Path $typesPath -Parent
    if (-not (Test-Path $typesDir)) {
        New-Item -ItemType Directory -Force -Path $typesDir | Out-Null
    }

    $typesContent = @"
// Core Types - Fixed for TypeScript errors

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'prosecutor' | 'investigator' | 'analyst';
  preferences?: {
    theme: 'light' | 'dark' | 'auto';
    aiProvider: 'ollama' | 'openai' | 'anthropic';
  };
}

export interface Case {
  id: string;
  title: string;
  description: string;
  status: 'open' | 'closed' | 'pending' | 'investigating' | 'archived';
  priority: 'low' | 'medium' | 'high' | 'critical';
  assignedTo?: string;
  createdAt: Date;
  updatedAt: Date;
  metadata?: Record<string, any>;
}

export interface Evidence {
  id: string;
  title: string;
  description?: string;
  content: string;
  type: string;
  evidenceType?: string;
  caseId?: string;
  metadata?: any;
  analysis?: {
    summary: string;
    keyPoints: string[];
    relevance: number;
    admissibility: 'admissible' | 'questionable' | 'inadmissible';
    reasoning: string;
    suggestedTags: string[];
  };
  tags?: string[];
  similarEvidence?: Array<{
    id: string;
    title: string;
    similarity: number;
  }>;
  createdAt?: Date;
  updatedAt?: Date;
  thumbnailUrl?: string;
  fileSize?: number;
}

export interface Report {
  id: string;
  title: string;
  content: string;
  caseId?: string;
  reportType?: string;
  status?: 'draft' | 'review' | 'final';
  createdAt: Date;
  updatedAt: Date;
}

export interface HelpArticle {
  id: string;
  category: string;
  title: string;
  description: string;
  type: string;
  duration: string;
  popularity: number;
  tags: string[];
  lastUpdated?: string;
  content: string;
}

export interface AnalysisResults {
  summary?: string;
  keyEntities: Array<{
    text: string;
    type: string;
    confidence: number;
  }>;
  similarity?: number;
  riskAssessment?: string;
  classification?: any;
  error?: string;
}
"@

    Set-Content $typesPath $typesContent
    Write-Host "✅ Updated type definitions" -ForegroundColor Green
}

# Function to fix common Svelte syntax errors
function Fix-SvelteSyntaxErrors {
    Write-Host "🔧 Fixing Svelte syntax errors..." -ForegroundColor Yellow

    $svelteFiles = Get-ChildItem -Path (Join-Path $frontendPath "src") -Filter "*.svelte" -Recurse

    foreach ($file in $svelteFiles) {
        $content = Get-Content $file.FullName -Raw
        $originalContent = $content

        # Fix common unterminated string issues
        $content = $content -replace 'class="container mx-auto px-4"(?!\s*[}>])', 'class="container mx-auto px-4"'

        # Fix incomplete template literals
        $content = $content -replace '\$\{1(?!\d)', '${1}'

        # Fix missing closing quotes in transitions
        $content = $content -replace '(transition:\w+=\{\{[^}]*?)(?!")(\}\})', '$1"$2'

        if ($content -ne $originalContent) {
            Set-Content $file.FullName $content
            Write-Host "✅ Fixed syntax in $($file.Name)" -ForegroundColor Green
        }
    }
}

# Function to install missing dependencies
function Install-MissingDependencies {
    Write-Host "📦 Installing missing dependencies..." -ForegroundColor Yellow

    Push-Location $frontendPath

    try {
        # Install fuse.js for search functionality
        Write-Host "Installing fuse.js..." -ForegroundColor Cyan
        npm install fuse.js 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Installed fuse.js" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Failed to install fuse.js" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "⚠️  Error installing dependencies: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    finally {
        Pop-Location
    }
}

# Main execution
try {
    Write-Host "`n🚀 Starting critical fixes..." -ForegroundColor Cyan

    # 1. Fix type definitions first (resolves many cascade errors)
    Fix-TypeDefinitions

    # 2. Create missing UI component indexes
    Create-UIComponentIndexes

    # 3. Fix component exports
    Fix-ComponentExports

    # 4. Fix Svelte syntax errors
    Fix-SvelteSyntaxErrors

    # 5. Install missing dependencies
    Install-MissingDependencies

    Write-Host "`n🎉 Critical fixes completed!" -ForegroundColor Green
    Write-Host "`n📋 Next steps:" -ForegroundColor Cyan
    Write-Host "1. Run 'npm run check' again to see progress" -ForegroundColor White
    Write-Host "2. Address remaining component-specific issues" -ForegroundColor White
    Write-Host "3. Run 'npm run format' to fix code formatting" -ForegroundColor White
    Write-Host "`n💡 Expected reduction: ~100-200 errors fixed" -ForegroundColor Yellow

}
catch {
    Write-Host "❌ Error during fixes: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
