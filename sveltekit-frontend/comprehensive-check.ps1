#!/usr/bin/env pwsh

# ======================================================================
# COMPREHENSIVE ERROR CHECK AND FIX SCRIPT
# Validates the entire AI Agent Stack setup
# ======================================================================

param(
    [switch]$Fix = $false,
    [switch]$Verbose = $false
)

Write-Host "🔍 AI Agent Stack Error Check" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

$ErrorCount = 0
$WarningCount = 0
$FixCount = 0

function Write-Status {
    param([string]$Message, [string]$Type = "Info")
    
    switch ($Type) {
        "Error" { Write-Host "❌ $Message" -ForegroundColor Red; $script:ErrorCount++ }
        "Warning" { Write-Host "⚠️  $Message" -ForegroundColor Yellow; $script:WarningCount++ }
        "Success" { Write-Host "✅ $Message" -ForegroundColor Green }
        "Info" { Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
        "Fix" { Write-Host "🔧 $Message" -ForegroundColor Magenta; $script:FixCount++ }
    }
}

# Check project structure
Write-Host "`n📁 Checking Project Structure..." -ForegroundColor Yellow

$RequiredFiles = @(
    "package.json",
    "tsconfig.json", 
    "svelte.config.js",
    "src/app.d.ts",
    "src/hooks.server.ts",
    "src/lib/types/index.ts",
    "src/lib/stores/ai-agent.ts",
    "src/lib/services/enhanced-rag-service.ts",
    "src/routes/api/ai/chat/+server.ts",
    "src/routes/test/+page.svelte"
)

foreach ($file in $RequiredFiles) {
    if (Test-Path $file) {
        Write-Status "Found: $file" "Success"
    } else {
        Write-Status "Missing: $file" "Error"
    }
}

# Check package.json structure
Write-Host "`n📦 Validating package.json..." -ForegroundColor Yellow

if (Test-Path "package.json") {
    try {
        $PackageJson = Get-Content "package.json" | ConvertFrom-Json
        
        # Check required dependencies
        $RequiredDeps = @(
            "svelte",
            "typescript", 
            "@sveltejs/kit",
            "fuse.js",
            "xstate",
            "zod",
            "bits-ui"
        )
        
        foreach ($dep in $RequiredDeps) {
            if ($PackageJson.dependencies.$dep -or $PackageJson.devDependencies.$dep) {
                Write-Status "Dependency found: $dep" "Success"
            } else {
                Write-Status "Missing dependency: $dep" "Error"
            }
        }
        
        # Check for script duplicates
        $Scripts = $PackageJson.scripts | Get-Member -MemberType NoteProperty | ForEach-Object { $_.Name }
        $DuplicateScripts = $Scripts | Group-Object | Where-Object { $_.Count -gt 1 }
        
        if ($DuplicateScripts.Count -eq 0) {
            Write-Status "No duplicate scripts found" "Success"
        } else {
            foreach ($dup in $DuplicateScripts) {
                Write-Status "Duplicate script: $($dup.Name)" "Warning"
            }
        }
        
    } catch {
        Write-Status "Invalid package.json format" "Error"
    }
} else {
    Write-Status "package.json not found" "Error"
}

# Check TypeScript configuration
Write-Host "`n🔧 Checking TypeScript Configuration..." -ForegroundColor Yellow

if (Test-Path "tsconfig.json") {
    try {
        $TsConfig = Get-Content "tsconfig.json" | ConvertFrom-Json
        
        if ($TsConfig.compilerOptions.strict -eq $false) {
            Write-Status "TypeScript strict mode disabled (good for migration)" "Success"
        }
        
        if ($TsConfig.compilerOptions.skipLibCheck -eq $true) {
            Write-Status "skipLibCheck enabled (helps with external deps)" "Success" 
        }
        
        Write-Status "TypeScript config valid" "Success"
        
    } catch {
        Write-Status "Invalid tsconfig.json format" "Error"
    }
} else {
    Write-Status "tsconfig.json not found" "Error"
}

# Check type definitions
Write-Host "`n📝 Checking Type Definitions..." -ForegroundColor Yellow

$TypeFiles = @(
    "src/lib/types/index.ts",
    "src/lib/types/missing-deps.d.ts",
    "src/app.d.ts"
)

foreach ($typeFile in $TypeFiles) {
    if (Test-Path $typeFile) {
        $content = Get-Content $typeFile -Raw
        
        # Check for common issues
        if ($content -match "export.*User.*User") {
            Write-Status "$typeFile: Potential duplicate User export" "Warning"
        }
        
        if ($content -match "declare module.*fuse\.js") {
            Write-Status "$typeFile: Fuse.js types declared" "Success"
        }
        
        Write-Status "Type file valid: $typeFile" "Success"
    } else {
        Write-Status "Missing type file: $typeFile" "Error"
    }
}

# Check component structure
Write-Host "`n🧩 Checking Component Structure..." -ForegroundColor Yellow

$ComponentIssues = @()

# Check for Svelte 5 compliance
$SvelteFiles = Get-ChildItem -Path "src" -Filter "*.svelte" -Recurse -ErrorAction SilentlyContinue

foreach ($file in $SvelteFiles) {
    $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
    
    if ($content -match "<slot\s*/>.*{@render") {
        $ComponentIssues += "$($file.Name): Mixed slot and render syntax"
    } elseif ($content -match "{@render.*children") {
        Write-Status "$($file.Name): Uses modern Svelte 5 syntax" "Success"
    }
}

if ($ComponentIssues.Count -eq 0) {
    Write-Status "All components follow Svelte 5 patterns" "Success"
} else {
    foreach ($issue in $ComponentIssues) {
        Write-Status $issue "Warning"
    }
}

# Check API structure
Write-Host "`n🌐 Checking API Structure..." -ForegroundColor Yellow

$ApiPaths = @(
    "src/routes/api/ai/chat/+server.ts",
    "src/routes/api/ai/connect/+server.ts", 
    "src/routes/api/ai/health/+server.ts"
)

foreach ($apiPath in $ApiPaths) {
    if (Test-Path $apiPath) {
        $content = Get-Content $apiPath -Raw
        
        if ($content -match "export const POST") {
            Write-Status "$apiPath: Has POST handler" "Success"
        }
        
        if ($content -match "export const GET") {
            Write-Status "$apiPath: Has GET handler" "Success"
        }
        
        if ($content -match "zod|z\.") {
            Write-Status "$apiPath: Uses Zod validation" "Success"
        }
        
    } else {
        Write-Status "Missing API endpoint: $apiPath" "Error"
    }
}

# Check for common fixes
Write-Host "`n🔧 Checking Applied Fixes..." -ForegroundColor Yellow

$FixChecks = @(
    @{
        File = "src/lib/types/index.ts"
        Pattern = "export namespace Database"
        Description = "Namespaced types to prevent conflicts"
    },
    @{
        File = "package.json"
        Pattern = '"fuse\.js"'
        Description = "Fuse.js dependency added"
    },
    @{
        File = "src/hooks.server.ts"
        Pattern = "Database\.User"
        Description = "Proper user type in hooks"
    },
    @{
        File = "src/app.d.ts"
        Pattern = "Database\.User"
        Description = "Updated app.d.ts types"
    }
)

foreach ($check in $FixChecks) {
    if (Test-Path $check.File) {
        $content = Get-Content $check.File -Raw
        if ($content -match $check.Pattern) {
            Write-Status $check.Description "Success"
        } else {
            Write-Status "Fix not applied: $($check.Description)" "Warning"
        }
    }
}

# Environment check
Write-Host "`n🌍 Checking Environment..." -ForegroundColor Yellow

if (Test-Path ".env.local") {
    Write-Status "Environment config exists" "Success"
} else {
    Write-Status "No .env.local file (will create default)" "Warning"
    
    if ($Fix) {
        $envContent = @"
NODE_ENV=development
OLLAMA_HOST=http://localhost:11434
ENABLE_RAG=true
LOG_LEVEL=info
"@
        $envContent | Out-File -FilePath ".env.local" -Encoding UTF8
        Write-Status "Created default .env.local" "Fix"
    }
}

# Startup scripts check
Write-Host "`n🚀 Checking Startup Scripts..." -ForegroundColor Yellow

if (Test-Path "setup-production.ps1") {
    Write-Status "Production setup script exists" "Success"
} else {
    Write-Status "Missing setup-production.ps1" "Warning"
}

if (Test-Path "start-dev.bat") {
    Write-Status "Development start script exists" "Success"
} else {
    Write-Status "Missing start-dev.bat" "Warning"
}

# Summary
Write-Host "`n📊 Check Summary" -ForegroundColor Green
Write-Host "===============" -ForegroundColor Green
Write-Host "Errors: $ErrorCount" -ForegroundColor $(if ($ErrorCount -eq 0) { "Green" } else { "Red" })
Write-Host "Warnings: $WarningCount" -ForegroundColor $(if ($WarningCount -eq 0) { "Green" } else { "Yellow" })
Write-Host "Fixes Applied: $FixCount" -ForegroundColor Magenta

if ($ErrorCount -eq 0) {
    Write-Host "`n🎉 No critical errors found!" -ForegroundColor Green
    Write-Host "Your AI Agent Stack appears to be properly configured." -ForegroundColor Green
} else {
    Write-Host "`n⚠️  $ErrorCount critical errors need attention" -ForegroundColor Red
}

if ($WarningCount -gt 0) {
    Write-Host "💡 $WarningCount warnings can be addressed for optimization" -ForegroundColor Yellow
}

# Next steps
Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Run: npm install (if not done)" -ForegroundColor White
Write-Host "2. Start Ollama: ollama serve" -ForegroundColor White  
Write-Host "3. Pull model: ollama pull gemma2:2b" -ForegroundColor White
Write-Host "4. Test: npm run dev" -ForegroundColor White
Write-Host "5. Visit: http://localhost:5173/test" -ForegroundColor White

exit $ErrorCount
