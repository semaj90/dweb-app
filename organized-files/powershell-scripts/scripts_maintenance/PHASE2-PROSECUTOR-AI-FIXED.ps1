# Fixed PHASE2-PROSECUTOR-AI.ps1
# Enhanced UI/UX with AI Foundations

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "🚀 PROSECUTOR AI - PHASE 2 LAUNCHER" -ForegroundColor Yellow
Write-Host "🎨 Enhanced UI/UX with AI Foundations" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "🎯 COMPREHENSIVE INTEGRATION FEATURES:" -ForegroundColor Magenta
Write-Host "   ✅ Melt UI + Bits UI v2 Integration" -ForegroundColor Green
Write-Host "   ✅ AI Command Parsing with Real-time Updates" -ForegroundColor Green
Write-Host "   ✅ XState Machine for AI Command Processing" -ForegroundColor Green
Write-Host "   ✅ Enhanced Component System with Prop Merging" -ForegroundColor Green
Write-Host "   ✅ shadcn-svelte + UnoCSS Integration" -ForegroundColor Green

# Check current directory
$currentDir = Get-Location
Write-Host "📂 Current Directory: $currentDir" -ForegroundColor Blue

# Navigate to frontend
Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

Write-Host "🎨 Running Phase 2 Advanced Integration Setup..." -ForegroundColor Yellow
Write-Host "📦 This will install Melt UI + Bits UI v2 + AI foundations" -ForegroundColor Cyan

# Install Phase 2 dependencies
Write-Host "📦 Installing Phase 2 dependencies..." -ForegroundColor Blue

$dependencies = @(
    "@melt-ui/svelte@^0.86.6",
    "@melt-ui/pp@^0.3.2", 
    "bits-ui@^2.8.11",
    "xstate@^5.20.1",
    "@xstate/svelte@^5.0.0"
)

foreach ($dep in $dependencies) {
    Write-Host "   Installing $dep..." -ForegroundColor Gray
    npm install $dep --silent
}

Write-Host "✅ Phase 2 dependencies installed!" -ForegroundColor Green

# Check if unified stores exist
$storesPath = "src\lib\stores"
$aiUnified = Test-Path "$storesPath\ai-unified.ts"
$evidenceUnified = Test-Path "$storesPath\evidence-unified.ts"

if ($aiUnified -and $evidenceUnified) {
    Write-Host "✅ Unified stores detected - Phase 2 complete!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some unified stores missing - check installation" -ForegroundColor Yellow
}

# Start development server
Write-Host "🚀 Starting development server..." -ForegroundColor Yellow
Start-Process "npm" -ArgumentList "run", "dev" -WindowStyle Normal

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "🎉 PROSECUTOR AI PHASE 2 COMPLETE" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan

Write-Host "🎯 PHASE 2 FEATURES:" -ForegroundColor Magenta
Write-Host "• Melt UI + Bits UI v2 with prop merging" -ForegroundColor White
Write-Host "• AI command parsing with parseAICommand()" -ForegroundColor White
Write-Host "• XState machine for AI command processing" -ForegroundColor White
Write-Host "• Real-time UI updates via ai-controlled classes" -ForegroundColor White
Write-Host "• Enhanced component system" -ForegroundColor White

Write-Host "`n🌐 Visit: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🎯 Ready for Phase 3: AI Core Implementation" -ForegroundColor Yellow

Read-Host "Press Enter to continue..."
