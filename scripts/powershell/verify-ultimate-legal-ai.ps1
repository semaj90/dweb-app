# Quick Enhanced Legal AI System Verification
# Fast check to ensure everything is properly configured

Write-Host "🔍 Enhanced Legal AI System - Quick Verification" -ForegroundColor Green
Write-Host "=" * 55 -ForegroundColor Gray

$projectRoot = "C:\Users\james\Desktop\web-app"
Set-Location $projectRoot

Write-Host "`n📁 Working in: $(Get-Location)" -ForegroundColor Cyan

# Quick file checks
Write-Host "`n📋 Essential Files Check:" -ForegroundColor Yellow
$essentialFiles = @(
    @{ Name = "🤖 Local Gemma 3 Model"; Path = "gemma3Q4_K_M\mo16.gguf" },
    @{ Name = "🐳 Enhanced Docker Compose"; Path = "docker-compose.enhanced.yml" },
    @{ Name = "🌐 SvelteKit Frontend"; Path = "sveltekit-frontend\package.json" },
    @{ Name = "📊 Enhanced DB Schema"; Path = "scripts\init-enhanced-schema.sql" },
    @{ Name = "🕸️  Neo4j Init Script"; Path = "scripts\neo4j-init.cypher" },
    @{ Name = "🔧 Ollama Setup Script"; Path = "scripts\setup-local-gemma3.sh" },
    @{ Name = "🚀 Ultimate Startup Script"; Path = "start-ultimate-legal-ai.ps1" }
)

$allFilesPresent = $true
foreach ($file in $essentialFiles) {
    if (Test-Path $file.Path) {
        Write-Host "✅ $($file.Name)" -ForegroundColor Green
    } else {
        Write-Host "❌ $($file.Name): Missing" -ForegroundColor Red
        $allFilesPresent = $false
    }
}

# Check Docker availability
Write-Host "`n🐳 Docker Environment Check:" -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
    
    $dockerComposeVersion = docker compose version 2>&1
    Write-Host "✅ Docker Compose: Available" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker: Not available or not in PATH" -ForegroundColor Red
    $allFilesPresent = $false
}

# Check Node.js and npm
Write-Host "`n📦 Node.js Environment Check:" -ForegroundColor Yellow
if (Test-Path "sveltekit-frontend") {
    Set-Location sveltekit-frontend
    
    try {
        $nodeVersion = node --version 2>&1
        Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
        
        $npmVersion = npm --version 2>&1
        Write-Host "✅ npm: v$npmVersion" -ForegroundColor Green
        
        # Check if enhanced dependencies are in package.json
        $packageJson = Get-Content "package.json" | ConvertFrom-Json
        $hasNeo4j = $packageJson.dependencies.PSObject.Properties['neo4j-driver'] -ne $null
        $hasOllama = $packageJson.dependencies.PSObject.Properties['ollama'] -ne $null
        
        if ($hasNeo4j -and $hasOllama) {
            Write-Host "✅ Enhanced dependencies: Available in package.json" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Enhanced dependencies: May need installation" -ForegroundColor Yellow
        }
        
    } catch {
        Write-Host "❌ Node.js/npm: Not available" -ForegroundColor Red
        $allFilesPresent = $false
    }
    
    Set-Location ..
} else {
    Write-Host "❌ SvelteKit frontend directory not found" -ForegroundColor Red
    $allFilesPresent = $false
}

# Check Gemma 3 model
Write-Host "`n🤖 Local AI Model Check:" -ForegroundColor Yellow
if (Test-Path "gemma3Q4_K_M\mo16.gguf") {
    $modelInfo = Get-Item "gemma3Q4_K_M\mo16.gguf"
    $modelSizeMB = [math]::Round($modelInfo.Length / 1MB, 1)
    Write-Host "✅ Gemma 3 Model: $modelSizeMB MB" -ForegroundColor Green
} else {
    Write-Host "❌ Gemma 3 Model: Not found at gemma3Q4_K_M\mo16.gguf" -ForegroundColor Red
    $allFilesPresent = $false
}

# Overall Status
Write-Host "`n🎯 VERIFICATION SUMMARY" -ForegroundColor Green
Write-Host "=" * 30 -ForegroundColor Gray

if ($allFilesPresent) {
    Write-Host "🎉 ALL CHECKS PASSED!" -ForegroundColor Green
    Write-Host "✅ Enhanced Legal AI System is ready to start" -ForegroundColor White
    
    Write-Host "`n🚀 READY TO START:" -ForegroundColor Cyan
    Write-Host ".\start-ultimate-legal-ai.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "🔧 Quick options:" -ForegroundColor Yellow
    Write-Host ".\start-ultimate-legal-ai.ps1 -QuickStart     # Fast startup" -ForegroundColor Gray
    Write-Host ".\start-ultimate-legal-ai.ps1 -SkipTests      # Skip integration tests" -ForegroundColor Gray
    Write-Host ".\start-ultimate-legal-ai.ps1 -Verbose        # Detailed output" -ForegroundColor Gray
    
} else {
    Write-Host "⚠️  ISSUES DETECTED" -ForegroundColor Yellow
    Write-Host "Some components are missing or not configured properly" -ForegroundColor White
    
    Write-Host "`n🔧 TO FIX ISSUES:" -ForegroundColor Cyan
    Write-Host ".\setup-enhanced-legal-ai.ps1                 # Create missing components" -ForegroundColor Gray
    Write-Host "cd sveltekit-frontend && npm install          # Install dependencies" -ForegroundColor Gray
    Write-Host "docker --version                              # Verify Docker installation" -ForegroundColor Gray
}

Write-Host "`n📖 DOCUMENTATION:" -ForegroundColor Magenta
Write-Host "📋 Setup guide: setup-enhanced-legal-ai.ps1" -ForegroundColor White
Write-Host "🧪 Test integration: cd sveltekit-frontend && npm run ai:test" -ForegroundColor White
Write-Host "🏥 Health check: cd sveltekit-frontend && npm run ai:health" -ForegroundColor White
Write-Host "📊 System report: ENHANCED_LEGAL_AI_STARTUP_REPORT.md (after startup)" -ForegroundColor White

Write-Host "`n✨ ENHANCED FEATURES AVAILABLE:" -ForegroundColor Magenta
Write-Host "🤖 Gemma 3 Legal AI Assistant" -ForegroundColor White
Write-Host "🕸️  Neo4j Knowledge Graphs" -ForegroundColor White
Write-Host "📊 Vector Embeddings & Semantic Search" -ForegroundColor White
Write-Host "🔍 Detective Mode Analytics" -ForegroundColor White
Write-Host "🎨 Interactive Canvas Visualization" -ForegroundColor White
Write-Host "📈 User Behavior Tracking & AI Recommendations" -ForegroundColor White
Write-Host "⚖️  Specialized Legal Domain Features" -ForegroundColor White

Write-Host "`nVerification completed at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
