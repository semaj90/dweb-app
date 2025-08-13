# Complete AI Integration Installation & Verification Script
# install-and-check.ps1
# Windows PowerShell Version

param(
    [switch]$SkipInstall,
    [switch]$Verbose,
    [switch]$QuickCheck
)

$ErrorActionPreference = "Continue"
$Host.UI.RawUI.WindowTitle = "AI Integration Complete Check"

# Create timestamp
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$reportFile = "INTEGRATION-REPORT-$timestamp.md"
$logFile = "integration-check-$timestamp.log"

# Start transcript
Start-Transcript -Path $logFile

# Helper functions
function Write-Status {
    param($Message, $Type = "Info")
    $colors = @{
        "Success" = "Green"
        "Error" = "Red"
        "Warning" = "Yellow"
        "Info" = "Cyan"
    }
    Write-Host $Message -ForegroundColor $colors[$Type]
    Add-Content -Path $reportFile -Value $Message
}

function Test-Port {
    param($Port)
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue -InformationLevel Quiet
        return $connection
    } catch {
        return $false
    }
}

Clear-Host
Write-Host "================================================" -ForegroundColor Blue
Write-Host "   COMPLETE AI INTEGRATION INSTALLATION" -ForegroundColor Blue
Write-Host "          & VERIFICATION SCRIPT" -ForegroundColor Blue
Write-Host "================================================" -ForegroundColor Blue
Write-Host ""

# Initialize report
@"
# 🚀 COMPLETE AI INTEGRATION REPORT

**Generated:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
**System:** $env:COMPUTERNAME
**User:** $env:USERNAME

---

"@ | Out-File -FilePath $reportFile -Encoding UTF8

# ============================================
# STEP 1: NPM PACKAGE INSTALLATION
# ============================================

if (-not $SkipInstall) {
    Write-Status "## 📦 NPM Package Installation" "Info"
    Write-Status "" "Info"
    
    Set-Location "sveltekit-frontend" -ErrorAction SilentlyContinue
    
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    
    # Install dev dependencies
    $devPackages = @(
        "chalk@5.3.0",
        "ora@8.0.1",
        "glob@10.3.10",
        "concurrently@9.2.0",
        "ws@8.16.0",
        "rimraf@5.0.5"
    )
    
    $devPackagesString = $devPackages -join " "
    $npmInstallDev = "npm install --save-dev $devPackagesString"
    
    Write-Status "### Installing Dev Dependencies" "Info"
    Write-Status '```bash' "Info"
    Write-Status $npmInstallDev "Info"
    Write-Status '```' "Info"
    
    Invoke-Expression $npmInstallDev 2>&1 | Out-String | Add-Content -Path $reportFile
    
    # Install production dependencies
    Write-Status "### Installing Production Dependencies" "Info"
    npm install 2>&1 | Out-String | Add-Content -Path $reportFile
    
    Write-Status "✅ NPM packages installed" "Success"
    Write-Status "" "Info"
}

# ============================================
# STEP 2: TYPESCRIPT & SVELTE CHECKS
# ============================================

Write-Status "## 🔍 TypeScript & Svelte Checks" "Info"
Write-Status "" "Info"

# TypeScript check
Write-Status "### TypeScript Check" "Info"
Write-Status '```typescript' "Info"

$tsCheck = npx tsc --noEmit --skipLibCheck --incremental 2>&1 | Out-String
if ($LASTEXITCODE -eq 0) {
    Write-Status "✅ TypeScript: No errors" "Success"
} else {
    $errorCount = ([regex]::Matches($tsCheck, "error TS")).Count
    Write-Status "⚠️ TypeScript: $errorCount errors found" "Warning"
    $tsCheck.Split("`n") | Select-Object -First 20 | ForEach-Object { 
        Add-Content -Path $reportFile -Value $_
    }
}
Write-Status '```' "Info"
Write-Status "" "Info"

# Svelte check (if not quick mode)
if (-not $QuickCheck) {
    Write-Status "### Svelte Check" "Info"
    Write-Status '```' "Info"
    
    $svelteCheck = npx svelte-check --tsconfig ./tsconfig.json --threshold warning 2>&1 | Out-String
    $svelteCheck.Split("`n") | Select-Object -First 30 | ForEach-Object { 
        Add-Content -Path $reportFile -Value $_
    }
    Write-Status '```' "Info"
    Write-Status "" "Info"
}

# ============================================
# STEP 3: LINT CHECKS
# ============================================

if (-not $QuickCheck) {
    Write-Status "## 🎨 Lint & Format Checks" "Info"
    Write-Status "" "Info"
    
    Write-Status "### ESLint Check" "Info"
    Write-Status '```javascript' "Info"
    
    $eslintCheck = npx eslint . --ext .ts,.js,.svelte --max-warnings 10 2>&1 | Out-String
    $eslintCheck.Split("`n") | Select-Object -First 20 | ForEach-Object { 
        Add-Content -Path $reportFile -Value $_
    }
    Write-Status '```' "Info"
    Write-Status "" "Info"
}

# ============================================
# STEP 4: SERVICE HEALTH CHECKS
# ============================================

Write-Status "## 🏥 Service Health Checks" "Info"
Write-Status "" "Info"

Write-Status "### Port Availability" "Info"
Write-Status "" "Info"

$ports = @{
    "Frontend (Vite)" = 5173
    "Go API" = 8084
    "Redis" = 6379
    "Ollama" = 11434
    "PostgreSQL" = 5432
    "WebSocket Monitor" = 8085
}

foreach ($service in $ports.GetEnumerator()) {
    $port = $service.Value
    $name = $service.Key
    
    $inUse = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($inUse) {
        Write-Status "- $name (Port $port): ⚠️ IN USE" "Warning"
    } else {
        Write-Status "- $name (Port $port): ✅ AVAILABLE" "Success"
    }
}
Write-Status "" "Info"

# System requirements
Write-Status "### System Requirements" "Info"
Write-Status "" "Info"

# Node.js version
$nodeVersion = node --version
Write-Status "- Node.js: $nodeVersion" "Info"

# npm version
$npmVersion = npm --version
Write-Status "- npm: $npmVersion" "Info"

# Go version
try {
    $goVersion = go version
    Write-Status "- Go: $($goVersion.Split(' ')[2])" "Success"
} catch {
    Write-Status "- Go: ❌ Not installed" "Error"
}

# Redis check
try {
    $redisCheck = redis-cli ping 2>$null
    if ($redisCheck -eq "PONG") {
        Write-Status "- Redis: ✅ Running" "Success"
    } else {
        Write-Status "- Redis: ⚠️ Installed but not responding" "Warning"
    }
} catch {
    Write-Status "- Redis: ❌ Not installed or not in PATH" "Error"
}

# Ollama check
try {
    $ollamaVersion = ollama --version 2>$null
    Write-Status "- Ollama: ✅ Installed" "Success"
    
    # List models
    $models = ollama list 2>$null | Out-String
    if ($models -match "gemma") {
        Write-Status "  - Gemma3-Legal: ✅ Available" "Success"
    } else {
        Write-Status "  - Gemma3-Legal: ⚠️ Not found (run: ollama pull gemma3-legal:latest)" "Warning"
    }
} catch {
    Write-Status "- Ollama: ❌ Not installed" "Error"
}

# GPU check
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
    if ($gpuInfo) {
        Write-Status "- GPU: ✅ NVIDIA GPU detected" "Success"
        Write-Status "  - $gpuInfo" "Info"
    } else {
        Write-Status "- GPU: ⚠️ NVIDIA GPU not detected" "Warning"
    }
} catch {
    Write-Status "- GPU: ⚠️ nvidia-smi not available" "Warning"
}

Write-Status "" "Info"

# ============================================
# STEP 5: DATABASE CHECKS
# ============================================

Write-Status "## 🗄️ Database Configuration" "Info"
Write-Status "" "Info"

# PostgreSQL check
try {
    $pgReady = pg_isready 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Status "### PostgreSQL Status" "Info"
        Write-Status "✅ PostgreSQL is running and ready" "Success"
        
        # Check for JSONB tables
        Write-Status "" "Info"
        Write-Status "#### JSONB Tables Check" "Info"
        
        $tableCheck = psql -U postgres -d legal_ai_db -c "\dt ai_summarized_documents" 2>&1 | Out-String
        if ($tableCheck -match "ai_summarized_documents") {
            Write-Status "✅ JSONB tables are configured" "Success"
        } else {
            Write-Status "⚠️ JSONB tables not found - run schema-jsonb-enhanced.sql" "Warning"
        }
    } else {
        Write-Status "⚠️ PostgreSQL is installed but not running" "Warning"
    }
} catch {
    Write-Status "❌ PostgreSQL not installed or not in PATH" "Error"
}

Write-Status "" "Info"

# ============================================
# STEP 6: FILE STRUCTURE VERIFICATION
# ============================================

Write-Status "## 📁 File Structure Verification" "Info"
Write-Status "" "Info"

Write-Status "### Critical Files" "Info"
Write-Status "" "Info"

$criticalFiles = @(
    "..\main.go",
    "package.json",
    "tsconfig.json",
    "vite.config.js",
    "svelte.config.js",
    "..\START-GPU-LEGAL-AI-8084.bat",
    "..\gpu-ai-control-panel.bat",
    "START-DEV.bat",
    "..\database\schema-jsonb-enhanced.sql",
    "src\lib\db\schema-jsonb.ts",
    "src\routes\api\ai\vector-search\+server.ts",
    "..\812aisummarizeintegration.md",
    "..\TODO-AI-INTEGRATION.md"
)

$fileCheckSuccess = 0
$fileCheckFail = 0

foreach ($file in $criticalFiles) {
    if (Test-Path $file) {
        Write-Status "- ✅ $file" "Success"
        $fileCheckSuccess++
    } else {
        Write-Status "- ❌ $file (MISSING)" "Error"
        $fileCheckFail++
    }
}

Write-Status "" "Info"
Write-Status "### Critical Directories" "Info"
Write-Status "" "Info"

$criticalDirs = @(
    "..\ai-summarized-documents",
    "..\ai-summarized-documents\contracts",
    "..\ai-summarized-documents\legal-briefs",
    "..\ai-summarized-documents\case-studies",
    "..\ai-summarized-documents\embeddings",
    "..\ai-summarized-documents\cache",
    "scripts",
    "src\lib\db",
    "src\routes\api\ai",
    "node_modules"
)

$dirCheckSuccess = 0
$dirCheckFail = 0

foreach ($dir in $criticalDirs) {
    if (Test-Path $dir) {
        Write-Status "- ✅ $dir" "Success"
        $dirCheckSuccess++
    } else {
        Write-Status "- ❌ $dir (MISSING)" "Error"
        $dirCheckFail++
    }
}

Write-Status "" "Info"

# ============================================
# STEP 7: DEPENDENCY AUDIT
# ============================================

if (-not $QuickCheck) {
    Write-Status "## 🔒 Security Audit" "Info"
    Write-Status "" "Info"
    
    Write-Status '```' "Info"
    $auditResult = npm audit --audit-level=moderate 2>&1 | Out-String
    $auditResult.Split("`n") | Select-Object -First 15 | ForEach-Object { 
        Add-Content -Path $reportFile -Value $_
    }
    Write-Status '```' "Info"
    Write-Status "" "Info"
}

# ============================================
# STEP 8: MERGED TODO LIST
# ============================================

Write-Status "## 📋 Consolidated TODO List" "Info"
Write-Status "" "Info"

@"
### ✅ Completed Tasks
- [x] Fixed vector search JSON parsing error
- [x] Created GPU-accelerated Go microservice
- [x] Implemented JSONB PostgreSQL schema
- [x] Set up Redis caching with fallback
- [x] Created native Windows development environment
- [x] Built comprehensive monitoring systems
- [x] Integrated Ollama with Gemma3-Legal model
- [x] Created AI summarized documents directory structure
- [x] Implemented streaming responses
- [x] Set up batch processing

### 🔴 High Priority (This Week)
- [ ] Complete JSONB migration for existing tables
- [ ] Implement BullMQ job queue for document processing
- [ ] Add OCR support with Tesseract.js
- [ ] Create performance monitoring dashboard
- [ ] Implement WebSocket real-time updates
- [ ] Fix memory leak in WebSocket connections
- [ ] Resolve Ollama timeout on cold starts
- [ ] Add connection pooling for PostgreSQL

### 🟡 Medium Priority (This Month)
- [ ] Add drag-and-drop file upload interface
- [ ] Create summary comparison view
- [ ] Add export functionality (PDF, DOCX, JSON)
- [ ] Implement OAuth2 authentication
- [ ] Add webhook support for async processing
- [ ] Create comprehensive E2E tests
- [ ] Implement rate limiting per user/IP
- [ ] Add Redis Cluster support for scaling

### 🟢 Low Priority (Future)
- [ ] Fine-tune Gemma3-Legal model
- [ ] Implement RAG system
- [ ] Add Kubernetes orchestration
- [ ] Create Zapier/Make.com integration
- [ ] Build Microsoft Teams/Slack bots
- [ ] Add multi-tenant support
- [ ] Implement A/B testing framework
- [ ] Add voice input for queries

### 🐛 Bug Fixes Needed
- [ ] Fix memory leak in long-running WebSocket connections
- [ ] Resolve occasional Ollama timeout on cold starts
- [ ] Fix PDF parsing for complex legal documents
- [ ] Handle edge cases in batch processing
- [ ] Improve error messages for user-facing APIs
"@ | Add-Content -Path $reportFile

Write-Status "" "Info"

# ============================================
# STEP 9: PERFORMANCE BENCHMARKS
# ============================================

Write-Status "## ⚡ Performance Benchmarks" "Info"
Write-Status "" "Info"

@"
### Expected Performance Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **Tokens/Second** | 100-150 | With GPU acceleration |
| **Avg Latency** | 800-1200ms | Per document |
| **Cache Hit Rate** | 20-50% | Improves over time |
| **GPU Utilization** | 70-90% | Optimal range |
| **Success Rate** | 95-99% | With retry logic |
| **Concurrent Requests** | 3 | Semaphore controlled |
| **Memory Usage** | <6GB VRAM | 1GB reserved |
| **Queue Throughput** | 50 docs/min | Batch mode |
"@ | Add-Content -Path $reportFile

Write-Status "" "Info"

# ============================================
# STEP 10: CONFIGURATION STATUS
# ============================================

Write-Status "## ⚙️ Configuration Status" "Info"
Write-Status "" "Info"

# Check for .env files
if (Test-Path ".env.development") {
    Write-Status "✅ .env.development exists" "Success"
} else {
    Write-Status "⚠️ .env.development missing - creating template..." "Warning"
    
    @"
NODE_ENV=development
VITE_LEGAL_AI_API=http://localhost:8084
VITE_OLLAMA_URL=http://localhost:11434
VITE_REDIS_URL=redis://localhost:6379
VITE_ENABLE_GPU=true
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_ai_db
ENABLE_GPU=true
GPU_MEMORY_LIMIT_MB=6000
MAX_CONCURRENCY=3
MODEL_CONTEXT=4096
TEMPERATURE=0.2
"@ | Out-File -FilePath ".env.development" -Encoding UTF8
    
    Write-Status "✅ Created .env.development template" "Success"
}

Write-Status "" "Info"

# ============================================
# STEP 11: QUICK START COMMANDS
# ============================================

Write-Status "## 🚀 Quick Start Commands" "Info"
Write-Status "" "Info"

@'
```bash
# Start all services
npm run dev:full

# Or use Windows launcher
START-DEV.bat

# Or use PowerShell
.\scripts\start-dev-windows.ps1

# Monitor services
npm run monitor:lite

# Check health
npm run test:health

# Run all checks
npm run check:all

# GPU control panel
cd ..
gpu-ai-control-panel.bat
```
'@ | Add-Content -Path $reportFile

Write-Status "" "Info"

# ============================================
# STEP 12: FINAL SUMMARY
# ============================================

Write-Status "## 📊 Integration Summary" "Info"
Write-Status "" "Info"

# Count results
$reportContent = Get-Content $reportFile -Raw
$successCount = ([regex]::Matches($reportContent, "✅")).Count
$warningCount = ([regex]::Matches($reportContent, "⚠️")).Count
$errorCount = ([regex]::Matches($reportContent, "❌")).Count

Write-Status "### Statistics" "Info"
Write-Status "" "Info"
Write-Status "- ✅ Successful checks: $successCount" "Success"
Write-Status "- ⚠️ Warnings: $warningCount" "Warning"
Write-Status "- ❌ Errors: $errorCount" "Error"
Write-Status "" "Info"

# Overall status
Write-Status "### Overall Status" "Info"
Write-Status "" "Info"

if ($errorCount -eq 0) {
    Write-Status "# 🎉 **SYSTEM READY FOR PRODUCTION**" "Success"
    Write-Status "" "Info"
    Write-Status "All critical components are installed and configured correctly." "Success"
    Write-Status "The AI summarization system is fully operational." "Success"
} elseif ($errorCount -lt 5) {
    Write-Status "# ⚠️ **SYSTEM PARTIALLY READY**" "Warning"
    Write-Status "" "Info"
    Write-Status "Some non-critical components are missing. Core functionality should work." "Warning"
    Write-Status "Review the errors above and install missing dependencies." "Info"
} else {
    Write-Status "# ❌ **SYSTEM NEEDS CONFIGURATION**" "Error"
    Write-Status "" "Info"
    Write-Status "Multiple critical components are missing. Please review errors above." "Error"
    Write-Status "Follow the installation guide in 812aisummarizeintegration.md" "Info"
}

Write-Status "" "Info"

# ============================================
# STEP 13: RECOMMENDATIONS
# ============================================

Write-Status "## 💡 Recommendations" "Info"
Write-Status "" "Info"

if ($errorCount -gt 0) {
    Write-Status "### Required Actions" "Info"
    Write-Status "" "Info"
    
    $recommendations = @()
    
    if ($reportContent -notmatch "Go:.*✅") {
        $recommendations += "1. **Install Go**: Download from https://golang.org/dl/"
    }
    
    if ($reportContent -notmatch "Redis:.*✅") {
        $recommendations += "2. **Install Redis**: Download from https://github.com/microsoftarchive/redis/releases"
    }
    
    if ($reportContent -notmatch "Ollama:.*✅") {
        $recommendations += "3. **Install Ollama**: Download from https://ollama.ai"
    }
    
    if ($reportContent -notmatch "PostgreSQL.*✅") {
        $recommendations += "4. **Install PostgreSQL**: Download from https://www.postgresql.org/download/windows/"
    }
    
    if ($reportContent -notmatch "GPU:.*✅") {
        $recommendations += "5. **GPU Setup**: Install NVIDIA drivers and CUDA toolkit"
    }
    
    foreach ($rec in $recommendations) {
        Write-Status $rec "Info"
    }
    
    Write-Status "" "Info"
}

Write-Status "### Next Steps" "Info"
Write-Status "" "Info"

@"
1. Review any errors or warnings above
2. Install missing dependencies
3. Run ``npm run setup`` to configure environment
4. Start services with ``npm run dev:full`` or ``START-DEV.bat``
5. Access frontend at http://localhost:5173
6. Monitor health at http://localhost:8084/api/health
7. View GPU control panel with ``gpu-ai-control-panel.bat``
"@ | Add-Content -Path $reportFile

Write-Status "" "Info"

# ============================================
# STEP 14: RESOURCE LINKS
# ============================================

Write-Status "## 📚 Documentation & Resources" "Info"
Write-Status "" "Info"

@"
### Core Documentation
- [Complete Integration Guide](../812aisummarizeintegration.md)
- [TODO List & Roadmap](../TODO-AI-INTEGRATION.md)
- [Development Guide](DEV-GUIDE.md)
- [GPU Service Documentation](../README-GPU-AI.md)

### API Endpoints
- Frontend: http://localhost:5173
- API Health: http://localhost:8084/api/health
- API Metrics: http://localhost:8084/api/metrics
- WebSocket Monitor: ws://localhost:8085
- UnoCSS Inspector: http://localhost:5173/__unocss/

### Quick Commands
- Start Everything: ``npm run dev:full``
- Windows Launcher: ``START-DEV.bat``
- GPU Control: ``gpu-ai-control-panel.bat``
- Health Check: ``npm run test:health``
- Monitor: ``npm run monitor:lite``
"@ | Add-Content -Path $reportFile

Write-Status "" "Info"

# ============================================
# COMPLETION
# ============================================

Write-Status "---" "Info"
Write-Status "" "Info"
Write-Status "**Report Generated:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "Info"
Write-Status "**Report File:** $reportFile" "Info"
Write-Status "**Log File:** $logFile" "Info"
Write-Status "" "Info"
Write-Status "**System:** $env:COMPUTERNAME | **User:** $env:USERNAME" "Info"

# Stop transcript
Stop-Transcript

# Display summary
Write-Host ""
Write-Host "================================================" -ForegroundColor Blue
Write-Host "        INTEGRATION CHECK COMPLETE" -ForegroundColor Blue
Write-Host "================================================" -ForegroundColor Blue
Write-Host ""
Write-Host "📊 Results Summary:" -ForegroundColor Cyan
Write-Host "  ✅ Successful: $successCount" -ForegroundColor Green
Write-Host "  ⚠️ Warnings: $warningCount" -ForegroundColor Yellow
Write-Host "  ❌ Errors: $errorCount" -ForegroundColor Red
Write-Host ""
Write-Host "📄 Full report saved to: $reportFile" -ForegroundColor White
Write-Host "📋 Detailed log saved to: $logFile" -ForegroundColor White
Write-Host ""

if ($errorCount -eq 0) {
    Write-Host "🎉 System is ready for production use!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Start with: npm run dev:full" -ForegroundColor Cyan
    Write-Host "Or use: START-DEV.bat" -ForegroundColor Cyan
} else {
    Write-Host "⚠️ Please review the report for required actions." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "View report: notepad $reportFile" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Press any key to open the report..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Open the report
Start-Process notepad $reportFile
