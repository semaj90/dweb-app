# PowerShell script to run all Playwright tests for the RAG application stack

Write-Host "🚀 Starting comprehensive test suite for RAG application..." -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check if Ollama is running
Write-Host "`n📍 Checking Ollama service..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Ollama is running" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Ollama is not running. Starting Ollama..." -ForegroundColor Red
    npm run ollama:start
    Start-Sleep -Seconds 5
}

# Check if the dev server is running
Write-Host "`n📍 Checking dev server..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5175" -UseBasicParsing -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Dev server is running" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Dev server not running. It will be started by Playwright." -ForegroundColor Yellow
}

# Run tests in sequence with detailed reporting
Write-Host "`n🧪 Running test suites..." -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Ollama Integration Tests
Write-Host "`n1️⃣ Running Ollama Integration Tests..." -ForegroundColor Magenta
npx playwright test ollama-integration.spec.ts --reporter=list

# 2. RAG System Tests
Write-Host "`n2️⃣ Running RAG System Tests..." -ForegroundColor Magenta
npx playwright test rag-system.spec.ts --reporter=list

# 3. PostgreSQL and pgvector Tests
Write-Host "`n3️⃣ Running PostgreSQL and pgvector Tests..." -ForegroundColor Magenta
npx playwright test postgresql-pgvector.spec.ts --reporter=list

# 4. Drizzle ORM Tests
Write-Host "`n4️⃣ Running Drizzle ORM Tests..." -ForegroundColor Magenta
npx playwright test drizzle-orm.spec.ts --reporter=list

# 5. RAG Pipeline Integration Tests
Write-Host "`n5️⃣ Running RAG Pipeline Integration Tests..." -ForegroundColor Magenta
npx playwright test rag-pipeline-integration.spec.ts --reporter=list

# 6. User Authentication and Session Tests
Write-Host "`n6️⃣ Running User Authentication and Session Tests..." -ForegroundColor Magenta
npx playwright test user-authentication-session.spec.ts --reporter=list

# 7. Global Store and State Management Tests
Write-Host "`n7️⃣ Running Global Store and State Management Tests..." -ForegroundColor Magenta
npx playwright test global-store-state.spec.ts --reporter=list

# 8. System Health and Logging Tests
Write-Host "`n8️⃣ Running System Health and Logging Tests..." -ForegroundColor Magenta
npx playwright test system-health-logging.spec.ts --reporter=list

# 9. Service Worker Tests
Write-Host "`n9️⃣ Running Service Worker Tests..." -ForegroundColor Magenta
npx playwright test service-worker.spec.ts --reporter=list

# 10. XState Machine Tests
Write-Host "`n🔟 Running XState Machine Tests..." -ForegroundColor Magenta
npx playwright test xstate-machines.spec.ts --reporter=list

# 11. LokiJS Caching Tests
Write-Host "`n1️⃣1️⃣ Running LokiJS Caching Tests..." -ForegroundColor Magenta
npx playwright test lokijs-caching.spec.ts --reporter=list

# 12. GPU Acceleration Tests (optional)
Write-Host "`n1️⃣2️⃣ Running GPU Acceleration Tests..." -ForegroundColor Magenta
try {
    npx playwright test gpu-acceleration.spec.ts --reporter=list
} catch {
    Write-Host "⚠️  GPU tests skipped or failed (this is OK if no GPU is available)" -ForegroundColor Yellow
}

# 13. GPU-Enabled Ollama Ingestion Tests
Write-Host "`n1️⃣3️⃣ Running GPU-Enabled Ollama Ingestion Tests..." -ForegroundColor Magenta
try {
    npx playwright test gpu-ollama-ingestion.spec.ts --reporter=list
} catch {
    Write-Host "⚠️  GPU ingestion tests skipped or failed (this is OK if no GPU is available)" -ForegroundColor Yellow
}

# Generate HTML report
Write-Host "`n📊 Generating test report..." -ForegroundColor Yellow
npx playwright show-report

Write-Host "`n✨ Test suite completed!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan