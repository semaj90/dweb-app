# Legal AI System - Comprehensive Test Runner (PowerShell)
# Tests Ollama, PostgreSQL, pgvector, Drizzle ORM, CUDA, and SvelteKit integration

param(
    [switch]$Quick,
    [switch]$GPU,
    [switch]$Help,
    [switch]$StartServices,
    [string]$Suite = ""
)

# Colors for output
function Write-Header {
    param($message)
    Write-Host "`n$('=' * 60)" -ForegroundColor Cyan
    Write-Host "🚀 $message" -ForegroundColor Cyan
    Write-Host "$('=' * 60)" -ForegroundColor Cyan
}

function Write-Success {
    param($message)
    Write-Host "✅ $message" -ForegroundColor Green
}

function Write-Error {
    param($message)
    Write-Host "❌ $message" -ForegroundColor Red
}

function Write-Warning {
    param($message)
    Write-Host "⚠️  $message" -ForegroundColor Yellow
}

function Write-Info {
    param($message)
    Write-Host "ℹ️  $message" -ForegroundColor Blue
}

if ($Help) {
    Write-Header "Legal AI System Test Runner"
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor White
    Write-Host "  .\run-tests.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor White
    Write-Host "  -Quick            Run essential tests only" -ForegroundColor White
    Write-Host "  -GPU              Run GPU/CUDA tests only" -ForegroundColor White
    Write-Host "  -StartServices    Start required services first" -ForegroundColor White
    Write-Host "  -Suite <name>     Run specific test suite" -ForegroundColor White
    Write-Host "  -Help             Show this help" -ForegroundColor White
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor White
    Write-Host "  .\run-tests.ps1 -StartServices" -ForegroundColor Gray
    Write-Host "  .\run-tests.ps1 -Quick" -ForegroundColor Gray
    Write-Host "  .\run-tests.ps1 -GPU" -ForegroundColor Gray
    Write-Host "  .\run-tests.ps1 -Suite token-usage" -ForegroundColor Gray
    Write-Host ""
    Write-Host "AVAILABLE TEST SUITES:" -ForegroundColor White
    Write-Host "  • token-usage       Token tracking and management" -ForegroundColor Gray
    Write-Host "  • rag-comprehensive Complete RAG system integration" -ForegroundColor Gray
    Write-Host "  • ollama-gpu        NVIDIA CUDA and GPU acceleration" -ForegroundColor Gray
    Write-Host "  • postgresql        Database and vector search" -ForegroundColor Gray
    Write-Host "  • performance       System performance and optimization" -ForegroundColor Gray
    exit 0
}

Write-Header "Legal AI System - Comprehensive Test Suite"
Write-Info "Testing Ollama, PostgreSQL, pgvector, Drizzle ORM, CUDA, and SvelteKit 2/Svelte 5"

# Start services if requested
if ($StartServices) {
    Write-Header "Starting Required Services"

    try {
        Write-Info "Starting Ollama service..."
        npm run ollama:start
        Start-Sleep 10

        Write-Info "Starting development environment..."
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev" -WindowStyle Minimized
        Start-Sleep 5

        Write-Success "Services started successfully"
    } catch {
        Write-Error "Failed to start services: $($_.Exception.Message)"
        exit 1
    }
}

# Pre-flight checks
Write-Header "Pre-flight System Checks"

$checks = @(
    @{ Name = "Node.js"; Command = "node"; Args = @("--version"); Required = $true },
    @{ Name = "npm"; Command = "npm"; Args = @("--version"); Required = $true },
    @{ Name = "Playwright"; Command = "npx"; Args = @("playwright", "--version"); Required = $true },
    @{ Name = "Docker"; Command = "docker"; Args = @("--version"); Required = $false }
)

$allPassed = $true

foreach ($check in $checks) {
    try {
        $result = & $check.Command $check.Args 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "$($check.Name) is available"
        } else {
            throw "Command failed"
        }
    } catch {
        if ($check.Required) {
            Write-Error "$($check.Name) is required but not available"
            $allPassed = $false
        } else {
            Write-Warning "$($check.Name) is not available (optional)"
        }
    }
}

# Check required files
$requiredFiles = @(
    "package.json",
    "playwright.config.ts",
    "tests\comprehensive-rag-system.spec.ts",
    "tests\token-usage.spec.ts",
    "sveltekit-frontend\src\routes\ai-demo\+page.svelte"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Success "$file exists"
    } else {
        Write-Error "$file is missing"
        $allPassed = $false
    }
}

if (-not $allPassed) {
    Write-Error "Pre-flight checks failed. Please resolve issues before running tests."
    exit 1
}

# Service health checks
Write-Header "Service Health Checks"

$services = @(
    @{ Name = "SvelteKit Dev Server"; URL = "http://localhost:5173"; Required = $false },
    @{ Name = "Ollama"; URL = "http://localhost:11434/api/tags"; Required = $true }
)

foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri $service.URL -Method GET -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "$($service.Name) is healthy"
        } else {
            Write-Warning "$($service.Name) responded with status $($response.StatusCode)"
        }
    } catch {
        if ($service.Required) {
            Write-Error "$($service.Name) is not available (required)"
            if (-not $StartServices) {
                Write-Info "Try running with -StartServices to start required services"
            }
        } else {
            Write-Warning "$($service.Name) is not available"
        }
    }
}

# Define test suites
$testSuites = @()

if ($Suite) {
    # Run specific suite
    switch ($Suite.ToLower()) {
        "token-usage" {
            $testSuites += @{ Name = "Token Usage Management"; Command = "npm"; Args = @("run", "test:token-usage") }
        }
        "rag-comprehensive" {
            $testSuites += @{ Name = "Comprehensive RAG System"; Command = "npm"; Args = @("run", "test:rag:comprehensive") }
        }
        "ollama-gpu" {
            $testSuites += @{ Name = "Ollama GPU Acceleration"; Command = "npm"; Args = @("run", "test:ollama") }
        }
        "postgresql" {
            $testSuites += @{ Name = "PostgreSQL Integration"; Command = "npm"; Args = @("run", "test:postgresql") }
        }
        "performance" {
            $testSuites += @{ Name = "Performance Tests"; Command = "npm"; Args = @("run", "test:performance") }
        }
        default {
            Write-Error "Unknown test suite: $Suite"
            exit 1
        }
    }
} elseif ($Quick) {
    # Quick tests
    $testSuites += @{ Name = "Token Usage Management"; Command = "npm"; Args = @("run", "test:token-usage") }
    $testSuites += @{ Name = "Essential RAG Tests"; Command = "npm"; Args = @("run", "test:rag") }
    Write-Info "Running quick tests only"
} elseif ($GPU) {
    # GPU tests only
    $testSuites += @{ Name = "Ollama GPU Acceleration"; Command = "npm"; Args = @("run", "test:ollama") }
    $testSuites += @{ Name = "CUDA Performance"; Command = "npm"; Args = @("run", "test:cuda") }
    Write-Info "Running GPU tests only"
} else {
    # Full test suite
    $testSuites += @{ Name = "Token Usage Management"; Command = "npm"; Args = @("run", "test:token-usage") }
    $testSuites += @{ Name = "Comprehensive RAG System"; Command = "npm"; Args = @("run", "test:rag:comprehensive") }
    $testSuites += @{ Name = "Ollama GPU Acceleration"; Command = "npm"; Args = @("run", "test:ollama") }
    $testSuites += @{ Name = "PostgreSQL Integration"; Command = "npm"; Args = @("run", "test:postgresql") }
    $testSuites += @{ Name = "Performance Tests"; Command = "npm"; Args = @("run", "test:performance") }
}

# Run test suites
$results = @()
$startTime = Get-Date

foreach ($suite in $testSuites) {
    Write-Header "$($suite.Name) Tests"

    try {
        $suiteStartTime = Get-Date
        & $suite.Command $suite.Args

        if ($LASTEXITCODE -eq 0) {
            $duration = (Get-Date) - $suiteStartTime
            Write-Success "$($suite.Name) tests completed in $($duration.TotalMilliseconds)ms"
            $results += @{ Name = $suite.Name; Success = $true; Duration = $duration.TotalMilliseconds }
        } else {
            throw "Tests failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-Error "$($suite.Name) tests failed: $($_.Exception.Message)"
        $results += @{ Name = $suite.Name; Success = $false; Error = $_.Exception.Message }
    }

    # Brief pause between suites
    Start-Sleep 2
}

# Generate report
$totalDuration = (Get-Date) - $startTime
$passed = $results | Where-Object { $_.Success -eq $true }
$failed = $results | Where-Object { $_.Success -ne $true }

Write-Header "Test Results Summary"

Write-Host "`n📊 Test Statistics:" -ForegroundColor Cyan
Write-Host "   Total Suites: $($results.Count)"
Write-Host "   Passed: $($passed.Count)" -ForegroundColor Green
Write-Host "   Failed: $($failed.Count)" -ForegroundColor $(if ($failed.Count -gt 0) { "Red" } else { "Green" })
Write-Host "   Total Duration: $($totalDuration.TotalSeconds) seconds"

if ($passed.Count -gt 0) {
    Write-Host "`n✅ Passed Tests:" -ForegroundColor Green
    $passed | ForEach-Object {
        Write-Host "   • $($_.Name) ($($_.Duration)ms)"
    }
}

if ($failed.Count -gt 0) {
    Write-Host "`n❌ Failed Tests:" -ForegroundColor Red
    $failed | ForEach-Object {
        Write-Host "   • $($_.Name): $($_.Error)"
    }
}

# Save detailed report
$reportPath = "test-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
$report = @{
    timestamp = (Get-Date).ToString("o")
    summary = @{
        total = $results.Count
        passed = $passed.Count
        failed = $failed.Count
        duration = $totalDuration.TotalMilliseconds
    }
    results = $results
} | ConvertTo-Json -Depth 4

$report | Out-File -FilePath $reportPath -Encoding UTF8
Write-Info "Detailed report saved to: $reportPath"

# Final status
if ($failed.Count -eq 0) {
    Write-Header "🎉 All Tests Passed!"
    Write-Success "Legal AI system is fully operational and ready for production."
    Write-Host ""
    Write-Host "🚀 NEXT STEPS:" -ForegroundColor Cyan
    Write-Host "1. Start full development stack: npm run dev:full" -ForegroundColor White
    Write-Host "2. Access AI demo: http://localhost:5173/ai-demo" -ForegroundColor White
    Write-Host "3. Test token usage slider and RAG capabilities" -ForegroundColor White
    Write-Host "4. Deploy to production: npm run deploy:optimized" -ForegroundColor White
    exit 0
} else {
    Write-Header "⚠️  Some Tests Failed"
    Write-Error "Please review the failed tests and resolve issues."
    Write-Host ""
    Write-Host "🔧 TROUBLESHOOTING:" -ForegroundColor Yellow
    Write-Host "1. Ensure all services are running: .\run-tests.ps1 -StartServices" -ForegroundColor White
    Write-Host "2. Check Ollama status: npm run ollama:status" -ForegroundColor White
    Write-Host "3. Verify GPU drivers: npm run test:gpu-only" -ForegroundColor White
    Write-Host "4. Review test logs above for specific error details" -ForegroundColor White
    exit 1
}
