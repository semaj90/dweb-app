# Worker Threads + SIMD + Copilot Regex Demo Runner
# PowerShell version with enhanced error handling and reporting

param(
    [int]$DocumentCount = 50,
    [int]$WorkerCount = 4,
    [switch]$Verbose,
    [switch]$SaveResults = $true
)

Write-Host ""
Write-Host "🚀 Legal AI Document Processing Demo" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Demonstrating: Worker Threads + SIMD Parsers + Copilot Regex" -ForegroundColor Gray
Write-Host ""

# Check Node.js installation
Write-Host "📋 Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js version: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found! Please install Node.js to continue." -ForegroundColor Red
    exit 1
}

# Check Node.js version compatibility (should be 18+)
$versionNumber = $nodeVersion -replace "v", "" -replace "\.", ""
if ([int]$versionNumber.Substring(0, 2) -lt 18) {
    Write-Host "⚠️  Warning: Node.js 18+ recommended for worker_threads support" -ForegroundColor Yellow
}

# Check demo file exists
Write-Host ""
Write-Host "🔍 Checking demo file..." -ForegroundColor Yellow
if (-not (Test-Path "worker-simd-copilot-demo.mjs")) {
    Write-Host "❌ Demo file 'worker-simd-copilot-demo.mjs' not found!" -ForegroundColor Red
    Write-Host "   Please make sure the file exists in the current directory." -ForegroundColor Gray
    exit 1
}

$fileSize = (Get-Item "worker-simd-copilot-demo.mjs").Length
Write-Host "✅ Demo file found ($([math]::Round($fileSize/1024, 1)) KB)" -ForegroundColor Green

# Display configuration
Write-Host ""
Write-Host "⚙️  Configuration:" -ForegroundColor Yellow
Write-Host "   Documents to process: $DocumentCount" -ForegroundColor Gray
Write-Host "   Worker threads: $WorkerCount" -ForegroundColor Gray
Write-Host "   Verbose output: $Verbose" -ForegroundColor Gray
Write-Host "   Save results: $SaveResults" -ForegroundColor Gray

# Set environment variables for demo configuration
$env:DEMO_DOCUMENT_COUNT = $DocumentCount
$env:DEMO_WORKER_COUNT = $WorkerCount
$env:DEMO_VERBOSE = if ($Verbose) { "true" } else { "false" }

Write-Host ""
Write-Host "🏃 Starting demo..." -ForegroundColor Cyan
Write-Host ""

# Run the demo with error handling
try {
    $startTime = Get-Date

    if ($Verbose) {
        node --trace-warnings worker-simd-copilot-demo.mjs
    } else {
        node worker-simd-copilot-demo.mjs
    }

    $endTime = Get-Date
    $duration = $endTime - $startTime

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Demo completed successfully!" -ForegroundColor Green
        Write-Host "⏱️  Total runtime: $($duration.TotalSeconds.ToString('F2')) seconds" -ForegroundColor Gray

        # Check for results file
        if ($SaveResults -and (Test-Path "demo-results.json")) {
            $resultsSize = (Get-Item "demo-results.json").Length
            Write-Host "💾 Results saved: demo-results.json ($([math]::Round($resultsSize/1024, 1)) KB)" -ForegroundColor Gray

            # Parse and display key metrics
            try {
                $results = Get-Content "demo-results.json" | ConvertFrom-Json
                Write-Host ""
                Write-Host "📊 Key Metrics:" -ForegroundColor Yellow
                Write-Host "   Documents processed: $($results.configuration.documentCount)" -ForegroundColor Gray
                Write-Host "   Worker threads used: $($results.configuration.workerCount)" -ForegroundColor Gray

                if ($results.benchmark) {
                    $mainTime = [math]::Round($results.benchmark.mainThread.processingTime, 2)
                    $workerTime = [math]::Round($results.benchmark.workerThread.processingTime, 2)
                    $speedup = [math]::Round($results.benchmark.speedup, 2)

                    Write-Host "   Main thread time: ${mainTime}ms" -ForegroundColor Gray
                    Write-Host "   Worker thread time: ${workerTime}ms" -ForegroundColor Gray
                    Write-Host "   Speedup: ${speedup}x" -ForegroundColor Gray
                }
            } catch {
                Write-Host "   (Could not parse detailed metrics)" -ForegroundColor Gray
            }
        }

        Write-Host ""
        Write-Host "🎉 Demo Summary:" -ForegroundColor Cyan
        Write-Host "   • Worker threads enabled parallel document processing" -ForegroundColor Gray
        Write-Host "   • SIMD-style optimizations improved performance" -ForegroundColor Gray
        Write-Host "   • Copilot regex patterns extracted legal entities" -ForegroundColor Gray
        Write-Host "   • Integration shows practical AI development workflow" -ForegroundColor Gray

    } else {
        Write-Host ""
        Write-Host "❌ Demo failed with exit code: $LASTEXITCODE" -ForegroundColor Red

        # Suggest troubleshooting steps
        Write-Host ""
        Write-Host "🛠️  Troubleshooting suggestions:" -ForegroundColor Yellow
        Write-Host "   1. Check Node.js version (18+ recommended)" -ForegroundColor Gray
        Write-Host "   2. Ensure worker_threads support is enabled" -ForegroundColor Gray
        Write-Host "   3. Run with -Verbose flag for detailed error output" -ForegroundColor Gray
        Write-Host "   4. Check system memory availability" -ForegroundColor Gray
    }

} catch {
    Write-Host ""
    Write-Host "❌ Demo execution failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "🛠️  Error details:" -ForegroundColor Yellow
    Write-Host "   $($_.Exception.ToString())" -ForegroundColor Gray
}

# Cleanup environment variables
Remove-Item Env:DEMO_DOCUMENT_COUNT -ErrorAction SilentlyContinue
Remove-Item Env:DEMO_WORKER_COUNT -ErrorAction SilentlyContinue
Remove-Item Env:DEMO_VERBOSE -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
