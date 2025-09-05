# PowerShell script to start the dev server and test the application
param(
    [switch]$SkipInstall,
    [switch]$TestOnly
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Deeds App Test Suite" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green

# Change to the web app directory
$webAppPath = "c:\Users\james\Downloads\Deeds-App-doesn-t-work--main (2)\web-app\sveltekit-frontend"
Set-Location $webAppPath

# Install dependencies if not skipped
if (-not $SkipInstall) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

# Start the development server if not test-only mode
if (-not $TestOnly) {
    Write-Host "🌐 Starting development server..." -ForegroundColor Yellow
    
    # Start the server in the background
    $serverJob = Start-Job -ScriptBlock {
        Set-Location $using:webAppPath
        npm run dev
    }
    
    Write-Host "⏱️ Waiting for server to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    
    # Test if server is running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5 -UseBasicParsing
        Write-Host "✅ Server is running on http://localhost:5173" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to connect to server" -ForegroundColor Red
        Stop-Job $serverJob -Force
        Remove-Job $serverJob -Force
        exit 1
    }
}

# Run the test suite
Write-Host "🧪 Running test suite..." -ForegroundColor Yellow
try {
    node test-app.mjs
    $testResult = $LASTEXITCODE
} catch {
    Write-Host "❌ Test suite failed to run: $_" -ForegroundColor Red
    $testResult = 1
}

# Cleanup
if (-not $TestOnly -and $serverJob) {
    Write-Host "🧹 Stopping development server..." -ForegroundColor Yellow
    Stop-Job $serverJob -Force
    Remove-Job $serverJob -Force
}

# Report results
if ($testResult -eq 0) {
    Write-Host "🎉 All tests passed!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Some tests failed" -ForegroundColor Red
}

Write-Host "
📋 Manual Testing Guide:
========================
1. Open http://localhost:5173 in your browser
2. Test user registration at /register
3. Test user login at /login
4. Test interactive canvas at /interactive-canvas
5. Check Ollama integration in AI features
6. Verify TipTap editor in rich text components

🔧 Troubleshooting:
==================
- If Ollama tests fail, run: ollama serve
- If canvas fails, check Fabric.js console errors
- If auth fails, check database connection
- For schema errors, run: npm run db:push
" -ForegroundColor Cyan

exit $testResult
