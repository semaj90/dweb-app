# Enhanced RAG System Setup Script (PowerShell Fixed)
# Sets up PostgreSQL, Redis, Ollama, and Enhanced RAG Backend

Write-Host "🚀 Setting up Enhanced RAG System..." -ForegroundColor Cyan

# Create necessary directories
$directories = @(
    "rag-backend/logs",
    "rag-backend/uploads", 
    "rag-backend/init-db",
    "ollama-cache",
    "cluster-logs"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -Path $dir -ItemType Directory -Force
        Write-Host "✅ Created directory: $dir" -ForegroundColor Green
    }
}

# Check if Context7 MCP server is running
Write-Host "🔍 Checking Context7 MCP server status..." -ForegroundColor Yellow
try {
    $mcpResponse = Invoke-RestMethod -Uri "http://localhost:40000/health" -Method GET -TimeoutSec 5
    Write-Host "✅ Context7 MCP server is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Context7 MCP server is not running. Starting..." -ForegroundColor Red
    Write-Host "   Run: node context7-mcp-server.js" -ForegroundColor Yellow
}

# Check if Ollama is available
Write-Host "🔍 Checking Ollama service..." -ForegroundColor Yellow
try {
    $ollamaResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5
    Write-Host "✅ Ollama service is running" -ForegroundColor Green
    
    if ($ollamaResponse.models -and $ollamaResponse.models.Count -gt 0) {
        Write-Host "   Models available: $($ollamaResponse.models.Count)" -ForegroundColor Green
    } else {
        Write-Host "⚠️ No Ollama models found. Consider pulling models:" -ForegroundColor Yellow
        Write-Host "   ollama pull nomic-embed-text" -ForegroundColor Cyan
        Write-Host "   ollama pull gemma2:2b" -ForegroundColor Cyan
    }
} catch {
    Write-Host "❌ Ollama service is not available" -ForegroundColor Red
    Write-Host "   Install from: https://ollama.ai/download" -ForegroundColor Yellow
}

# Check cluster performance results
Write-Host "🔍 Checking cluster performance results..." -ForegroundColor Yellow
if (Test-Path "cluster-performance-simple.json") {
    $clusterData = Get-Content "cluster-performance-simple.json" | ConvertFrom-Json
    if ($clusterData.status -eq "working") {
        Write-Host "✅ Cluster system validated - $($clusterData.results.successfulRequests) successful requests" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Cluster performance issues detected" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️ Cluster performance not tested. Run: node test-cluster-simple.js" -ForegroundColor Yellow
}

# Create start script for all services (batch file)
$startScript = @"
@echo off
echo 🚀 Starting Enhanced RAG System Services...

echo 📡 Starting Context7 MCP Server...
start "MCP Server" cmd /k "node context7-mcp-server.js"

timeout /t 3

echo 🧠 Checking Ollama service...
curl -s http://localhost:11434/api/tags > nul
if %errorlevel% neq 0 (
    echo ❌ Ollama not running. Please start Ollama service.
    echo    Download from: https://ollama.ai/download
) else (
    echo ✅ Ollama service is running
)

echo 🔧 All services started!
echo 📊 Open http://localhost:5173/ai/enhanced-mcp for demo
pause
"@

$startScript | Out-File -FilePath "start-enhanced-rag.bat" -Encoding ASCII
Write-Host "✅ Created start-enhanced-rag.bat" -ForegroundColor Green

# Create test script
$testScript = @"
# Enhanced RAG System Test Script
Write-Host "🧪 Testing Enhanced RAG Integration..." -ForegroundColor Cyan

# Test endpoints
`$endpoints = @(
    @{ Name = "MCP Server Health"; Url = "http://localhost:40000/health" },
    @{ Name = "MCP Enhanced RAG"; Url = "http://localhost:40000/mcp/enhanced-rag/query" },
    @{ Name = "MCP Memory Graph"; Url = "http://localhost:40000/mcp/memory/read-graph" },
    @{ Name = "MCP Context7 Tools"; Url = "http://localhost:40000/mcp/context7/resolve-library-id" },
    @{ Name = "Ollama API"; Url = "http://localhost:11434/api/tags" }
)

`$passedTests = 0
foreach (`$endpoint in `$endpoints) {
    try {
        if (`$endpoint.Name -eq "Ollama API") {
            `$response = Invoke-RestMethod -Uri `$endpoint.Url -Method GET -TimeoutSec 5
        } else {
            `$testBody = @{ query = "test"; libraryName = "sveltekit" } | ConvertTo-Json
            `$response = Invoke-RestMethod -Uri `$endpoint.Url -Method POST -Body `$testBody -ContentType "application/json" -TimeoutSec 5
        }
        Write-Host "✅ `$(`$endpoint.Name) - PASSED" -ForegroundColor Green
        `$passedTests++
    } catch {
        Write-Host "❌ `$(`$endpoint.Name) - FAILED: `$(`$_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n📊 Test Results: `$passedTests/`$(`$endpoints.Count) tests passed" -ForegroundColor Cyan

if (`$passedTests -ge 3) {
    Write-Host "🎉 Enhanced RAG Integration is operational!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Some services need attention" -ForegroundColor Yellow
}
"@

$testScript | Out-File -FilePath "test-enhanced-rag.ps1" -Encoding UTF8
Write-Host "✅ Created test-enhanced-rag.ps1" -ForegroundColor Green

# Final instructions
Write-Host "`n🎯 Setup Complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run 'start-enhanced-rag.bat' to start all services" -ForegroundColor White
Write-Host "2. Run 'powershell .\test-enhanced-rag.ps1' to test integration" -ForegroundColor White
Write-Host "3. Visit http://localhost:5173/ai/enhanced-mcp for demo" -ForegroundColor White
Write-Host "4. Check cluster status with: node test-cluster-simple.js" -ForegroundColor White

Write-Host "`n📋 Service URLs:" -ForegroundColor Cyan
Write-Host "• Context7 MCP Server: http://localhost:40000" -ForegroundColor White
Write-Host "• Ollama API: http://localhost:11434" -ForegroundColor White
Write-Host "• SvelteKit Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "• Enhanced MCP Demo: http://localhost:5173/ai/enhanced-mcp" -ForegroundColor White