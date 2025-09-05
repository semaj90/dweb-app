#!/usr/bin/env powershell
# Enhanced RAG System - Production Status Check
# Verifies all components are running and accessible

Write-Host "🚀 Enhanced RAG System - Production Status Check" -ForegroundColor Cyan
Write-Host "Date: $(Get-Date)" -ForegroundColor Gray
Write-Host ""

# Check if Docker Desktop is running
Write-Host "🐳 Checking Docker Status..." -ForegroundColor Yellow
try {
    $dockerProcess = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
    if ($dockerProcess) {
        Write-Host "   ✅ Docker Desktop is running" -ForegroundColor Green

        # Check Docker containers
        $containers = docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Docker CLI accessible" -ForegroundColor Green
            Write-Host "   Containers:" -ForegroundColor Gray
            Write-Host $containers -ForegroundColor Gray
        } else {
            Write-Host "   ⚠️ Docker CLI not accessible - services may not be running" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   ⚠️ Docker Desktop not running - some features may be limited" -ForegroundColor Yellow
        Write-Host "   💡 To start Docker: Start → Docker Desktop" -ForegroundColor Cyan
    }
} catch {
    Write-Host "   ⚠️ Could not check Docker status" -ForegroundColor Yellow
}

Write-Host ""

# Check SvelteKit Development Server
Write-Host "🌐 Checking SvelteKit Server..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "   ✅ SvelteKit server running at http://localhost:5173" -ForegroundColor Green
    }
} catch {
    Write-Host "   ❌ SvelteKit server not accessible at http://localhost:5173" -ForegroundColor Red
    Write-Host "   💡 Run: npm run dev" -ForegroundColor Cyan
}

# Check RAG Studio
Write-Host "🧪 Checking RAG Studio..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5173/rag-studio" -TimeoutSec 5 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "   ✅ RAG Studio accessible at http://localhost:5173/rag-studio" -ForegroundColor Green
    }
} catch {
    Write-Host "   ❌ RAG Studio not accessible" -ForegroundColor Red
}

# Test API Endpoints
Write-Host "🔌 Testing API Endpoints..." -ForegroundColor Yellow

$endpoints = @(
    @{ Path = "/api/rag?action=status"; Name = "RAG Service" },
    @{ Path = "/api/libraries"; Name = "Library Sync" },
    @{ Path = "/api/orchestrator"; Name = "Agent Orchestrator" },
    @{ Path = "/api/evaluation"; Name = "Evaluation Service" }
)

foreach ($endpoint in $endpoints) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173$($endpoint.Path)" -TimeoutSec 3 -UseBasicParsing
        Write-Host "   ✅ $($endpoint.Name): HTTP $($response.StatusCode)" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️ $($endpoint.Name): Not responding" -ForegroundColor Yellow
    }
}

# Check MCP Server Configuration
Write-Host ""
Write-Host "🔧 Checking VS Code MCP Integration..." -ForegroundColor Yellow
$claudeConfigPath = "$env:APPDATA\Claude\claude_desktop_config.json"
if (Test-Path $claudeConfigPath) {
    Write-Host "   ✅ Claude desktop config found" -ForegroundColor Green
    $mcpServerPath = "C:/Users/james/Desktop/deeds-web/deeds-web-app/mcp/custom-context7-server.js"
    if (Test-Path $mcpServerPath) {
        Write-Host "   ✅ MCP server file exists" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️ MCP server file not found" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ⚠️ Claude desktop config not found" -ForegroundColor Yellow
}

# Check VS Code Extension
Write-Host ""
Write-Host "🎯 Checking VS Code Extension..." -ForegroundColor Yellow
$extensionPath = ".vscode\extensions\mcp-context7-assistant"
if (Test-Path $extensionPath) {
    Write-Host "   ✅ VS Code extension installed" -ForegroundColor Green
    $extensionJs = "$extensionPath\out\extension.js"
    if (Test-Path $extensionJs) {
        $size = (Get-Item $extensionJs).Length
        Write-Host "   ✅ Extension compiled ($([math]::Round($size/1024, 1))KB)" -ForegroundColor Green
    }
} else {
    Write-Host "   ⚠️ VS Code extension not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📊 System Summary:" -ForegroundColor Cyan
Write-Host "   🌐 Web Interface: http://localhost:5173" -ForegroundColor White
Write-Host "   🧪 RAG Studio: http://localhost:5173/rag-studio" -ForegroundColor White
Write-Host "   🔧 VS Code: Ctrl+Shift+P → 'Context7 MCP'" -ForegroundColor White
Write-Host "   📚 Redis UI: http://localhost:8001 (if Docker running)" -ForegroundColor White

Write-Host ""
Write-Host "🚀 Quick Start Commands:" -ForegroundColor Cyan
Write-Host "   npm run dev          # Start development server" -ForegroundColor White
Write-Host "   npm run start        # Start Docker services" -ForegroundColor White
Write-Host "   npm run status       # Check Docker status" -ForegroundColor White

Write-Host ""
Write-Host "✨ Enhanced RAG System Status Check Complete!" -ForegroundColor Green
