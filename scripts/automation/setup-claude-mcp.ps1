#!/usr/bin/env powershell
# Claude Desktop MCP Context7 Setup Script

Write-Host "Setting up Claude Desktop MCP Context7 Configuration..." -ForegroundColor Cyan
Write-Host ""

# Create Claude config directory if it doesn't exist
$claudeConfigDir = "$env:APPDATA\Claude"
if (-not (Test-Path $claudeConfigDir)) {
    Write-Host "Creating Claude config directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $claudeConfigDir | Out-Null
}

# Copy the configuration file
$sourceConfig = "claude_desktop_config.json"
$destConfig = "$claudeConfigDir\claude_desktop_config.json"

if (Test-Path $sourceConfig) {
    Write-Host "Copying Claude Desktop configuration..." -ForegroundColor Yellow
    Copy-Item $sourceConfig $destConfig -Force
    
    if (Test-Path $destConfig) {
        Write-Host "✅ Configuration copied successfully!" -ForegroundColor Green
        Write-Host "   Location: $destConfig" -ForegroundColor Gray
    } else {
        Write-Host "❌ Failed to copy configuration" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Source configuration file not found: $sourceConfig" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test Node.js environment
Write-Host "Testing Node.js environment..." -ForegroundColor Yellow

try {
    $nodeVersion = node --version 2>$null
    if ($nodeVersion) {
        Write-Host "✅ Node.js is available" -ForegroundColor Green
        Write-Host "   Version: $nodeVersion" -ForegroundColor Gray
    } else {
        throw "Node.js not found"
    }
} catch {
    Write-Host "❌ Node.js not found in PATH" -ForegroundColor Red
    Write-Host "   Please ensure Node.js is installed and in your PATH" -ForegroundColor Yellow
}

# Test npm
try {
    $npmVersion = npm --version 2>$null
    if ($npmVersion) {
        Write-Host "✅ npm is available" -ForegroundColor Green
        Write-Host "   Version: $npmVersion" -ForegroundColor Gray
    } else {
        throw "npm not found"
    }
} catch {
    Write-Host "❌ npm not found in PATH" -ForegroundColor Red
}

# Test npx
try {
    $npxVersion = npx --version 2>$null
    if ($npxVersion) {
        Write-Host "✅ npx is available" -ForegroundColor Green
        Write-Host "   Version: $npxVersion" -ForegroundColor Gray
    } else {
        throw "npx not found"
    }
} catch {
    Write-Host "❌ npx not found in PATH" -ForegroundColor Red
}

Write-Host ""

# Test Context7 server
Write-Host "Testing Context7 server..." -ForegroundColor Yellow
Write-Host "This may take a moment as it downloads the package..." -ForegroundColor Gray

try {
    $testResult = npx --yes @modelcontextprotocol/server-context7 --help 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Context7 server is accessible" -ForegroundColor Green
    } else {
        throw "Context7 server test failed"
    }
} catch {
    Write-Host "❌ Context7 server test failed" -ForegroundColor Red
    Write-Host "   This might be due to network issues or Node.js environment problems" -ForegroundColor Yellow
    Write-Host "   Error: $_" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Setup complete! Please:" -ForegroundColor Cyan
Write-Host "1. Restart Claude Desktop completely" -ForegroundColor White
Write-Host "2. Check if Context7 appears in your available tools" -ForegroundColor White
Write-Host "3. Test with a question about your project structure" -ForegroundColor White
Write-Host ""

# Display current configuration
Write-Host "Current configuration:" -ForegroundColor Cyan
if (Test-Path $destConfig) {
    Get-Content $destConfig | Write-Host -ForegroundColor Gray
}

Read-Host "Press Enter to continue"
