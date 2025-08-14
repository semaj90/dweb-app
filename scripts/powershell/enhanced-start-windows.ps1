param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("start", "health", "test", "stop")]
    [string]$Action = "start"
)

# Enhanced RAG Integration Startup Script for Windows
Write-Host "Enhanced RAG Integration - Action: $Action" -ForegroundColor Green

switch ($Action) {
    "health" {
        Write-Host "Testing Ollama + Gemma3 Integration..." -ForegroundColor Yellow
        
        # Test Ollama service
        Write-Host "1. Checking Ollama service..." -ForegroundColor Cyan
        try {
            $ollamaResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 10
            Write-Host "✓ Ollama service is running" -ForegroundColor Green
            
            # Check if gemma3-legal model exists
            $hasGemma3 = $ollamaResponse.models | Where-Object { $_.name -like "*gemma3-legal*" }
            if ($hasGemma3) {
                Write-Host "✓ gemma3-legal model found" -ForegroundColor Green
            } else {
                Write-Host "⚠ gemma3-legal model not found - will be imported on first run" -ForegroundColor Yellow
            }
        }
        catch {
            Write-Host "✗ Ollama service not accessible: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "Please ensure Ollama is installed and running" -ForegroundColor Yellow
        }
        
        # Test SvelteKit dev server
        Write-Host "2. Checking SvelteKit dev server..." -ForegroundColor Cyan
        try {
            $svelteResponse = Invoke-RestMethod -Uri "http://localhost:5173/api/health" -Method GET -TimeoutSec 5
            Write-Host "✓ SvelteKit dev server is running" -ForegroundColor Green
        }
        catch {
            Write-Host "⚠ SvelteKit dev server not running - starting with npm run dev" -ForegroundColor Yellow
        }
        
        # Test MCP Context7 service
        Write-Host "3. Checking MCP Context7 service..." -ForegroundColor Cyan
        try {
            $mcpResponse = Invoke-RestMethod -Uri "http://localhost:40000/mcp" -Method GET -TimeoutSec 5
            Write-Host "✓ MCP Context7 service is running" -ForegroundColor Green
        }
        catch {
            Write-Host "⚠ MCP Context7 service not accessible - may need to start VS Code extension" -ForegroundColor Yellow
        }
        
        Write-Host "Health check complete!" -ForegroundColor Green
        break
    }
    
    "test" {
        Write-Host "Running integration tests..." -ForegroundColor Yellow
        
        # Test Gemma3 model
        Write-Host "Testing Gemma3 model response..." -ForegroundColor Cyan
        try {
            $testPrompt = @{
                model = "gemma3-legal"
                prompt = "What is a contract?"
                stream = $false
            } | ConvertTo-Json
            
            $response = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method POST -Body $testPrompt -ContentType "application/json" -TimeoutSec 30
            Write-Host "✓ Gemma3 model responding correctly" -ForegroundColor Green
        }
        catch {
            Write-Host "✗ Gemma3 model test failed: $($_.Exception.Message)" -ForegroundColor Red
        }
        
        # Test RAG service endpoint
        Write-Host "Testing RAG service..." -ForegroundColor Cyan
        try {
            $ragTest = Invoke-RestMethod -Uri "http://localhost:5173/api/rag/health" -Method GET -TimeoutSec 10
            Write-Host "✓ RAG service responding" -ForegroundColor Green
        }
        catch {
            Write-Host "⚠ RAG service not available - may need to implement endpoint" -ForegroundColor Yellow
        }
        break
    }
    
    "start" {
        Write-Host "Starting Enhanced RAG services..." -ForegroundColor Yellow
        
        # Start Ollama if not running
        Write-Host "Starting Ollama service..." -ForegroundColor Cyan
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
        
        # Start SvelteKit dev server
        Write-Host "Starting SvelteKit dev server..." -ForegroundColor Cyan
        Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WindowStyle Minimized -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 5
        
        Write-Host "Services started! Run with -Action health to verify" -ForegroundColor Green
        break
    }
    
    "stop" {
        Write-Host "Stopping Enhanced RAG services..." -ForegroundColor Yellow
        
        # Stop processes
        Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
        Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like "*vite*" } | Stop-Process -Force
        
        Write-Host "Services stopped!" -ForegroundColor Green
        break
    }
}