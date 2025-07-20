#!/usr/bin/env pwsh
# Test script for Gemma3 Local LLM integration
# This script tests the AI ask endpoint with Gemma3 local inference

Write-Host "Testing Gemma3 Local LLM Integration..." -ForegroundColor Green

# Configuration
$BaseUrl = "http://localhost:5173"
$ApiEndpoint = "$BaseUrl/api/ai/ask"

# Test queries for legal AI assistant
$TestQueries = @(
    @{
        query = "What are the key elements of a valid contract?"
        description = "Basic legal question about contracts"
    },
    @{
        query = "Analyze evidence for negligence in this case"
        description = "Evidence analysis query"
    },
    @{
        query = "What are the discovery obligations in this jurisdiction?"
        description = "Procedural law question"
    }
)

# Function to test AI endpoint
function Test-AIEndpoint {
    param(
        [string]$Query,
        [string]$Description
    )
    
    Write-Host "`nTesting: $Description" -ForegroundColor Cyan
    Write-Host "Query: $Query" -ForegroundColor Yellow
    
    $Body = @{
        query = $Query
        options = @{
            provider = "local"  # Force local Gemma3 usage
            maxSources = 3
            temperature = 0.7
            maxTokens = 512
        }
    } | ConvertTo-Json -Depth 3
    
    try {
        $Response = Invoke-RestMethod -Uri $ApiEndpoint -Method POST -Body $Body -ContentType "application/json" -TimeoutSec 30
        
        if ($Response.success) {
            Write-Host "✅ Success!" -ForegroundColor Green
            Write-Host "Provider: $($Response.data.provider)" -ForegroundColor Magenta
            Write-Host "Model: $($Response.data.model)" -ForegroundColor Magenta
            Write-Host "Confidence: $($Response.data.confidence)" -ForegroundColor Magenta
            Write-Host "Execution Time: $($Response.data.executionTime)ms" -ForegroundColor Magenta
            Write-Host "Answer Preview: $($Response.data.answer.Substring(0, [Math]::Min(150, $Response.data.answer.Length)))..." -ForegroundColor White
            Write-Host "Sources Found: $($Response.data.sources.Count)" -ForegroundColor Magenta
        } else {
            Write-Host "❌ Failed: $($Response.error)" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Request Failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Function to check if server is running
function Test-ServerAvailability {
    try {
        $Response = Invoke-WebRequest -Uri $BaseUrl -Method GET -TimeoutSec 5 -UseBasicParsing
        return $true
    } catch {
        return $false
    }
}

# Main execution
Write-Host "Checking server availability..." -ForegroundColor Yellow

if (-not (Test-ServerAvailability)) {
    Write-Host "❌ Server is not running at $BaseUrl" -ForegroundColor Red
    Write-Host "Please start the SvelteKit development server with:" -ForegroundColor Yellow
    Write-Host "  npm run dev" -ForegroundColor White
    exit 1
}

Write-Host "✅ Server is running" -ForegroundColor Green

# Test Gemma3 availability endpoint first
Write-Host "`nTesting Gemma3 availability..." -ForegroundColor Yellow
try {
    $GemmaTestResponse = Invoke-RestMethod -Uri "$BaseUrl/api/ai/test-gemma3" -Method POST -Body '{"prompt":"Test"}' -ContentType "application/json" -TimeoutSec 10
    
    if ($GemmaTestResponse.success) {
        Write-Host "✅ Gemma3 is available and responding" -ForegroundColor Green
        Write-Host "Model: $($GemmaTestResponse.data.model)" -ForegroundColor Magenta
    } else {
        Write-Host "⚠️  Gemma3 test failed: $($GemmaTestResponse.error)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Gemma3 test endpoint not available: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Run test queries
Write-Host "`n" + "="*60 -ForegroundColor Blue
Write-Host "Running AI Ask Endpoint Tests with Gemma3" -ForegroundColor Blue
Write-Host "="*60 -ForegroundColor Blue

foreach ($TestCase in $TestQueries) {
    Test-AIEndpoint -Query $TestCase.query -Description $TestCase.description
    Start-Sleep -Seconds 2  # Small delay between requests
}

Write-Host "`n" + "="*60 -ForegroundColor Blue
Write-Host "Testing Complete!" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Blue

Write-Host "`nTo test manually, visit:" -ForegroundColor Yellow
Write-Host "  $BaseUrl/test-gemma3" -ForegroundColor White
Write-Host "`nOr use curl:" -ForegroundColor Yellow
Write-Host "  curl -X POST $ApiEndpoint -H 'Content-Type: application/json' -d '{\"query\":\"What is a contract?\",\"options\":{\"provider\":\"local\"}}'" -ForegroundColor White
