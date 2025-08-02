# Legal AI System - Comprehensive API Testing Script
# Tests all endpoints with JSON payloads and validates responses

param(
    [string]$BaseUrl = "http://localhost:5173",
    [switch]$HealthCheck = $false,
    [switch]$TestAll = $false,
    [switch]$Verbose = $false,
    [string]$OutputFile = "test-results.json"
)

$ErrorActionPreference = "Continue"
$script:TestResults = @{
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Tests = @()
    Summary = @{
        Total = 0
        Passed = 0
        Failed = 0
        Warnings = 0
    }
}

# Test configuration
$script:Config = @{
    Headers = @{
        "Content-Type" = "application/json"
        "Accept" = "application/json"
    }
    Timeout = 30
    VectorDimensions = 384
}

# Helper functions
function Write-TestResult {
    param(
        [string]$TestName,
        [string]$Status,
        [string]$Message,
        [object]$Details = $null
    )
    
    $color = switch ($Status) {
        "PASS" { "Green" }
        "FAIL" { "Red" }
        "WARN" { "Yellow" }
        "INFO" { "Cyan" }
        default { "White" }
    }
    
    Write-Host "[$Status] $TestName - $Message" -ForegroundColor $color
    
    if ($Verbose -and $Details) {
        Write-Host "Details: $($Details | ConvertTo-Json -Depth 3)" -ForegroundColor Gray
    }
    
    $script:TestResults.Tests += @{
        Name = $TestName
        Status = $Status
        Message = $Message
        Details = $Details
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    }
    
    $script:TestResults.Summary.Total++
    switch ($Status) {
        "PASS" { $script:TestResults.Summary.Passed++ }
        "FAIL" { $script:TestResults.Summary.Failed++ }
        "WARN" { $script:TestResults.Summary.Warnings++ }
    }
}

function Invoke-APITest {
    param(
        [string]$Endpoint,
        [string]$Method = "GET",
        [object]$Body = $null,
        [hashtable]$Headers = $script:Config.Headers,
        [int]$ExpectedStatus = 200
    )
    
    try {
        $params = @{
            Uri = "$BaseUrl$Endpoint"
            Method = $Method
            Headers = $Headers
            TimeoutSec = $script:Config.Timeout
        }
        
        if ($Body) {
            $params.Body = $Body | ConvertTo-Json -Depth 10
        }
        
        $response = Invoke-RestMethod @params -StatusCodeVariable statusCode
        
        return @{
            Success = $statusCode -eq $ExpectedStatus
            StatusCode = $statusCode
            Data = $response
            Error = $null
        }
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        return @{
            Success = $false
            StatusCode = $statusCode
            Data = $null
            Error = $_.Exception.Message
        }
    }
}

# Health Check Tests
function Test-SystemHealth {
    Write-Host "`n=== System Health Checks ===" -ForegroundColor Cyan
    
    # Test main health endpoint
    $result = Invoke-APITest -Endpoint "/api/health" -Method "GET"
    if ($result.Success) {
        Write-TestResult "API Health" "PASS" "API is responding" $result.Data
    } else {
        Write-TestResult "API Health" "FAIL" "API not responding" $result.Error
    }
    
    # Test Docker services
    $services = @{
        "PostgreSQL" = @{ Port = 5432; Container = "legal_ai_postgres" }
        "Redis" = @{ Port = 6379; Container = "legal_ai_redis" }
        "Qdrant" = @{ Port = 6333; Container = "legal_ai_qdrant" }
        "Ollama" = @{ Port = 11434; Container = "legal_ai_ollama" }
    }
    
    foreach ($service in $services.Keys) {
        $config = $services[$service]
        try {
            $connection = Test-NetConnection -ComputerName localhost -Port $config.Port -WarningAction SilentlyContinue -InformationLevel Quiet
            if ($connection) {
                # Additional Docker health check
                $health = docker inspect $config.Container --format='{{.State.Health.Status}}' 2>$null
                if ($health -eq "healthy" -or $health -eq $null) {
                    Write-TestResult "$service Health" "PASS" "Service is healthy on port $($config.Port)"
                } else {
                    Write-TestResult "$service Health" "WARN" "Service running but health status: $health"
                }
            } else {
                Write-TestResult "$service Health" "FAIL" "Service not accessible on port $($config.Port)"
            }
        } catch {
            Write-TestResult "$service Health" "FAIL" "Error checking service: $_"
        }
    }
    
    # Test Qdrant collections
    try {
        $collections = Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method GET
        $expectedCollections = @("legal_documents", "case_embeddings", "evidence_vectors")
        
        foreach ($expected in $expectedCollections) {
            if ($collections.result.collections.name -contains $expected) {
                Write-TestResult "Qdrant Collection: $expected" "PASS" "Collection exists"
            } else {
                Write-TestResult "Qdrant Collection: $expected" "FAIL" "Collection missing"
            }
        }
    } catch {
        Write-TestResult "Qdrant Collections" "FAIL" "Cannot access Qdrant: $_"
    }
    
    # Test Ollama models
    try {
        $models = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
        $requiredModels = @("nomic-embed-text", "gemma3-legal", "gemma:2b")
        
        foreach ($required in $requiredModels) {
            if ($models.models.name -contains $required -or $models.models.name -like "$required*") {
                Write-TestResult "Ollama Model: $required" "PASS" "Model available"
            } else {
                Write-TestResult "Ollama Model: $required" "WARN" "Model not found"
            }
        }
    } catch {
        Write-TestResult "Ollama Models" "FAIL" "Cannot access Ollama: $_"
    }
}

# API Endpoint Tests
function Test-CaseScoring {
    Write-Host "`n=== Case Scoring API Tests ===" -ForegroundColor Cyan
    
    # Test case scoring endpoint
    $testCase = @{
        case_id = [guid]::NewGuid().ToString()
        case_data = @{
            title = "Test Case - Fraud Investigation"
            description = "Complex financial fraud case involving multiple defendants"
            evidence_count = 15
            witness_count = 5
            complexity = "high"
        }
        scoring_criteria = @{
            evidence_strength = 0.8
            witness_reliability = 0.7
            legal_precedent = 0.9
            public_interest = 0.6
        }
    }
    
    $result = Invoke-APITest -Endpoint "/api/case-scoring" -Method "POST" -Body $testCase
    if ($result.Success -and $result.Data.score -ge 0 -and $result.Data.score -le 100) {
        Write-TestResult "Case Scoring" "PASS" "Score: $($result.Data.score)/100" $result.Data
    } else {
        Write-TestResult "Case Scoring" "FAIL" "Invalid response or score out of range" $result
    }
    
    # Test scoring with temperature parameter
    $testCase.temperature = 0.3
    $result = Invoke-APITest -Endpoint "/api/case-scoring" -Method "POST" -Body $testCase
    if ($result.Success) {
        Write-TestResult "Case Scoring (Low Temp)" "PASS" "Temperature control working" $result.Data
    } else {
        Write-TestResult "Case Scoring (Low Temp)" "FAIL" "Temperature parameter failed" $result
    }
}

function Test-VectorSearch {
    Write-Host "`n=== Vector Search API Tests ===" -ForegroundColor Cyan
    
    # Test document embedding
    $document = @{
        id = [guid]::NewGuid().ToString()
        content = "This is a test legal document about contract law and breach of contract cases."
        metadata = @{
            type = "contract"
            date = Get-Date -Format "yyyy-MM-dd"
        }
    }
    
    $result = Invoke-APITest -Endpoint "/api/documents/embed" -Method "POST" -Body $document
    if ($result.Success -and $result.Data.embedding -and $result.Data.embedding.Count -eq $Config.VectorDimensions) {
        Write-TestResult "Document Embedding" "PASS" "Generated 384-dim vector" @{dimensions = $result.Data.embedding.Count}
        
        # Store the document
        $storeResult = Invoke-APITest -Endpoint "/api/documents" -Method "POST" -Body @{
            document = $document
            embedding = $result.Data.embedding
        }
        
        if ($storeResult.Success) {
            Write-TestResult "Document Storage" "PASS" "Document stored with vector"
        } else {
            Write-TestResult "Document Storage" "FAIL" "Failed to store document" $storeResult
        }
    } else {
        Write-TestResult "Document Embedding" "FAIL" "Invalid embedding dimensions" $result
    }
    
    # Test similarity search
    $searchQuery = @{
        query = "contract breach litigation"
        limit = 5
        threshold = 0.7
    }
    
    $result = Invoke-APITest -Endpoint "/api/documents/search" -Method "POST" -Body $searchQuery
    if ($result.Success -and $result.Data.results) {
        Write-TestResult "Vector Search" "PASS" "Found $($result.Data.results.Count) similar documents" $result.Data
    } else {
        Write-TestResult "Vector Search" "FAIL" "Search failed" $result
    }
}

function Test-AIChat {
    Write-Host "`n=== AI Chat API Tests ===" -ForegroundColor Cyan
    
    # Test basic chat
    $chatRequest = @{
        messages = @(
            @{
                role = "user"
                content = "What are the key elements of a valid contract?"
            }
        )
        model = "gemma3-legal"
        temperature = 0.7
        max_tokens = 500
    }
    
    $result = Invoke-APITest -Endpoint "/api/ai/chat" -Method "POST" -Body $chatRequest
    if ($result.Success -and $result.Data.response) {
        Write-TestResult "AI Chat" "PASS" "Response received" @{responseLength = $result.Data.response.Length}
    } else {
        Write-TestResult "AI Chat" "FAIL" "No response from AI" $result
    }
    
    # Test with context
    $chatRequest.context = @{
        case_id = [guid]::NewGuid().ToString()
        relevant_documents = @("doc1", "doc2")
    }
    
    $result = Invoke-APITest -Endpoint "/api/ai/chat" -Method "POST" -Body $chatRequest
    if ($result.Success) {
        Write-TestResult "AI Chat with Context" "PASS" "Contextual response received"
    } else {
        Write-TestResult "AI Chat with Context" "FAIL" "Context handling failed" $result
    }
}

function Test-EvidenceSynthesis {
    Write-Host "`n=== Evidence Synthesis API Tests ===" -ForegroundColor Cyan
    
    $synthesisRequest = @{
        case_id = [guid]::NewGuid().ToString()
        evidence_items = @(
            @{
                id = "ev1"
                type = "document"
                content = "Financial records showing suspicious transactions"
                relevance = 0.9
            },
            @{
                id = "ev2"
                type = "testimony"
                content = "Witness statement about defendant's behavior"
                relevance = 0.7
            }
        )
        synthesis_type = "summary"
    }
    
    $result = Invoke-APITest -Endpoint "/api/evidence/synthesize" -Method "POST" -Body $synthesisRequest
    if ($result.Success -and $result.Data.synthesis) {
        Write-TestResult "Evidence Synthesis" "PASS" "Synthesis generated" $result.Data
    } else {
        Write-TestResult "Evidence Synthesis" "FAIL" "Synthesis failed" $result
    }
}

function Test-DatabaseOperations {
    Write-Host "`n=== Database Operation Tests ===" -ForegroundColor Cyan
    
    # Test case creation
    $newCase = @{
        title = "API Test Case $(Get-Random -Maximum 9999)"
        description = "Automated test case"
        status = "open"
        priority = "medium"
    }
    
    $result = Invoke-APITest -Endpoint "/api/cases" -Method "POST" -Body $newCase
    if ($result.Success -and $result.Data.id) {
        Write-TestResult "Case Creation" "PASS" "Case created with ID: $($result.Data.id)"
        
        # Test case retrieval
        $getResult = Invoke-APITest -Endpoint "/api/cases/$($result.Data.id)" -Method "GET"
        if ($getResult.Success) {
            Write-TestResult "Case Retrieval" "PASS" "Case retrieved successfully"
        } else {
            Write-TestResult "Case Retrieval" "FAIL" "Failed to retrieve case" $getResult
        }
        
        # Test case update
        $updateData = @{
            status = "closed"
            resolution = "test completed"
        }
        
        $updateResult = Invoke-APITest -Endpoint "/api/cases/$($result.Data.id)" -Method "PATCH" -Body $updateData
        if ($updateResult.Success) {
            Write-TestResult "Case Update" "PASS" "Case updated successfully"
        } else {
            Write-TestResult "Case Update" "FAIL" "Failed to update case" $updateResult
        }
    } else {
        Write-TestResult "Case Creation" "FAIL" "Failed to create case" $result
    }
}

function Test-PerformanceMetrics {
    Write-Host "`n=== Performance Tests ===" -ForegroundColor Cyan
    
    # Test response times
    $endpoints = @(
        @{ Name = "Health Check"; Endpoint = "/api/health"; MaxTime = 100 }
        @{ Name = "Vector Search"; Endpoint = "/api/documents/search"; MaxTime = 500; Method = "POST"; Body = @{query = "test"} }
        @{ Name = "Case List"; Endpoint = "/api/cases"; MaxTime = 200 }
    )
    
    foreach ($test in $endpoints) {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        
        $params = @{
            Endpoint = $test.Endpoint
            Method = if ($test.Method) { $test.Method } else { "GET" }
        }
        if ($test.Body) { $params.Body = $test.Body }
        
        $result = Invoke-APITest @params
        $stopwatch.Stop()
        
        $responseTime = $stopwatch.ElapsedMilliseconds
        if ($result.Success -and $responseTime -le $test.MaxTime) {
            Write-TestResult "Performance: $($test.Name)" "PASS" "${responseTime}ms (max: $($test.MaxTime)ms)"
        } elseif ($result.Success) {
            Write-TestResult "Performance: $($test.Name)" "WARN" "${responseTime}ms exceeds target of $($test.MaxTime)ms"
        } else {
            Write-TestResult "Performance: $($test.Name)" "FAIL" "Request failed"
        }
    }
}

function Test-ErrorHandling {
    Write-Host "`n=== Error Handling Tests ===" -ForegroundColor Cyan
    
    # Test invalid requests
    $invalidTests = @(
        @{
            Name = "Invalid JSON"
            Endpoint = "/api/cases"
            Method = "POST"
            Body = "{ invalid json"
            Headers = @{"Content-Type" = "application/json"}
            ExpectedStatus = 400
        }
        @{
            Name = "Missing Required Fields"
            Endpoint = "/api/case-scoring"
            Method = "POST"
            Body = @{ incomplete = $true }
            ExpectedStatus = 400
        }
        @{
            Name = "Invalid UUID"
            Endpoint = "/api/cases/not-a-uuid"
            Method = "GET"
            ExpectedStatus = 400
        }
        @{
            Name = "Non-existent Resource"
            Endpoint = "/api/cases/00000000-0000-0000-0000-000000000000"
            Method = "GET"
            ExpectedStatus = 404
        }
    )
    
    foreach ($test in $invalidTests) {
        if ($test.Body -is [string]) {
            # For invalid JSON test
            try {
                $params = @{
                    Uri = "$BaseUrl$($test.Endpoint)"
                    Method = $test.Method
                    Headers = $test.Headers
                    Body = $test.Body
                    TimeoutSec = $script:Config.Timeout
                }
                $response = Invoke-WebRequest @params
                $statusCode = $response.StatusCode
            } catch {
                $statusCode = $_.Exception.Response.StatusCode.value__
            }
            
            if ($statusCode -eq $test.ExpectedStatus) {
                Write-TestResult "Error Handling: $($test.Name)" "PASS" "Correct status code: $statusCode"
            } else {
                Write-TestResult "Error Handling: $($test.Name)" "FAIL" "Expected $($test.ExpectedStatus), got $statusCode"
            }
        } else {
            $result = Invoke-APITest -Endpoint $test.Endpoint -Method $test.Method -Body $test.Body -ExpectedStatus $test.ExpectedStatus
            if ($result.StatusCode -eq $test.ExpectedStatus) {
                Write-TestResult "Error Handling: $($test.Name)" "PASS" "Correct error response"
            } else {
                Write-TestResult "Error Handling: $($test.Name)" "FAIL" "Incorrect error handling" $result
            }
        }
    }
}

# Main execution
function Main {
    Write-Host @"
╔═══════════════════════════════════════════════════════╗
║          Legal AI System - API Test Suite              ║
║                   Version 1.0.0                        ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

    if ($HealthCheck) {
        Test-SystemHealth
    } elseif ($TestAll) {
        Test-SystemHealth
        Test-CaseScoring
        Test-VectorSearch
        Test-AIChat
        Test-EvidenceSynthesis
        Test-DatabaseOperations
        Test-PerformanceMetrics
        Test-ErrorHandling
    } else {
        # Run basic tests
        Test-SystemHealth
        Test-CaseScoring
        Test-VectorSearch
    }
    
    # Generate summary
    Write-Host "`n=== Test Summary ===" -ForegroundColor Cyan
    Write-Host "Total Tests: $($script:TestResults.Summary.Total)" -ForegroundColor White
    Write-Host "Passed: $($script:TestResults.Summary.Passed)" -ForegroundColor Green
    Write-Host "Failed: $($script:TestResults.Summary.Failed)" -ForegroundColor Red
    Write-Host "Warnings: $($script:TestResults.Summary.Warnings)" -ForegroundColor Yellow
    
    # Calculate success rate
    if ($script:TestResults.Summary.Total -gt 0) {
        $successRate = [Math]::Round(($script:TestResults.Summary.Passed / $script:TestResults.Summary.Total) * 100, 2)
        $color = if ($successRate -ge 90) { "Green" } elseif ($successRate -ge 70) { "Yellow" } else { "Red" }
        Write-Host "Success Rate: $successRate%" -ForegroundColor $color
    }
    
    # Save results
    $script:TestResults | ConvertTo-Json -Depth 10 | Out-File $OutputFile
    Write-Host "`nResults saved to: $OutputFile" -ForegroundColor Gray
    
    # Create TODO items for failures
    if ($script:TestResults.Summary.Failed -gt 0) {
        $failedTests = $script:TestResults.Tests | Where-Object { $_.Status -eq "FAIL" }
        foreach ($failed in $failedTests) {
            $todoPath = Join-Path (Split-Path $PSScriptRoot) "TODO.md"
            $todoEntry = "- [ ] Fix test failure: $($failed.Name) - $($failed.Message)`n"
            Add-Content -Path $todoPath -Value $todoEntry -ErrorAction SilentlyContinue
        }
        Write-Host "`nTODO items created for failed tests" -ForegroundColor Yellow
    }
}

# Run main
Main
