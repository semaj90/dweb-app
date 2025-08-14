# Legal AI System - Comprehensive Health Check
# Tests all system components and provides detailed diagnostics

param(
    [switch]$Detailed = $false,
    [switch]$Fix = $false,
    [switch]$Json = $false
)

$ErrorActionPreference = "Continue"

# System configuration
$script:Config = @{
    Services = @{
        PostgreSQL = @{
            Container = "legal_ai_postgres"
            Port = 5432
            HealthCheck = { docker exec legal_ai_postgres pg_isready -U postgres -d prosecutor_db }
            FixCommand = { docker-compose restart postgres }
        }
        Redis = @{
            Container = "legal_ai_redis"
            Port = 6379
            HealthCheck = { docker exec legal_ai_redis redis-cli ping }
            FixCommand = { docker-compose restart redis }
        }
        Qdrant = @{
            Container = "legal_ai_qdrant"
            Port = 6333
            HealthCheck = { (Invoke-RestMethod -Uri "http://localhost:6333/health" -Method GET).status -eq "ok" }
            FixCommand = { docker-compose restart qdrant }
        }
        Ollama = @{
            Container = "legal_ai_ollama"
            Port = 11434
            HealthCheck = { (Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET).models.Count -gt 0 }
            FixCommand = { docker-compose restart ollama }
        }
        Neo4j = @{
            Container = "legal_ai_neo4j"
            Port = 7474
            HealthCheck = { 
                $auth = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("neo4j:legal-ai-2024"))
                $response = Invoke-RestMethod -Uri "http://localhost:7474/db/neo4j/tx" -Method POST -Headers @{Authorization="Basic $auth"} -Body '{"statements":[{"statement":"RETURN 1"}]}' -ContentType "application/json"
                $response.results.Count -gt 0
            }
            FixCommand = { docker-compose restart neo4j }
        }
    }
    
    RequiredModels = @("nomic-embed-text", "gemma:2b", "gemma3-legal")
    RequiredCollections = @("legal_documents", "case_embeddings", "evidence_vectors")
    VectorDimensions = 384
}

# Health check results
$script:HealthStatus = @{
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Overall = "Unknown"
    Services = @{}
    Issues = @()
    Recommendations = @()
}

# Helper functions
function Write-HealthStatus {
    param(
        [string]$Component,
        [string]$Status,
        [string]$Message,
        [hashtable]$Details = @{}
    )
    
    $icon = switch ($Status) {
        "Healthy" { "✓"; $color = "Green" }
        "Warning" { "⚠"; $color = "Yellow" }
        "Critical" { "✗"; $color = "Red" }
        "Info" { "ℹ"; $color = "Cyan" }
        default { "?"; $color = "Gray" }
    }
    
    if (-not $Json) {
        Write-Host "$icon $Component : $Status - $Message" -ForegroundColor $color
        
        if ($Detailed -and $Details.Count -gt 0) {
            foreach ($key in $Details.Keys) {
                Write-Host "    $key : $($Details[$key])" -ForegroundColor Gray
            }
        }
    }
    
    $script:HealthStatus.Services[$Component] = @{
        Status = $Status
        Message = $Message
        Details = $Details
    }
}

function Add-Issue {
    param(
        [string]$Severity,
        [string]$Component,
        [string]$Description,
        [string]$Resolution = ""
    )
    
    $script:HealthStatus.Issues += @{
        Severity = $Severity
        Component = $Component
        Description = $Description
        Resolution = $Resolution
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    }
}

function Add-Recommendation {
    param(
        [string]$Category,
        [string]$Recommendation,
        [string]$Command = ""
    )
    
    $script:HealthStatus.Recommendations += @{
        Category = $Category
        Recommendation = $Recommendation
        Command = $Command
    }
}

# Health check functions
function Test-DockerService {
    param(
        [string]$ServiceName,
        [hashtable]$ServiceConfig
    )
    
    $details = @{}
    
    # Check if container exists
    $container = docker ps -a --filter "name=$($ServiceConfig.Container)" --format "{{.Names}},{{.Status}},{{.State}}" 2>$null
    
    if (-not $container) {
        Write-HealthStatus $ServiceName "Critical" "Container not found" @{Container = $ServiceConfig.Container}
        Add-Issue "Critical" $ServiceName "Container $($ServiceConfig.Container) does not exist" "Run: docker-compose up -d"
        return $false
    }
    
    $containerInfo = $container -split ','
    $details.ContainerStatus = $containerInfo[1]
    
    # Check if running
    if ($containerInfo[2] -ne "running") {
        Write-HealthStatus $ServiceName "Critical" "Container not running" $details
        Add-Issue "Critical" $ServiceName "Container is not running" "Run: docker-compose start $($ServiceConfig.Container)"
        
        if ($Fix) {
            Write-Host "  Attempting to fix..." -ForegroundColor Yellow
            & $ServiceConfig.FixCommand
        }
        return $false
    }
    
    # Check port availability
    $portOpen = Test-NetConnection -ComputerName localhost -Port $ServiceConfig.Port -WarningAction SilentlyContinue -InformationLevel Quiet
    if (-not $portOpen) {
        Write-HealthStatus $ServiceName "Warning" "Port $($ServiceConfig.Port) not accessible" $details
        Add-Issue "Warning" $ServiceName "Port $($ServiceConfig.Port) is not accessible" "Check firewall or port binding"
        return $false
    }
    
    # Run service-specific health check
    try {
        $healthResult = & $ServiceConfig.HealthCheck
        if ($healthResult) {
            # Additional service-specific checks
            switch ($ServiceName) {
                "PostgreSQL" {
                    $dbSize = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT pg_size_pretty(pg_database_size('prosecutor_db'))" 2>$null
                    $details.DatabaseSize = $dbSize.Trim()
                    
                    $tableCount = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'" 2>$null
                    $details.TableCount = $tableCount.Trim()
                }
                "Redis" {
                    $info = docker exec legal_ai_redis redis-cli INFO memory 2>$null
                    if ($info) {
                        $usedMemory = ($info | Select-String "used_memory_human:(.+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }).Trim()
                        $details.MemoryUsage = $usedMemory
                    }
                }
                "Qdrant" {
                    $collections = Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method GET
                    $details.Collections = $collections.result.collections.Count
                    $details.TotalVectors = ($collections.result.collections | Measure-Object -Property vectors_count -Sum).Sum
                }
                "Ollama" {
                    $models = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
                    $details.LoadedModels = $models.models.Count
                    $details.Models = ($models.models.name -join ", ")
                }
            }
            
            Write-HealthStatus $ServiceName "Healthy" "Service operational" $details
            return $true
        } else {
            Write-HealthStatus $ServiceName "Warning" "Health check failed" $details
            Add-Issue "Warning" $ServiceName "Service health check failed" "Check service logs: docker logs $($ServiceConfig.Container)"
            return $false
        }
    } catch {
        Write-HealthStatus $ServiceName "Critical" "Health check error: $_" $details
        Add-Issue "Critical" $ServiceName $_.Exception.Message "Check service configuration"
        return $false
    }
}

function Test-VectorConfiguration {
    if (-not $Json) {
        Write-Host "`n=== Vector Configuration ===" -ForegroundColor Cyan
    }
    
    $vectorHealthy = $true
    
    # Check Qdrant collections
    try {
        $collections = Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method GET
        
        foreach ($requiredCollection in $Config.RequiredCollections) {
            $collection = $collections.result.collections | Where-Object { $_.name -eq $requiredCollection }
            
            if ($collection) {
                if ($collection.config.params.vectors.size -eq $Config.VectorDimensions) {
                    Write-HealthStatus "Collection: $requiredCollection" "Healthy" "Correct dimensions ($($Config.VectorDimensions))" @{
                        Vectors = $collection.vectors_count
                        Segments = $collection.segments_count
                    }
                } else {
                    Write-HealthStatus "Collection: $requiredCollection" "Critical" "Wrong dimensions: $($collection.config.params.vectors.size)" @{
                        Expected = $Config.VectorDimensions
                        Actual = $collection.config.params.vectors.size
                    }
                    Add-Issue "Critical" "Qdrant" "Collection $requiredCollection has wrong vector dimensions" "Recreate collection with correct dimensions"
                    $vectorHealthy = $false
                }
            } else {
                Write-HealthStatus "Collection: $requiredCollection" "Critical" "Collection missing"
                Add-Issue "Critical" "Qdrant" "Required collection $requiredCollection not found" "Create collection via API or restart services"
                $vectorHealthy = $false
            }
        }
    } catch {
        Write-HealthStatus "Vector Configuration" "Critical" "Cannot check collections: $_"
        $vectorHealthy = $false
    }
    
    # Check embedding model
    try {
        $models = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
        $embeddingModel = $models.models | Where-Object { $_.name -like "*nomic-embed*" }
        
        if ($embeddingModel) {
            Write-HealthStatus "Embedding Model" "Healthy" "nomic-embed-text available" @{
                Size = [math]::Round($embeddingModel.size / 1GB, 2).ToString() + " GB"
            }
        } else {
            Write-HealthStatus "Embedding Model" "Critical" "nomic-embed-text not found"
            Add-Issue "Critical" "Ollama" "Embedding model not available" "Run: ollama pull nomic-embed-text"
            $vectorHealthy = $false
        }
    } catch {
        Write-HealthStatus "Embedding Model" "Warning" "Cannot verify model"
    }
    
    return $vectorHealthy
}

function Test-DatabaseSchema {
    if (-not $Json) {
        Write-Host "`n=== Database Schema ===" -ForegroundColor Cyan
    }
    
    try {
        # Check required tables
        $tables = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public'" 2>$null
        $tableList = $tables -split "`n" | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_.Trim() }
        
        $requiredTables = @("cases", "evidence", "documents", "case_scores", "system_health")
        $missingTables = $requiredTables | Where-Object { $_ -notin $tableList }
        
        if ($missingTables.Count -eq 0) {
            Write-HealthStatus "Database Schema" "Healthy" "All required tables exist" @{
                TableCount = $tableList.Count
            }
        } else {
            Write-HealthStatus "Database Schema" "Critical" "Missing tables: $($missingTables -join ', ')"
            Add-Issue "Critical" "PostgreSQL" "Required database tables missing" "Run database migrations: npm run db:migrate"
            
            if ($Fix) {
                Write-Host "  Running migrations..." -ForegroundColor Yellow
                Push-Location $PSScriptRoot
                npm run db:push
                Pop-Location
            }
        }
        
        # Check extensions
        $extensions = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT extname FROM pg_extension" 2>$null
        $extList = $extensions -split "`n" | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_.Trim() }
        
        if ("vector" -in $extList) {
            Write-HealthStatus "pgvector Extension" "Healthy" "Vector extension installed"
        } else {
            Write-HealthStatus "pgvector Extension" "Critical" "Vector extension not installed"
            Add-Issue "Critical" "PostgreSQL" "pgvector extension missing" "Install pgvector extension"
        }
        
        return $missingTables.Count -eq 0
    } catch {
        Write-HealthStatus "Database Schema" "Critical" "Cannot check schema: $_"
        return $false
    }
}

function Test-APIEndpoints {
    if (-not $Json) {
        Write-Host "`n=== API Endpoints ===" -ForegroundColor Cyan
    }
    
    $apiHealthy = $true
    $baseUrl = "http://localhost:5173"
    
    # Check if development server is running
    try {
        $response = Invoke-RestMethod -Uri "$baseUrl/api/health" -Method GET -TimeoutSec 5
        Write-HealthStatus "API Server" "Healthy" "Development server responding" @{
            Status = $response.status
            Version = $response.version
        }
    } catch {
        Write-HealthStatus "API Server" "Critical" "Development server not responding"
        Add-Issue "Critical" "API" "Development server not running" "Run: cd sveltekit-frontend && npm run dev"
        Add-Recommendation "Development" "Start the development server" "cd sveltekit-frontend && npm run dev"
        return $false
    }
    
    # Test critical endpoints
    $endpoints = @(
        @{Name = "Case Scoring"; Path = "/api/case-scoring"; Method = "POST"}
        @{Name = "Document Embedding"; Path = "/api/documents/embed"; Method = "POST"}
        @{Name = "Vector Search"; Path = "/api/documents/search"; Method = "POST"}
        @{Name = "AI Chat"; Path = "/api/ai/chat"; Method = "POST"}
    )
    
    foreach ($endpoint in $endpoints) {
        try {
            # Use OPTIONS request to check endpoint availability
            $response = Invoke-WebRequest -Uri "$baseUrl$($endpoint.Path)" -Method OPTIONS -TimeoutSec 5
            if ($response.StatusCode -eq 200 -or $response.StatusCode -eq 204) {
                Write-HealthStatus "Endpoint: $($endpoint.Name)" "Healthy" "Endpoint available"
            } else {
                Write-HealthStatus "Endpoint: $($endpoint.Name)" "Warning" "Unexpected status: $($response.StatusCode)"
                $apiHealthy = $false
            }
        } catch {
            if ($_.Exception.Response.StatusCode -eq 405) {
                # Method not allowed means endpoint exists
                Write-HealthStatus "Endpoint: $($endpoint.Name)" "Healthy" "Endpoint exists"
            } else {
                Write-HealthStatus "Endpoint: $($endpoint.Name)" "Warning" "Cannot verify endpoint"
                $apiHealthy = $false
            }
        }
    }
    
    return $apiHealthy
}

function Show-SystemSummary {
    if ($Json) {
        $script:HealthStatus | ConvertTo-Json -Depth 5
        return
    }
    
    Write-Host "`n=== System Health Summary ===" -ForegroundColor Cyan
    
    # Count statuses
    $statusCounts = @{
        Healthy = 0
        Warning = 0
        Critical = 0
    }
    
    foreach ($service in $script:HealthStatus.Services.Values) {
        if ($statusCounts.ContainsKey($service.Status)) {
            $statusCounts[$service.Status]++
        }
    }
    
    # Determine overall status
    if ($statusCounts.Critical -gt 0) {
        $script:HealthStatus.Overall = "Critical"
        $overallColor = "Red"
    } elseif ($statusCounts.Warning -gt 0) {
        $script:HealthStatus.Overall = "Warning"
        $overallColor = "Yellow"
    } else {
        $script:HealthStatus.Overall = "Healthy"
        $overallColor = "Green"
    }
    
    Write-Host "Overall Status: $($script:HealthStatus.Overall)" -ForegroundColor $overallColor
    Write-Host "Healthy: $($statusCounts.Healthy)" -ForegroundColor Green
    Write-Host "Warnings: $($statusCounts.Warning)" -ForegroundColor Yellow
    Write-Host "Critical: $($statusCounts.Critical)" -ForegroundColor Red
    
    # Show issues
    if ($script:HealthStatus.Issues.Count -gt 0) {
        Write-Host "`n=== Issues Found ===" -ForegroundColor Yellow
        
        $criticalIssues = $script:HealthStatus.Issues | Where-Object { $_.Severity -eq "Critical" }
        $warningIssues = $script:HealthStatus.Issues | Where-Object { $_.Severity -eq "Warning" }
        
        if ($criticalIssues) {
            Write-Host "`nCritical Issues:" -ForegroundColor Red
            foreach ($issue in $criticalIssues) {
                Write-Host "  • [$($issue.Component)] $($issue.Description)" -ForegroundColor Red
                if ($issue.Resolution) {
                    Write-Host "    → Resolution: $($issue.Resolution)" -ForegroundColor Gray
                }
            }
        }
        
        if ($warningIssues) {
            Write-Host "`nWarnings:" -ForegroundColor Yellow
            foreach ($issue in $warningIssues) {
                Write-Host "  • [$($issue.Component)] $($issue.Description)" -ForegroundColor Yellow
                if ($issue.Resolution) {
                    Write-Host "    → Resolution: $($issue.Resolution)" -ForegroundColor Gray
                }
            }
        }
    }
    
    # Show recommendations
    if ($script:HealthStatus.Recommendations.Count -gt 0) {
        Write-Host "`n=== Recommendations ===" -ForegroundColor Cyan
        foreach ($rec in $script:HealthStatus.Recommendations) {
            Write-Host "  • [$($rec.Category)] $($rec.Recommendation)" -ForegroundColor Cyan
            if ($rec.Command) {
                Write-Host "    → Command: $($rec.Command)" -ForegroundColor Gray
            }
        }
    }
    
    # Quick start guide if system is not healthy
    if ($script:HealthStatus.Overall -ne "Healthy") {
        Write-Host "`n=== Quick Fix Guide ===" -ForegroundColor Yellow
        Write-Host "1. Ensure Docker Desktop is running" -ForegroundColor White
        Write-Host "2. Run: .\install.ps1" -ForegroundColor White
        Write-Host "3. Run: .\health-check.ps1 -Fix" -ForegroundColor White
        Write-Host "4. If issues persist, check logs: docker-compose logs" -ForegroundColor White
    }
    
    # Save detailed report
    $reportFile = Join-Path $PSScriptRoot "health-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    $script:HealthStatus | ConvertTo-Json -Depth 5 | Out-File $reportFile
    Write-Host "`nDetailed report saved to: $reportFile" -ForegroundColor Gray
}

# Main execution
function Main {
    if (-not $Json) {
        Write-Host @"
╔═══════════════════════════════════════════════════════╗
║       Legal AI System - Health Check v1.0.0            ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan
    }
    
    # Test all services
    if (-not $Json) {
        Write-Host "`n=== Docker Services ===" -ForegroundColor Cyan
    }
    
    foreach ($service in $Config.Services.Keys) {
        Test-DockerService -ServiceName $service -ServiceConfig $Config.Services[$service]
    }
    
    # Test configurations
    Test-VectorConfiguration
    Test-DatabaseSchema
    Test-APIEndpoints
    
    # Performance recommendations
    if ($Detailed) {
        # Check system resources
        $memory = Get-CimInstance Win32_OperatingSystem
        $availableGB = [math]::Round($memory.FreePhysicalMemory / 1MB / 1024, 2)
        
        if ($availableGB -lt 4) {
            Add-Recommendation "Performance" "Low memory available ($availableGB GB). Consider closing other applications" ""
        }
        
        # Check Docker resources
        $dockerInfo = docker system df --format json | ConvertFrom-Json
        if ($dockerInfo) {
            $spaceGB = [math]::Round($dockerInfo.LayersSize / 1GB, 2)
            if ($spaceGB -gt 20) {
                Add-Recommendation "Storage" "Docker using $spaceGB GB. Consider pruning: docker system prune -a" "docker system prune -a"
            }
        }
    }
    
    # Show summary
    Show-SystemSummary
}

# Run health check
Main
