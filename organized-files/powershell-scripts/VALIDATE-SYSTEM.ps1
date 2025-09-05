# Legal AI System - Comprehensive Production Validation
# Enterprise-grade system validation with performance metrics

param(
    [switch]$Production = $false,
    [switch]$Benchmark = $false,
    [switch]$ExportReport = $false,
    [string]$ReportPath = ".\validation-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').html"
)

$ErrorActionPreference = "Continue"
$script:StartTime = Get-Date

# Validation configuration
$script:Config = @{
    RequiredServices = @{
        PostgreSQL = @{ Port = 5432; Container = "legal_ai_postgres"; Critical = $true }
        Redis = @{ Port = 6379; Container = "legal_ai_redis"; Critical = $true }
        Qdrant = @{ Port = 6333; Container = "legal_ai_qdrant"; Critical = $true }
        Ollama = @{ Port = 11434; Container = "legal_ai_ollama"; Critical = $true }
        Neo4j = @{ Port = 7474; Container = "legal_ai_neo4j"; Critical = $false }
    }
    
    PerformanceThresholds = @{
        DockerMemory = 8GB
        FreeSpace = 20GB
        CPUUsage = 80
        DatabaseConnections = 100
        VectorSearchTime = 500  # milliseconds
        EmbeddingTime = 1000    # milliseconds
        APIResponseTime = 200   # milliseconds
    }
    
    RequiredModels = @("nomic-embed-text", "gemma3-legal", "gemma:2b")
    RequiredCollections = @("legal_documents", "case_embeddings", "evidence_vectors")
    VectorDimensions = 384
}

# Validation results
$script:Results = @{
    Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Environment = @{}
    Services = @{}
    Performance = @{}
    Security = @{}
    DataIntegrity = @{}
    Recommendations = @()
    Errors = @()
    Score = 0
}

# Helper functions
function Write-ValidationStep {
    param(
        [string]$Category,
        [string]$Test,
        [string]$Status,
        [string]$Details = "",
        [hashtable]$Metrics = @{}
    )
    
    $icon = switch ($Status) {
        "PASS" { "✓"; $color = "Green" }
        "FAIL" { "✗"; $color = "Red" }
        "WARN" { "⚠"; $color = "Yellow" }
        "INFO" { "ℹ"; $color = "Cyan" }
        default { "?"; $color = "Gray" }
    }
    
    $output = "$icon [$Category] $Test"
    if ($Details) { $output += " - $Details" }
    Write-Host $output -ForegroundColor $color
    
    if ($Metrics.Count -gt 0 -and $Production) {
        foreach ($metric in $Metrics.GetEnumerator()) {
            Write-Host "    → $($metric.Key): $($metric.Value)" -ForegroundColor Gray
        }
    }
    
    # Store result
    if (-not $script:Results.$Category) {
        $script:Results.$Category = @{}
    }
    
    $script:Results.$Category[$Test] = @{
        Status = $Status
        Details = $Details
        Metrics = $Metrics
        Timestamp = Get-Date -Format "HH:mm:ss.fff"
    }
}

function Test-SystemEnvironment {
    Write-Host "`n=== System Environment Validation ===" -ForegroundColor Cyan
    
    # OS Information
    $os = Get-CimInstance Win32_OperatingSystem
    $script:Results.Environment.OS = @{
        Name = $os.Caption
        Version = $os.Version
        Architecture = $os.OSArchitecture
        TotalMemory = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
        FreeMemory = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    }
    
    $memoryUsage = [math]::Round((($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize) * 100, 2)
    
    if ($script:Results.Environment.OS.FreeMemory -lt 4096) {
        Write-ValidationStep "Environment" "Memory" "WARN" "Low memory: $($script:Results.Environment.OS.FreeMemory) GB free" @{
            "Total Memory" = "$($script:Results.Environment.OS.TotalMemory) GB"
            "Memory Usage" = "$memoryUsage%"
        }
        $script:Results.Recommendations += "Consider freeing up memory or increasing system RAM"
    } else {
        Write-ValidationStep "Environment" "Memory" "PASS" "Sufficient memory available" @{
            "Free Memory" = "$($script:Results.Environment.OS.FreeMemory) GB"
            "Memory Usage" = "$memoryUsage%"
        }
    }
    
    # Disk Space
    $drive = Get-PSDrive -Name (Get-Location).Drive.Name
    $freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
    
    if ($freeSpaceGB -lt $Config.PerformanceThresholds.FreeSpace) {
        Write-ValidationStep "Environment" "Disk Space" "WARN" "Low disk space" @{
            "Free Space" = "$freeSpaceGB GB"
            "Required" = "$($Config.PerformanceThresholds.FreeSpace) GB"
        }
    } else {
        Write-ValidationStep "Environment" "Disk Space" "PASS" "Adequate disk space" @{
            "Free Space" = "$freeSpaceGB GB"
        }
    }
    
    # CPU Information
    $cpu = Get-CimInstance Win32_Processor
    $cpuUsage = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
    
    Write-ValidationStep "Environment" "CPU" "INFO" $cpu[0].Name @{
        "Cores" = $cpu[0].NumberOfCores
        "Logical Processors" = $cpu[0].NumberOfLogicalProcessors
        "Current Usage" = "$([math]::Round($cpuUsage, 2))%"
    }
    
    # Docker Information
    try {
        $dockerVersion = docker version --format '{{.Server.Version}}'
        $dockerInfo = docker info --format '{{ json . }}' | ConvertFrom-Json
        
        Write-ValidationStep "Environment" "Docker" "PASS" "Docker Desktop running" @{
            "Version" = $dockerVersion
            "Containers" = $dockerInfo.Containers
            "Images" = $dockerInfo.Images
            "Storage Driver" = $dockerInfo.Driver
        }
        
        # Check Docker resources
        if ($dockerInfo.MemTotal -lt 8GB) {
            Write-ValidationStep "Environment" "Docker Memory" "WARN" "Docker memory limit low" @{
                "Allocated" = "$([math]::Round($dockerInfo.MemTotal / 1GB, 2)) GB"
                "Recommended" = "8+ GB"
            }
            $script:Results.Recommendations += "Increase Docker Desktop memory allocation to 8GB+"
        }
    } catch {
        Write-ValidationStep "Environment" "Docker" "FAIL" "Docker not available" @{ "Error" = $_.Exception.Message }
        $script:Results.Errors += "Docker Desktop not running or not installed"
    }
    
    # PowerShell Version
    Write-ValidationStep "Environment" "PowerShell" "PASS" "PowerShell $($PSVersionTable.PSVersion)" @{
        "Edition" = $PSVersionTable.PSEdition
        "OS" = $PSVersionTable.OS
    }
}

function Test-ServiceHealth {
    Write-Host "`n=== Service Health Validation ===" -ForegroundColor Cyan
    
    foreach ($service in $Config.RequiredServices.GetEnumerator()) {
        $serviceName = $service.Key
        $serviceConfig = $service.Value
        
        # Port availability
        $portOpen = Test-NetConnection -ComputerName localhost -Port $serviceConfig.Port -WarningAction SilentlyContinue -InformationLevel Quiet
        
        if (-not $portOpen) {
            Write-ValidationStep "Services" "$serviceName Port" "FAIL" "Port $($serviceConfig.Port) not accessible"
            if ($serviceConfig.Critical) {
                $script:Results.Errors += "$serviceName service not accessible on port $($serviceConfig.Port)"
            }
            continue
        }
        
        # Container health
        try {
            $containerStats = docker stats $serviceConfig.Container --no-stream --format "{{ json . }}" | ConvertFrom-Json
            $health = docker inspect $serviceConfig.Container --format='{{.State.Health.Status}}' 2>$null
            
            $metrics = @{
                "Memory Usage" = $containerStats.MemUsage
                "CPU Usage" = $containerStats.CPUPerc
                "Network I/O" = $containerStats.NetIO
                "Block I/O" = $containerStats.BlockIO
            }
            
            if ($health -eq "healthy" -or $health -eq $null) {
                Write-ValidationStep "Services" $serviceName "PASS" "Service healthy" $metrics
            } else {
                Write-ValidationStep "Services" $serviceName "WARN" "Health: $health" $metrics
            }
            
            # Service-specific tests
            switch ($serviceName) {
                "PostgreSQL" {
                    Test-PostgreSQLHealth
                }
                "Redis" {
                    Test-RedisHealth
                }
                "Qdrant" {
                    Test-QdrantHealth
                }
                "Ollama" {
                    Test-OllamaHealth
                }
                "Neo4j" {
                    Test-Neo4jHealth
                }
            }
        } catch {
            Write-ValidationStep "Services" $serviceName "FAIL" "Container not found or not running"
            if ($serviceConfig.Critical) {
                $script:Results.Errors += "$serviceName container not running"
            }
        }
    }
}

function Test-PostgreSQLHealth {
    try {
        # Database connection test
        $dbTest = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -c "SELECT version();" 2>&1
        if ($LASTEXITCODE -eq 0) {
            # Get detailed stats
            $tableCount = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>&1
            $dbSize = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT pg_size_pretty(pg_database_size('prosecutor_db'));" 2>&1
            $connections = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT COUNT(*) FROM pg_stat_activity;" 2>&1
            
            Write-ValidationStep "Services" "PostgreSQL Database" "PASS" "Database operational" @{
                "Tables" = $tableCount.Trim()
                "Size" = $dbSize.Trim()
                "Connections" = $connections.Trim()
            }
            
            # Check pgvector extension
            $vectorCheck = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';" 2>&1
            if ($vectorCheck -and $vectorCheck.Trim()) {
                Write-ValidationStep "Services" "pgvector Extension" "PASS" "Version: $($vectorCheck.Trim())"
            } else {
                Write-ValidationStep "Services" "pgvector Extension" "FAIL" "Extension not installed"
                $script:Results.Errors += "pgvector extension missing - vector search will fail"
            }
        } else {
            Write-ValidationStep "Services" "PostgreSQL Database" "FAIL" "Cannot connect to database"
        }
    } catch {
        Write-ValidationStep "Services" "PostgreSQL Database" "FAIL" "Database test failed: $_"
    }
}

function Test-RedisHealth {
    try {
        # Redis info
        $redisInfo = docker exec legal_ai_redis redis-cli INFO server 2>&1
        if ($LASTEXITCODE -eq 0) {
            $redisMemory = docker exec legal_ai_redis redis-cli INFO memory 2>&1
            
            # Parse memory usage
            $usedMemory = ($redisMemory | Select-String "used_memory_human:(.+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }).Trim()
            
            Write-ValidationStep "Services" "Redis Cache" "PASS" "Redis operational" @{
                "Memory Usage" = $usedMemory
                "Version" = (($redisInfo | Select-String "redis_version:(.+)").Matches[0].Groups[1].Value).Trim()
            }
            
            # Test Redis modules
            $modules = docker exec legal_ai_redis redis-cli MODULE LIST 2>&1
            if ($modules -match "search" -and $modules -match "ReJSON") {
                Write-ValidationStep "Services" "Redis Modules" "PASS" "RedisSearch and RedisJSON loaded"
            } else {
                Write-ValidationStep "Services" "Redis Modules" "WARN" "Some modules may be missing"
            }
        }
    } catch {
        Write-ValidationStep "Services" "Redis Cache" "FAIL" "Redis test failed: $_"
    }
}

function Test-QdrantHealth {
    try {
        # Qdrant health
        $qdrantHealth = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method GET
        
        if ($qdrantHealth.status -eq "ok") {
            # Get collections
            $collections = Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method GET
            
            $totalVectors = 0
            $collectionDetails = @{}
            
            foreach ($col in $collections.result.collections) {
                $totalVectors += $col.vectors_count
                $collectionDetails[$col.name] = @{
                    Vectors = $col.vectors_count
                    Dimension = $col.config.params.vectors.size
                    Status = $col.status
                }
                
                # Validate dimensions
                if ($col.config.params.vectors.size -ne $Config.VectorDimensions) {
                    Write-ValidationStep "Services" "Qdrant Collection: $($col.name)" "FAIL" `
                        "Wrong dimensions: $($col.config.params.vectors.size) (expected: $($Config.VectorDimensions))"
                    $script:Results.Errors += "Vector dimension mismatch in collection: $($col.name)"
                } else {
                    Write-ValidationStep "Services" "Qdrant Collection: $($col.name)" "PASS" `
                        "$($col.vectors_count) vectors" @{
                            "Segments" = $col.segments_count
                            "Status" = $col.status
                        }
                }
            }
            
            Write-ValidationStep "Services" "Qdrant Vector DB" "PASS" "Qdrant operational" @{
                "Version" = $qdrantHealth.version
                "Collections" = $collections.result.collections.Count
                "Total Vectors" = $totalVectors
            }
        }
    } catch {
        Write-ValidationStep "Services" "Qdrant Vector DB" "FAIL" "Qdrant test failed: $_"
        $script:Results.Errors += "Qdrant not accessible - vector search unavailable"
    }
}

function Test-OllamaHealth {
    try {
        # Ollama models
        $models = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
        
        $modelList = @()
        $totalSize = 0
        
        foreach ($model in $models.models) {
            $modelList += $model.name
            $totalSize += $model.size
            
            if ($model.name -in $Config.RequiredModels) {
                Write-ValidationStep "Services" "Model: $($model.name)" "PASS" `
                    "Size: $([math]::Round($model.size / 1GB, 2)) GB"
            }
        }
        
        # Check required models
        $missingModels = $Config.RequiredModels | Where-Object { $_ -notin $modelList }
        if ($missingModels) {
            foreach ($missing in $missingModels) {
                Write-ValidationStep "Services" "Model: $missing" "FAIL" "Model not found"
                $script:Results.Errors += "Required model missing: $missing"
            }
        }
        
        Write-ValidationStep "Services" "Ollama AI Service" "PASS" "Ollama operational" @{
            "Models Loaded" = $models.models.Count
            "Total Size" = "$([math]::Round($totalSize / 1GB, 2)) GB"
        }
    } catch {
        Write-ValidationStep "Services" "Ollama AI Service" "FAIL" "Ollama test failed: $_"
        $script:Results.Errors += "Ollama not accessible - AI features unavailable"
    }
}

function Test-Neo4jHealth {
    try {
        $auth = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("neo4j:legal-ai-2024"))
        $response = Invoke-RestMethod -Uri "http://localhost:7474/db/neo4j/tx" -Method POST `
            -Headers @{Authorization="Basic $auth"} `
            -Body '{"statements":[{"statement":"MATCH (n) RETURN COUNT(n) as count"}]}' `
            -ContentType "application/json"
        
        if ($response.results) {
            $nodeCount = $response.results[0].data[0].row[0]
            Write-ValidationStep "Services" "Neo4j Graph DB" "PASS" "Neo4j operational" @{
                "Nodes" = $nodeCount
            }
        }
    } catch {
        Write-ValidationStep "Services" "Neo4j Graph DB" "WARN" "Neo4j not accessible (non-critical)"
    }
}

function Test-Performance {
    if (-not $Benchmark) {
        Write-Host "`n=== Performance Tests ===" -ForegroundColor Cyan
        Write-Host "  Skipping (use -Benchmark flag to enable)" -ForegroundColor Gray
        return
    }
    
    Write-Host "`n=== Performance Benchmarks ===" -ForegroundColor Cyan
    
    # Test embedding generation
    try {
        Write-Host "  Testing embedding generation..." -ForegroundColor Gray
        $testText = "This is a test legal document for performance benchmarking of the embedding service."
        
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/embeddings" -Method POST `
            -Body (@{model = "nomic-embed-text"; prompt = $testText} | ConvertTo-Json) `
            -ContentType "application/json"
        $stopwatch.Stop()
        
        if ($response.embedding -and $response.embedding.Count -eq $Config.VectorDimensions) {
            $time = $stopwatch.ElapsedMilliseconds
            $status = if ($time -le $Config.PerformanceThresholds.EmbeddingTime) { "PASS" } else { "WARN" }
            
            Write-ValidationStep "Performance" "Embedding Generation" $status `
                "$time ms" @{
                    "Threshold" = "$($Config.PerformanceThresholds.EmbeddingTime) ms"
                    "Vector Size" = $response.embedding.Count
                }
        } else {
            Write-ValidationStep "Performance" "Embedding Generation" "FAIL" "Invalid response"
        }
    } catch {
        Write-ValidationStep "Performance" "Embedding Generation" "FAIL" "Test failed: $_"
    }
    
    # Test vector search
    try {
        Write-Host "  Testing vector search..." -ForegroundColor Gray
        
        # First, check if we have any vectors
        $collections = Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method GET
        $testCollection = $collections.result.collections | Where-Object { $_.vectors_count -gt 0 } | Select-Object -First 1
        
        if ($testCollection) {
            # Generate random vector for search
            $testVector = 1..$Config.VectorDimensions | ForEach-Object { Get-Random -Minimum -1.0 -Maximum 1.0 }
            
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            $searchBody = @{
                vector = $testVector
                limit = 10
                with_payload = $true
            } | ConvertTo-Json -Depth 10
            
            $searchResult = Invoke-RestMethod -Uri "http://localhost:6333/collections/$($testCollection.name)/points/search" `
                -Method POST -Body $searchBody -ContentType "application/json"
            $stopwatch.Stop()
            
            $time = $stopwatch.ElapsedMilliseconds
            $status = if ($time -le $Config.PerformanceThresholds.VectorSearchTime) { "PASS" } else { "WARN" }
            
            Write-ValidationStep "Performance" "Vector Search" $status `
                "$time ms" @{
                    "Threshold" = "$($Config.PerformanceThresholds.VectorSearchTime) ms"
                    "Collection" = $testCollection.name
                    "Vectors Searched" = $testCollection.vectors_count
                }
        } else {
            Write-ValidationStep "Performance" "Vector Search" "INFO" "No vectors to search"
        }
    } catch {
        Write-ValidationStep "Performance" "Vector Search" "FAIL" "Test failed: $_"
    }
    
    # Test API response time
    try {
        Write-Host "  Testing API response time..." -ForegroundColor Gray
        
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $health = Invoke-RestMethod -Uri "http://localhost:5173/api/health" -Method GET
        $stopwatch.Stop()
        
        $time = $stopwatch.ElapsedMilliseconds
        $status = if ($time -le $Config.PerformanceThresholds.APIResponseTime) { "PASS" } else { "WARN" }
        
        Write-ValidationStep "Performance" "API Response" $status `
            "$time ms" @{
                "Threshold" = "$($Config.PerformanceThresholds.APIResponseTime) ms"
                "Endpoint" = "/api/health"
            }
    } catch {
        Write-ValidationStep "Performance" "API Response" "INFO" "API server not running"
    }
    
    # Database query performance
    try {
        Write-Host "  Testing database performance..." -ForegroundColor Gray
        
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $result = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -c "SELECT COUNT(*) FROM cases;" 2>&1
        $stopwatch.Stop()
        
        if ($LASTEXITCODE -eq 0) {
            $time = $stopwatch.ElapsedMilliseconds
            Write-ValidationStep "Performance" "Database Query" "PASS" "$time ms"
        }
    } catch {
        Write-ValidationStep "Performance" "Database Query" "FAIL" "Test failed"
    }
}

function Test-Security {
    Write-Host "`n=== Security Validation ===" -ForegroundColor Cyan
    
    # Check for .env files
    $envFiles = @(".env", "sveltekit-frontend\.env")
    $exposedSecrets = $false
    
    foreach ($envFile in $envFiles) {
        if (Test-Path $envFile) {
            $content = Get-Content $envFile -Raw
            
            # Check for default passwords
            if ($content -match "postgres:postgres" -or $content -match "legal-ai-2024") {
                Write-ValidationStep "Security" "Default Passwords" "WARN" "Using default passwords in $envFile"
                $script:Results.Recommendations += "Change default passwords before production deployment"
                $exposedSecrets = $true
            }
            
            # Check for placeholder secrets
            if ($content -match "your-.*-here" -or $content -match "change-in-production") {
                Write-ValidationStep "Security" "Placeholder Secrets" "WARN" "Found placeholder values in $envFile"
                $script:Results.Recommendations += "Update placeholder values in environment files"
            }
        }
    }
    
    if (-not $exposedSecrets) {
        Write-ValidationStep "Security" "Environment Files" "PASS" "No obvious security issues found"
    }
    
    # Check Docker socket exposure
    $dockerCompose = Get-Content "docker-compose.yml" -Raw
    if ($dockerCompose -match "/var/run/docker.sock") {
        Write-ValidationStep "Security" "Docker Socket" "WARN" "Docker socket mounted in container"
        $script:Results.Recommendations += "Review Docker socket mounting for security implications"
    }
    
    # Check network exposure
    $exposedPorts = @()
    foreach ($port in $Config.RequiredServices.Values.Port) {
        $binding = netstat -an | Select-String ":$port\s+.*LISTENING" | Select-String "0.0.0.0"
        if ($binding) {
            $exposedPorts += $port
        }
    }
    
    if ($exposedPorts.Count -gt 0) {
        Write-ValidationStep "Security" "Network Exposure" "INFO" "Ports exposed on all interfaces: $($exposedPorts -join ', ')"
        if ($Production) {
            $script:Results.Recommendations += "Consider binding services to localhost only for production"
        }
    } else {
        Write-ValidationStep "Security" "Network Exposure" "PASS" "Services bound to localhost"
    }
}

function Test-DataIntegrity {
    Write-Host "`n=== Data Integrity Validation ===" -ForegroundColor Cyan
    
    # Check database schema
    try {
        $requiredTables = @("cases", "evidence", "documents", "case_scores", "system_health")
        $tables = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public';" 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            $tableList = $tables -split "`n" | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }
            $missingTables = $requiredTables | Where-Object { $_ -notin $tableList }
            
            if ($missingTables.Count -eq 0) {
                Write-ValidationStep "DataIntegrity" "Database Schema" "PASS" "All required tables present"
            } else {
                Write-ValidationStep "DataIntegrity" "Database Schema" "FAIL" "Missing tables: $($missingTables -join ', ')"
                $script:Results.Errors += "Database schema incomplete - run migrations"
            }
            
            # Check for orphaned records
            $orphanCheck = docker exec legal_ai_postgres psql -U postgres -d prosecutor_db -t -c "SELECT COUNT(*) FROM evidence WHERE case_id NOT IN (SELECT id FROM cases);" 2>&1
            if ($orphanCheck -and [int]$orphanCheck.Trim() -gt 0) {
                Write-ValidationStep "DataIntegrity" "Referential Integrity" "WARN" "$($orphanCheck.Trim()) orphaned evidence records"
                $script:Results.Recommendations += "Clean up orphaned records in evidence table"
            }
        }
    } catch {
        Write-ValidationStep "DataIntegrity" "Database Schema" "FAIL" "Cannot validate schema"
    }
    
    # Check vector collection integrity
    try {
        $collections = Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method GET
        
        foreach ($required in $Config.RequiredCollections) {
            $collection = $collections.result.collections | Where-Object { $_.name -eq $required }
            if (-not $collection) {
                Write-ValidationStep "DataIntegrity" "Vector Collection: $required" "FAIL" "Collection missing"
                $script:Results.Errors += "Required vector collection missing: $required"
            } elseif ($collection.status -ne "green") {
                Write-ValidationStep "DataIntegrity" "Vector Collection: $required" "WARN" "Status: $($collection.status)"
            } else {
                Write-ValidationStep "DataIntegrity" "Vector Collection: $required" "PASS" "Collection healthy"
            }
        }
    } catch {
        Write-ValidationStep "DataIntegrity" "Vector Collections" "FAIL" "Cannot validate collections"
    }
}

function Calculate-ValidationScore {
    $totalTests = 0
    $passedTests = 0
    $criticalFailures = 0
    
    foreach ($category in $script:Results.Keys) {
        if ($category -in @("Services", "Environment", "Performance", "Security", "DataIntegrity")) {
            foreach ($test in $script:Results.$category.Values) {
                $totalTests++
                if ($test.Status -eq "PASS") {
                    $passedTests++
                } elseif ($test.Status -eq "FAIL") {
                    if ($category -eq "Services") {
                        $criticalFailures++
                    }
                }
            }
        }
    }
    
    # Calculate base score
    $baseScore = if ($totalTests -gt 0) { ($passedTests / $totalTests) * 100 } else { 0 }
    
    # Apply penalties
    $penalty = $criticalFailures * 20
    $finalScore = [math]::Max(0, $baseScore - $penalty)
    
    $script:Results.Score = [math]::Round($finalScore, 2)
    
    return @{
        Score = $script:Results.Score
        Total = $totalTests
        Passed = $passedTests
        Failed = $totalTests - $passedTests
        Critical = $criticalFailures
    }
}

function Export-ValidationReport {
    if (-not $ExportReport) { return }
    
    $html = @"
<!DOCTYPE html>
<html>
<head>
    <title>Legal AI System - Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        .score { font-size: 48px; font-weight: bold; text-align: center; padding: 20px; }
        .score.pass { color: #4CAF50; }
        .score.warn { color: #FF9800; }
        .score.fail { color: #F44336; }
        .summary { display: flex; justify-content: space-around; margin: 20px 0; }
        .summary-item { text-align: center; }
        .summary-value { font-size: 24px; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f0f0f0; }
        .pass { color: #4CAF50; }
        .fail { color: #F44336; }
        .warn { color: #FF9800; }
        .info { color: #2196F3; }
        .metric { font-size: 0.9em; color: #666; }
        .recommendations { background: #FFF3E0; padding: 15px; border-radius: 4px; margin: 20px 0; }
        .errors { background: #FFEBEE; padding: 15px; border-radius: 4px; margin: 20px 0; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Legal AI System - Validation Report</h1>
        <p>Generated: $($script:Results.Timestamp)</p>
        
        <div class="score $(if ($script:Results.Score -ge 80) { 'pass' } elseif ($script:Results.Score -ge 60) { 'warn' } else { 'fail' })">
            $($script:Results.Score)%
        </div>
        
        <div class="summary">
            <div class="summary-item">
                <div class="summary-value">$($scoring.Total)</div>
                <div>Total Tests</div>
            </div>
            <div class="summary-item">
                <div class="summary-value pass">$($scoring.Passed)</div>
                <div>Passed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value fail">$($scoring.Failed)</div>
                <div>Failed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value fail">$($scoring.Critical)</div>
                <div>Critical</div>
            </div>
        </div>
"@
    
    # Add test results by category
    foreach ($category in @("Environment", "Services", "Performance", "Security", "DataIntegrity")) {
        if ($script:Results.$category) {
            $html += "<h2>$category</h2><table><tr><th>Test</th><th>Status</th><th>Details</th><th>Metrics</th></tr>"
            
            foreach ($test in $script:Results.$category.GetEnumerator()) {
                $status = $test.Value.Status.ToLower()
                $metricsHtml = ""
                
                if ($test.Value.Metrics) {
                    $metricsHtml = ($test.Value.Metrics.GetEnumerator() | ForEach-Object {
                        "<span class='metric'>$($_.Key): $($_.Value)</span>"
                    }) -join "<br/>"
                }
                
                $html += "<tr><td>$($test.Key)</td><td class='$status'>$($test.Value.Status)</td><td>$($test.Value.Details)</td><td>$metricsHtml</td></tr>"
            }
            
            $html += "</table>"
        }
    }
    
    # Add recommendations
    if ($script:Results.Recommendations.Count -gt 0) {
        $html += "<div class='recommendations'><h3>Recommendations</h3><ul>"
        foreach ($rec in $script:Results.Recommendations) {
            $html += "<li>$rec</li>"
        }
        $html += "</ul></div>"
    }
    
    # Add errors
    if ($script:Results.Errors.Count -gt 0) {
        $html += "<div class='errors'><h3>Critical Errors</h3><ul>"
        foreach ($err in $script:Results.Errors) {
            $html += "<li>$err</li>"
        }
        $html += "</ul></div>"
    }
    
    $html += "</div></body></html>"
    
    $html | Out-File $ReportPath -Encoding UTF8
    Write-Host "`nValidation report exported to: $ReportPath" -ForegroundColor Green
}

# Main execution
function Main {
    Write-Host @"
╔═══════════════════════════════════════════════════════╗
║   Legal AI System - Comprehensive Validation v1.0      ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

    if ($Production) {
        Write-Host "Running in PRODUCTION mode - all tests enabled" -ForegroundColor Yellow
    }
    
    # Run validation tests
    Test-SystemEnvironment
    Test-ServiceHealth
    Test-Performance
    Test-Security
    Test-DataIntegrity
    
    # Calculate score
    $scoring = Calculate-ValidationScore
    
    # Display summary
    Write-Host "`n=== Validation Summary ===" -ForegroundColor Cyan
    Write-Host "Overall Score: $($scoring.Score)%" -ForegroundColor $(
        if ($scoring.Score -ge 80) { "Green" }
        elseif ($scoring.Score -ge 60) { "Yellow" }
        else { "Red" }
    )
    
    Write-Host "`nTests Summary:" -ForegroundColor White
    Write-Host "  Total: $($scoring.Total)" -ForegroundColor White
    Write-Host "  Passed: $($scoring.Passed)" -ForegroundColor Green
    Write-Host "  Failed: $($scoring.Failed)" -ForegroundColor Red
    Write-Host "  Critical Failures: $($scoring.Critical)" -ForegroundColor Red
    
    # Key metrics
    Write-Host "`nKey Metrics:" -ForegroundColor Cyan
    Write-Host "  Execution Time: $([math]::Round((Get-Date - $script:StartTime).TotalSeconds, 2))s" -ForegroundColor Gray
    
    # Show errors if any
    if ($script:Results.Errors.Count -gt 0) {
        Write-Host "`n⚠️  Critical Issues:" -ForegroundColor Red
        foreach ($error in $script:Results.Errors) {
            Write-Host "  • $error" -ForegroundColor Red
        }
    }
    
    # Show recommendations
    if ($script:Results.Recommendations.Count -gt 0) {
        Write-Host "`n💡 Recommendations:" -ForegroundColor Yellow
        foreach ($rec in $script:Results.Recommendations | Select-Object -Unique) {
            Write-Host "  • $rec" -ForegroundColor Yellow
        }
    }
    
    # Production readiness
    if ($Production) {
        Write-Host "`n🏭 Production Readiness:" -ForegroundColor Cyan
        if ($scoring.Score -ge 90 -and $scoring.Critical -eq 0) {
            Write-Host "  ✅ System is PRODUCTION READY" -ForegroundColor Green
        } elseif ($scoring.Score -ge 70) {
            Write-Host "  ⚠️  System needs minor improvements for production" -ForegroundColor Yellow
        } else {
            Write-Host "  ❌ System is NOT ready for production" -ForegroundColor Red
        }
    }
    
    # Export report
    Export-ValidationReport
    
    # Exit code based on score
    if ($scoring.Critical -gt 0) {
        exit 2  # Critical failures
    } elseif ($scoring.Score -lt 60) {
        exit 1  # Too many failures
    } else {
        exit 0  # Success
    }
}

# Run validation
Main
