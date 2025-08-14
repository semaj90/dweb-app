# ================================================================================
# NEO4J LEGAL AI INTEGRATION SETUP (Native Windows, No Docker)
# ================================================================================
# Downloads and configures Neo4j Community Edition for Legal AI Platform
# Integrates with PostgreSQL, pgvector, Qdrant, Drizzle ORM, and Ollama
# ================================================================================

param(
    [switch]$Download,
    [switch]$Configure,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Install
)

$NEO4J_VERSION = "5.23.0"
$NEO4J_URL = "https://dist.neo4j.org/neo4j-community-$NEO4J_VERSION-windows.zip"
$NEO4J_DIR = "neo4j-community-$NEO4J_VERSION"
$NEO4J_HOME = ".\$NEO4J_DIR"

function Download-Neo4j {
    Write-Host "Downloading Neo4j Community Edition v$NEO4J_VERSION..." -ForegroundColor Yellow
    
    if (-not (Test-Path "neo4j-community-$NEO4J_VERSION-windows.zip")) {
        try {
            Invoke-WebRequest -Uri $NEO4J_URL -OutFile "neo4j-community-$NEO4J_VERSION-windows.zip"
            Write-Host "[OK] Neo4j downloaded successfully" -ForegroundColor Green
        } catch {
            Write-Host "[ERR] Failed to download Neo4j: $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "[OK] Neo4j archive already exists" -ForegroundColor Green
    }
    
    # Extract Neo4j
    if (-not (Test-Path $NEO4J_DIR)) {
        try {
            Expand-Archive -Path "neo4j-community-$NEO4J_VERSION-windows.zip" -DestinationPath "." -Force
            Write-Host "[OK] Neo4j extracted to $NEO4J_DIR" -ForegroundColor Green
        } catch {
            Write-Host "[ERR] Failed to extract Neo4j: $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "[OK] Neo4j directory already exists" -ForegroundColor Green
    }
    
    return $true
}

function Configure-Neo4j {
    Write-Host "Configuring Neo4j for Legal AI Platform..." -ForegroundColor Yellow
    
    $configFile = "$NEO4J_HOME\conf\neo4j.conf"
    
    # Create configuration for Legal AI integration
    $config = @"
# Neo4j Legal AI Configuration
# Generated on $(Get-Date)

# Network connector configuration
server.default_listen_address=0.0.0.0
server.default_advertised_address=localhost

# HTTP Connector
server.http.enabled=true
server.http.listen_address=:7474

# BOLT Connector 
server.bolt.enabled=true
server.bolt.listen_address=:7687

# Security configuration
dbms.security.auth_enabled=true
server.db.query_timeout=0

# Memory settings optimized for Legal AI
server.memory.heap.initial_size=1G
server.memory.heap.max_size=2G
server.memory.pagecache.size=1G

# Transaction settings for bulk imports
dbms.transaction.timeout=300s
dbms.transaction.concurrent.maximum=1000

# Logging configuration
server.logs.config=conf/logback.xml
server.logs.debug.level=INFO

# APOC Plugin configuration (for PostgreSQL integration)
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

# Legal AI specific indexes and constraints
dbms.default_database=legalai

# Performance tuning for vector similarity
cypher.runtime=pipelined
cypher.streaming=true
"@

    try {
        $config | Out-File -FilePath $configFile -Encoding UTF8 -Force
        Write-Host "[OK] Neo4j configuration created" -ForegroundColor Green
    } catch {
        Write-Host "[ERR] Failed to create Neo4j configuration: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    
    return $true
}

function Start-Neo4j {
    Write-Host "Starting Neo4j Community Edition..." -ForegroundColor Yellow
    
    if (-not (Test-Path "$NEO4J_HOME\bin\neo4j.bat")) {
        Write-Host "[ERR] Neo4j not found. Run with -Download first." -ForegroundColor Red
        return $false
    }
    
    try {
        Push-Location $NEO4J_HOME
        Start-Process -FilePath "bin\neo4j.bat" -ArgumentList "console" -WindowStyle Minimized
        Pop-Location
        
        Start-Sleep 5
        Write-Host "[OK] Neo4j started successfully" -ForegroundColor Green
        Write-Host "Neo4j Browser: http://localhost:7474" -ForegroundColor Cyan
        Write-Host "Default credentials: neo4j/neo4j (change on first login)" -ForegroundColor Cyan
    } catch {
        Pop-Location
        Write-Host "[ERR] Failed to start Neo4j: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    
    return $true
}

function Stop-Neo4j {
    Write-Host "Stopping Neo4j..." -ForegroundColor Yellow
    
    try {
        Get-Process -Name "*neo4j*" -ErrorAction SilentlyContinue | Stop-Process -Force
        Write-Host "[OK] Neo4j stopped" -ForegroundColor Green
    } catch {
        Write-Host "[ERR] Failed to stop Neo4j: $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Test-Neo4jStatus {
    Write-Host "Checking Neo4j Status..." -ForegroundColor Yellow
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:7474" -Method Get -TimeoutSec 5
        Write-Host "[OK] Neo4j is running and accessible" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[ERR] Neo4j is not responding" -ForegroundColor Red
        return $false
    }
}

function Install-Neo4jIntegration {
    Write-Host "Installing Neo4j Legal AI Integration..." -ForegroundColor Yellow
    
    # Download and configure Neo4j
    if (-not (Download-Neo4j)) { return $false }
    if (-not (Configure-Neo4j)) { return $false }
    
    # Create integration scripts
    $integrationScript = @"
# Neo4j Legal AI Integration Scripts
# These scripts sync data between PostgreSQL, Neo4j, and Qdrant

# 1. Create Legal Knowledge Graph Schema
CREATE CONSTRAINT legal_case_id IF NOT EXISTS FOR (c:LegalCase) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT legal_document_id IF NOT EXISTS FOR (d:LegalDocument) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT legal_entity_id IF NOT EXISTS FOR (e:LegalEntity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT legal_precedent_id IF NOT EXISTS FOR (p:LegalPrecedent) REQUIRE p.id IS UNIQUE;

# 2. Create Legal Relationship Types
// CASE_DOCUMENT: Links cases to their documents
// DOCUMENT_ENTITY: Links documents to mentioned entities
// ENTITY_PRECEDENT: Links entities to legal precedents
// CASE_PRECEDENT: Links cases to applicable precedents
// SIMILAR_CASE: Links similar legal cases

# 3. Vector Similarity Index for Semantic Search
CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
FOR (d:LegalDocument) ON (d.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
"@

    $integrationScript | Out-File -FilePath "neo4j-legal-schema.cypher" -Encoding UTF8
    
    Write-Host "[OK] Neo4j Legal AI integration configured" -ForegroundColor Green
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host "1. Start Neo4j: .\neo4j-integration-setup.ps1 -Start" -ForegroundColor Cyan
    Write-Host "2. Run schema: Execute neo4j-legal-schema.cypher in Neo4j Browser" -ForegroundColor Cyan
    Write-Host "3. Configure PostgreSQL sync in Go services" -ForegroundColor Cyan
    
    return $true
}

# Main execution logic
switch ($true) {
    $Download { Download-Neo4j }
    $Configure { Configure-Neo4j }
    $Start { Start-Neo4j }
    $Stop { Stop-Neo4j }
    $Status { Test-Neo4jStatus }
    $Install { Install-Neo4jIntegration }
    default {
        Write-Host "Neo4j Legal AI Integration Setup" -ForegroundColor Green
        Write-Host "Usage:" -ForegroundColor White
        Write-Host "  .\neo4j-integration-setup.ps1 -Install    # Full installation" -ForegroundColor Cyan
        Write-Host "  .\neo4j-integration-setup.ps1 -Download   # Download Neo4j" -ForegroundColor Cyan
        Write-Host "  .\neo4j-integration-setup.ps1 -Configure # Configure for Legal AI" -ForegroundColor Cyan
        Write-Host "  .\neo4j-integration-setup.ps1 -Start     # Start Neo4j" -ForegroundColor Cyan
        Write-Host "  .\neo4j-integration-setup.ps1 -Stop      # Stop Neo4j" -ForegroundColor Cyan
        Write-Host "  .\neo4j-integration-setup.ps1 -Status    # Check status" -ForegroundColor Cyan
    }
}