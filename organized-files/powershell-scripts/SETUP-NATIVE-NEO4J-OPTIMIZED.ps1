# Neo4j Native Windows Setup - Low Memory / Cache-Friendly Configuration
# Optimized for legal AI workloads with aggressive caching

param(
    [string]$Neo4jVersion = "5.23.0",
    [string]$InstallPath = "C:\neo4j",
    [int]$HeapSize = 512,
    [int]$PageCache = 512,
    [switch]$InstallService = $false
)

Write-Host "🔗 Neo4j Native Windows Setup - Cache-Optimized" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "⚠️ This script requires Administrator privileges for service installation" -ForegroundColor Yellow
    Write-Host "Re-run as Administrator or use -InstallService:$false" -ForegroundColor Yellow
}

# Create installation directory
if (-not (Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath -Force
    Write-Host "✅ Created directory: $InstallPath" -ForegroundColor Green
}

# Download Neo4j Community Edition
$downloadUrl = "https://dist.neo4j.org/neo4j-community-$Neo4jVersion-windows.zip"
$zipPath = "$InstallPath\neo4j-community-$Neo4jVersion-windows.zip"
$extractPath = "$InstallPath\neo4j-community-$Neo4jVersion"

if (-not (Test-Path $extractPath)) {
    Write-Host "📥 Downloading Neo4j $Neo4jVersion..." -ForegroundColor Blue
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing
        Write-Host "✅ Downloaded: $zipPath" -ForegroundColor Green
        
        # Extract archive
        Write-Host "📦 Extracting Neo4j..." -ForegroundColor Blue
        Expand-Archive -Path $zipPath -DestinationPath $InstallPath -Force
        Remove-Item $zipPath
        Write-Host "✅ Extracted to: $extractPath" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Download failed: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ Neo4j already exists: $extractPath" -ForegroundColor Green
}

# Create optimized configuration
$configPath = "$extractPath\conf\neo4j.conf"
Write-Host "⚙️ Creating cache-optimized configuration..." -ForegroundColor Blue

$optimizedConfig = @"
# Neo4j Cache-Optimized Configuration for Legal AI
# Designed for low memory usage with aggressive caching

#*****************************************************************
# Memory Settings - Optimized for ${HeapSize}MB heap / ${PageCache}MB page cache
#*****************************************************************
server.memory.heap.initial_size=${HeapSize}m
server.memory.heap.max_size=${HeapSize}m
server.memory.pagecache.size=${PageCache}m

#*****************************************************************
# Network Settings
#*****************************************************************
server.default_listen_address=0.0.0.0
server.bolt.enabled=true
server.bolt.listen_address=localhost:7687
server.http.enabled=true
server.http.listen_address=localhost:7474

#*****************************************************************
# Performance Optimizations for Caching
#*****************************************************************
# Disable query logging for performance
dbms.logs.query.enabled=false
dbms.logs.query.threshold=0s

# Disable metrics collection
metrics.enabled=false
server.metrics.enabled=false

# Optimize for read-heavy workloads (legal case analysis)
dbms.transaction.timeout=30s
dbms.transaction.concurrent.maximum=200

# Connection pooling
server.bolt.thread_pool_max_size=100
server.bolt.thread_pool_keep_alive=5m

#*****************************************************************
# Legal AI Specific Optimizations
#*****************************************************************
# Enable parallel query execution
cypher.enable_parallel_runtime=true
cypher.enable_parallel_runtime_support=true

# Cache query plans aggressively
dbms.query_cache_size=1000
dbms.query_statistics_divergence_threshold=0.75

#*****************************************************************
# Authentication (Default for development)
#*****************************************************************
dbms.security.auth_enabled=false
# For production, set: dbms.security.auth_enabled=true

#*****************************************************************
# Backup and Logs (Minimal for performance)
#*****************************************************************
dbms.logs.debug.level=WARN
dbms.logs.gc.enabled=false

#*****************************************************************
# Legal AI Database Settings
#*****************************************************************
# Initial default database
initial.dbms.default_database=legal_ai_graph
"@

# Write configuration
$optimizedConfig | Out-File -FilePath $configPath -Encoding UTF8 -Force
Write-Host "✅ Configuration written: $configPath" -ForegroundColor Green

# Create startup scripts
$startScript = @"
@echo off
REM Neo4j Legal AI Startup Script - Cache Optimized
echo 🔗 Starting Neo4j Legal AI Graph Database...
echo 📊 Configuration: ${HeapSize}MB heap, ${PageCache}MB page cache
echo 🌐 Web Interface: http://localhost:7474
echo ⚡ Bolt Protocol: bolt://localhost:7687
echo.

cd /d "$extractPath"
bin\neo4j.bat console
"@

$startScriptPath = "$extractPath\START-NEO4J-LEGAL-AI.bat"
$startScript | Out-File -FilePath $startScriptPath -Encoding ASCII -Force
Write-Host "✅ Start script created: $startScriptPath" -ForegroundColor Green

# Create service installation script
$serviceScript = @"
@echo off
REM Install Neo4j as Windows Service
echo 🔧 Installing Neo4j as Windows Service...

cd /d "$extractPath"
bin\neo4j.bat install-service
if %errorlevel% == 0 (
    echo ✅ Service installed successfully
    echo 🚀 Starting Neo4j service...
    net start neo4j
    echo.
    echo 🌐 Neo4j Browser: http://localhost:7474
    echo ⚡ Service Status: sc query neo4j
) else (
    echo ❌ Service installation failed
    echo ℹ️ Run as Administrator or use START-NEO4J-LEGAL-AI.bat
)
pause
"@

$serviceScriptPath = "$extractPath\INSTALL-NEO4J-SERVICE.bat"
$serviceScript | Out-File -FilePath $serviceScriptPath -Encoding ASCII -Force
Write-Host "✅ Service script created: $serviceScriptPath" -ForegroundColor Green

# Create legal AI sample data script
$sampleDataScript = @"
// Legal AI Sample Data - Case Relationships and Precedents
// Run this in Neo4j Browser after startup

// Create legal case nodes
CREATE (c1:Case {
    id: 'case_001',
    title: 'Smith v. Johnson Contract Dispute',
    case_type: 'contract',
    jurisdiction: 'federal',
    year: 2023,
    status: 'closed',
    outcome: 'settlement'
})

CREATE (c2:Case {
    id: 'case_002', 
    title: 'Personal Injury - Vehicle Accident',
    case_type: 'tort',
    jurisdiction: 'state',
    year: 2023,
    status: 'active',
    damages: 75000
})

// Create precedent nodes
CREATE (p1:Precedent {
    id: 'precedent_001',
    title: 'Landmark Contract Interpretation',
    citation: '123 F.3d 456 (2022)',
    authority: 'high',
    binding: true,
    court: 'Federal Circuit'
})

CREATE (p2:Precedent {
    id: 'precedent_002',
    title: 'Negligence Standard for Vehicles',
    citation: '456 State 789 (2021)',
    authority: 'medium',
    binding: false,
    court: 'State Supreme'
})

// Create relationships
CREATE (c1)-[:CITES {relevance: 'high', weight: 0.85, page: 23}]->(p1)
CREATE (c2)-[:REFERENCES {relevance: 'medium', weight: 0.70, context: 'negligence_standard'}]->(p2)
CREATE (c1)-[:RELATED_TO {similarity: 0.65, relationship_type: 'similar_facts'}]->(c2)

// Create legal entities
CREATE (e1:Entity {
    id: 'entity_001',
    name: 'Smith Corporation',
    type: 'corporate',
    role: 'plaintiff'
})

CREATE (e2:Entity {
    id: 'entity_002',
    name: 'Johnson Industries',
    type: 'corporate', 
    role: 'defendant'
})

// Link entities to cases
CREATE (e1)-[:PARTY_IN {role: 'plaintiff', representation: 'law_firm_a'}]->(c1)
CREATE (e2)-[:PARTY_IN {role: 'defendant', representation: 'law_firm_b'}]->(c1)

// Create indexes for performance
CREATE INDEX case_id_index FOR (c:Case) ON (c.id);
CREATE INDEX precedent_citation_index FOR (p:Precedent) ON (p.citation);
CREATE INDEX entity_name_index FOR (e:Entity) ON (e.name);

// Legal AI query examples:
// MATCH (c:Case)-[:CITES]->(p:Precedent) WHERE c.case_type = 'contract' RETURN c, p;
// MATCH (c1:Case)-[:RELATED_TO]-(c2:Case) RETURN c1, c2;
// MATCH (e:Entity)-[:PARTY_IN]->(c:Case) WHERE e.type = 'corporate' RETURN e, c;
"@

$sampleDataPath = "$extractPath\legal-ai-sample-data.cypher"
$sampleDataScript | Out-File -FilePath $sampleDataPath -Encoding UTF8 -Force
Write-Host "✅ Sample data script: $sampleDataPath" -ForegroundColor Green

# Create connection test script
$testScript = @"
const neo4j = require('neo4j-driver');

async function testNeo4jConnection() {
    const driver = neo4j.driver('bolt://localhost:7687');
    
    try {
        const session = driver.session();
        const result = await session.run('RETURN "Neo4j Legal AI Connected!" as message');
        
        console.log('✅ Neo4j Connection Success');
        console.log('📊 Result:', result.records[0].get('message'));
        
        // Test legal AI queries
        const caseQuery = await session.run('MATCH (c:Case) RETURN count(c) as case_count');
        console.log('📁 Total Cases:', caseQuery.records[0].get('case_count').toNumber());
        
        await session.close();
    } catch (error) {
        console.error('❌ Neo4j Connection Failed:', error.message);
    } finally {
        await driver.close();
    }
}

testNeo4jConnection();
"@

$testScriptPath = "$extractPath\test-neo4j-connection.js"
$testScript | Out-File -FilePath $testScriptPath -Encoding UTF8 -Force
Write-Host "✅ Connection test: $testScriptPath" -ForegroundColor Green

# Install as service if requested
if ($InstallService) {
    try {
        Push-Location $extractPath
        & ".\bin\neo4j.bat" install-service
        Start-Service neo4j
        Write-Host "✅ Neo4j service installed and started" -ForegroundColor Green
        Pop-Location
    }
    catch {
        Write-Host "⚠️ Service installation failed: $_" -ForegroundColor Yellow
        Write-Host "   Use $serviceScriptPath as Administrator" -ForegroundColor Yellow
        Pop-Location
    }
}

# Display summary
Write-Host ""
Write-Host "🎉 Neo4j Legal AI Setup Complete!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Installation Path: $extractPath" -ForegroundColor White
Write-Host "💾 Memory Configuration: ${HeapSize}MB heap, ${PageCache}MB page cache" -ForegroundColor White
Write-Host "🌐 Web Interface: http://localhost:7474" -ForegroundColor White
Write-Host "⚡ Bolt Connection: bolt://localhost:7687" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Quick Start Commands:" -ForegroundColor Cyan
Write-Host "   Start manually: $startScriptPath" -ForegroundColor White
Write-Host "   Install service: $serviceScriptPath (as Admin)" -ForegroundColor White
Write-Host "   Load sample data: Open Neo4j Browser → paste $sampleDataPath" -ForegroundColor White
Write-Host "   Test connection: node $testScriptPath" -ForegroundColor White
Write-Host ""
Write-Host "⚡ Performance Features Enabled:" -ForegroundColor Cyan
Write-Host "   ✅ Query logging disabled" -ForegroundColor White
Write-Host "   ✅ Metrics collection disabled" -ForegroundColor White  
Write-Host "   ✅ Parallel query runtime enabled" -ForegroundColor White
Write-Host "   ✅ Aggressive query plan caching" -ForegroundColor White
Write-Host "   ✅ Optimized for read-heavy legal workloads" -ForegroundColor White
Write-Host ""
Write-Host "🔗 Integration Ready For:" -ForegroundColor Cyan
Write-Host "   • Redis SWR cache layer (port 6380)" -ForegroundColor White
Write-Host "   • TinyGo WASM graph processing" -ForegroundColor White
Write-Host "   • XState background refresh" -ForegroundColor White
Write-Host "   • IndexedDB client snapshots" -ForegroundColor White
Write-Host ""

# Create integration status checker
$statusChecker = @"
@echo off
echo 🔍 Legal AI System Status Check
echo ================================

echo 📊 Checking Neo4j...
curl -s http://localhost:7474 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Neo4j Web Interface: http://localhost:7474
) else (
    echo ❌ Neo4j not running - use $startScriptPath
)

echo 📊 Checking Redis Cache...
redis-cli ping >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Redis Cache: Available
) else (
    echo ⚠️ Redis Cache: Not available
)

echo 📊 Checking Graph Service...
curl -s http://localhost:7474/health >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Simple Graph Service: http://localhost:7474
) else (
    echo ❌ Graph Service not running
)

echo.
echo 💡 Run this: $startScriptPath
echo 💡 Then visit: http://localhost:7474
echo.
pause
"@

$statusPath = "$InstallPath\CHECK-LEGAL-AI-STATUS.bat"
$statusChecker | Out-File -FilePath $statusPath -Encoding ASCII -Force
Write-Host "📊 Status checker: $statusPath" -ForegroundColor Blue

Write-Host "🎯 Next: Run $startScriptPath to start Neo4j!" -ForegroundColor Yellow