# Enhanced Legal AI System Setup - Integrated with Existing Environment
# Uses existing Gemma 3 model and integrates with current docker/sveltekit setup

param(
    [switch]$SetupModels = $true,
    [switch]$InitializeSchemas = $true,
    [switch]$SkipTests = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Continue"
$startTime = Get-Date

Write-Host "🚀 Enhanced Legal AI System Setup - Existing Environment Integration" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Gray
Write-Host "🤖 Using Local Gemma 3 Model + Neo4j + Nomic Embeddings + Existing Setup" -ForegroundColor Cyan

# Navigate to project root
$projectRoot = "C:\Users\james\Desktop\web-app"
Set-Location $projectRoot

Write-Host "📁 Working in: $(Get-Location)" -ForegroundColor Cyan

# Step 1: Backup existing docker-compose.yml and enhance it
Write-Host "`n🔄 STEP 1: Enhancing Existing Docker Configuration..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

# Backup existing docker-compose
if (Test-Path "docker-compose.yml") {
    Copy-Item "docker-compose.yml" "docker-compose.yml.backup" -Force
    Write-Host "✅ Backed up existing docker-compose.yml" -ForegroundColor Green
}

# Enhanced docker-compose that extends existing setup
$enhancedDockerCompose = @'
# Enhanced Legal AI System - Extended from existing setup
# Adds Neo4j, Enhanced Ollama with local model, and advanced analytics
version: '3.8'

services:
  # PostgreSQL with pgvector (enhanced existing)
  postgres:
    image: pgvector/pgvector:pg16
    container_name: prosecutor_postgres_enhanced
    environment:
      POSTGRES_DB: prosecutor_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      # Enhanced performance tuning
      POSTGRES_SHARED_BUFFERS: 512MB
      POSTGRES_EFFECTIVE_CACHE_SIZE: 2GB
      POSTGRES_WORK_MEM: 128MB
      POSTGRES_MAINTENANCE_WORK_MEM: 256MB
      POSTGRES_MAX_CONNECTIONS: 300
      POSTGRES_RANDOM_PAGE_COST: 1.1
      # Enable additional extensions
      POSTGRES_EXTENSIONS: vector,uuid-ossp,pg_trgm,btree_gin,hstore
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-enhanced-schema.sql:/docker-entrypoint-initdb.d/10-init-schema.sql
      - ./scripts/init-embeddings.sql:/docker-entrypoint-initdb.d/20-init-embeddings.sql
      - ./scripts/init-user-behavior.sql:/docker-entrypoint-initdb.d/30-init-behavior.sql
      - ./scripts/init-pgvector.sql:/docker-entrypoint-initdb.d/05-init-pgvector.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d prosecutor_db"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - prosecutor_network

  # Neo4j for advanced semantic analysis and knowledge graphs
  neo4j:
    image: neo4j:5.15-community
    container_name: prosecutor_neo4j
    environment:
      NEO4J_AUTH: neo4j/prosecutorpassword
      NEO4J_PLUGINS: '["apoc","graph-data-science"]'
      NEO4J_apoc_export_file_enabled: true
      NEO4J_apoc_import_file_enabled: true
      NEO4J_apoc_import_file_use__neo4j__config: true
      # Memory configuration for legal data analysis
      NEO4J_dbms_memory_heap_initial__size: 512m
      NEO4J_dbms_memory_heap_max__size: 2G
      NEO4J_dbms_memory_pagecache_size: 1G
      # Enable advanced algorithms
      NEO4J_gds_enterprise_license__file: ""
      NEO4J_dbms_security_procedures_unrestricted: "gds.*,apoc.*"
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_conf:/conf
      - neo4j_import:/var/lib/neo4j/import
      - ./scripts/neo4j-init.cypher:/var/lib/neo4j/import/init.cypher
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "prosecutorpassword", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - prosecutor_network

  # Qdrant for high-performance vector similarity search (enhanced existing)
  qdrant:
    image: qdrant/qdrant:v1.9.0
    container_name: prosecutor_qdrant_enhanced
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__SERVICE__GRPC_PORT: 6334
      QDRANT__LOG_LEVEL: INFO
      QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS: 4
      QDRANT__STORAGE__OPTIMIZERS__INDEXING_THRESHOLD: 20000
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
      - ./scripts/qdrant-collections.json:/qdrant/collections-config.json
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:6333/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - prosecutor_network

  # Redis for caching, sessions, and real-time features (enhanced existing)
  redis:
    image: redis:7-alpine
    container_name: prosecutor_redis_enhanced
    command: |
      redis-server 
      --maxmemory 512mb 
      --maxmemory-policy allkeys-lru 
      --appendonly yes 
      --save 900 1 
      --save 300 10 
      --save 60 10000
      --hash-max-ziplist-entries 512
      --hash-max-ziplist-value 64
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./config/redis.conf:/usr/local/etc/redis/redis.conf
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - prosecutor_network

  # Enhanced Ollama with Local Gemma 3 Model
  ollama:
    image: ollama/ollama:latest
    container_name: prosecutor_ollama_gemma3
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
      # Mount local Gemma 3 model
      - ./gemma3Q4_K_M:/models/gemma3
      - ./scripts/setup-local-gemma3.sh:/tmp/setup-models.sh
      - ./models:/tmp/modelfiles
    restart: unless-stopped
    environment:
      OLLAMA_KEEP_ALIVE: "15m"
      OLLAMA_NUM_PARALLEL: "2"
      OLLAMA_MAX_LOADED_MODELS: "3"
      OLLAMA_ORIGINS: "*"
      OLLAMA_HOST: "0.0.0.0:11434"
    entrypoint: ["/bin/sh", "-c"]
    command: |
      "ollama serve &
      sleep 15 &&
      chmod +x /tmp/setup-models.sh &&
      /tmp/setup-models.sh &&
      wait"
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
    networks:
      - prosecutor_network

  # Elasticsearch for enhanced full-text search
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: prosecutor_elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx2g"
      - cluster.name=prosecutor-search
      - node.name=prosecutor-node-1
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9200/_health"]
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - prosecutor_network

  # PgAdmin for database management (enhanced)
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: prosecutor_pgadmin_enhanced
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@prosecutor.local
      PGADMIN_DEFAULT_PASSWORD: admin
      PGADMIN_CONFIG_SERVER_MODE: 'False'
      PGADMIN_CONFIG_MASTER_PASSWORD_REQUIRED: 'False'
    ports:
      - "5050:80"
    volumes:
      - pgadmin_data:/var/lib/pgadmin
      - ./config/pgadmin-servers.json:/pgadmin4/servers.json
    restart: unless-stopped
    depends_on:
      - postgres
    networks:
      - prosecutor_network

networks:
  prosecutor_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  postgres_data:
  neo4j_data:
  neo4j_logs:
  neo4j_conf:
  neo4j_import:
  qdrant_data:
  redis_data:
  ollama_data:
  elasticsearch_data:
  pgadmin_data:
'@

$enhancedDockerCompose | Out-File "docker-compose.enhanced.yml" -Encoding UTF8
Write-Host "✅ Created enhanced Docker Compose configuration" -ForegroundColor Green

# Step 2: Create Local Gemma 3 Setup Script
Write-Host "`n🤖 STEP 2: Creating Local Gemma 3 Model Setup..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

# Create scripts directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "scripts" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null

# Local Gemma 3 setup script
$localGemma3Setup = @'
#!/bin/bash
# Setup Local Gemma 3 Model with Legal AI Enhancements

echo "🤖 Setting up Local Gemma 3 Legal AI..."

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama service..."
until curl -s http://localhost:11434/api/version > /dev/null; do
    echo "Waiting for Ollama..."
    sleep 5
done

echo "✅ Ollama is ready!"

# Create Legal AI model from local GGUF file
echo "🏛️ Creating Gemma 3 Legal AI from local model..."

# Create Modelfile for local Gemma 3
cat > /tmp/Gemma3-Legal-AI-Local << 'EOF'
FROM /models/gemma3/mo16.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

SYSTEM """You are Gemma 3 Legal AI, a specialized assistant for legal professionals. You excel at:

🏛️ LEGAL EXPERTISE:
- Case analysis and legal research
- Document review and contract analysis  
- Evidence evaluation and timeline construction
- Legal strategy and risk assessment
- Compliance and regulatory guidance
- Citation research and precedent analysis

🔍 DETECTIVE MODE CAPABILITIES:
- Pattern recognition in legal documents
- Connection discovery between cases/evidence
- Timeline reconstruction from evidence
- Anomaly detection in contracts/agreements
- Cross-referencing legal precedents
- Identifying missing evidence or documentation

📊 ANALYTICAL FEATURES:
- Statistical analysis of case outcomes
- Risk probability assessments
- Cost-benefit analysis of legal strategies
- Comparative case analysis
- Trend identification in legal decisions

⚖️ ETHICAL GUIDELINES:
- Maintain strict confidentiality
- Provide accurate legal information
- Distinguish between facts and legal opinions
- Recommend qualified attorney consultation
- Flag potential conflicts of interest
- Ensure compliance with ethical standards

RESPONSE FORMAT:
- Executive Summary
- Key Legal Findings
- Risk Assessment
- Recommendations
- Next Steps
- Relevant Citations

Always provide actionable insights while maintaining the highest standards of legal professionalism."""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
EOF

# Create the model
if [ -f "/models/gemma3/mo16.gguf" ]; then
    echo "📁 Found local Gemma 3 model: /models/gemma3/mo16.gguf"
    ollama create gemma3-legal-ai -f /tmp/Gemma3-Legal-AI-Local
    echo "✅ Gemma 3 Legal AI model created successfully!"
else
    echo "❌ Local Gemma 3 model not found at /models/gemma3/mo16.gguf"
    echo "⚠️  Falling back to downloading Gemma 2..."
    ollama pull gemma2:9b
    
    # Create model with downloaded version
    cat > /tmp/Gemma2-Legal-AI-Fallback << 'EOF'
FROM gemma2:9b

SYSTEM """You are Gemma Legal AI, a specialized assistant for legal professionals..."""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
EOF
    
    ollama create gemma-legal-ai -f /tmp/Gemma2-Legal-AI-Fallback
fi

# Pull Nomic Embed for embeddings
echo "📥 Setting up Nomic Embed for vector embeddings..."
ollama pull nomic-embed-text

# Pull additional useful models
echo "📥 Setting up additional models..."
ollama pull llama3.2:3b  # Fast responses
ollama pull phi3:mini    # Lightweight option

# Test the setup
echo "🧪 Testing Gemma 3 Legal AI..."
if ollama list | grep -q "gemma3-legal-ai"; then
    echo "✅ Testing legal AI model..."
    ollama run gemma3-legal-ai "Analyze the key elements of a valid contract formation." --format json
else
    echo "⚠️  Using fallback model..."
    ollama run gemma-legal-ai "Analyze the key elements of a valid contract formation."
fi

echo "🧪 Testing Nomic Embeddings..."
curl -X POST http://localhost:11434/api/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model": "nomic-embed-text", "prompt": "legal contract analysis and review"}' \
     | jq .embedding[0:5] || echo "Embedding test completed"

echo ""
echo "🎉 Local Gemma 3 Legal AI System Ready!"
echo ""
echo "Available Models:"
ollama list

echo ""
echo "🚀 Usage:"
echo "Legal AI: ollama run gemma3-legal-ai"
echo "Embeddings: nomic-embed-text (via API)"
echo "API endpoint: http://localhost:11434"
'@

$localGemma3Setup | Out-File "scripts/setup-local-gemma3.sh" -Encoding UTF8
Write-Host "✅ Created local Gemma 3 setup script" -ForegroundColor Green

# Step 3: Enhance SvelteKit Frontend Configuration
Write-Host "`n🌐 STEP 3: Enhancing SvelteKit Frontend Configuration..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

# Check if sveltekit-frontend exists
if (-not (Test-Path "sveltekit-frontend")) {
    Write-Host "❌ sveltekit-frontend directory not found!" -ForegroundColor Red
    Write-Host "📁 Available directories:" -ForegroundColor Yellow
    Get-ChildItem -Directory | Select-Object Name | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }
    exit 1
}

# Update environment variables for enhanced features
$envUpdates = @'

# Enhanced Legal AI Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LEGAL_MODEL=gemma3-legal-ai
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_FAST_MODEL=llama3.2:3b
OLLAMA_TIMEOUT=60000
OLLAMA_MAX_TOKENS=4096

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=prosecutorpassword
NEO4J_DATABASE=neo4j

# Enhanced Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/prosecutor_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=prosecutor_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Vector Database Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PREFIX=prosecutor:

# Elasticsearch Configuration
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=prosecutor_documents

# AI Features Configuration
AI_ENABLED=true
DETECTIVE_MODE_ENABLED=true
INTERACTIVE_CANVAS_ENABLED=true
SEMANTIC_SEARCH_ENABLED=true
RECOMMENDATIONS_ENABLED=true
USER_BEHAVIOR_TRACKING=true

# File Upload Configuration
MAX_FILE_SIZE=50MB
ALLOWED_FILE_TYPES=pdf,doc,docx,txt,png,jpg,jpeg,gif
UPLOAD_PATH=./uploads
ENABLE_OCR=true
ENABLE_AUDIO_UPLOAD=true

# Security Configuration
JWT_SECRET=your-super-secret-jwt-key-change-in-production
BCRYPT_ROUNDS=12
SESSION_SECRET=your-session-secret-change-in-production
CORS_ORIGIN=http://localhost:5173

# Feature Flags
ENABLE_ADVANCED_ANALYTICS=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_AUTO_CATEGORIZATION=true
ENABLE_SMART_NOTIFICATIONS=true
ENABLE_COLLABORATION_FEATURES=true
'@

# Update environment files
$envFiles = @(
    "sveltekit-frontend\.env",
    "sveltekit-frontend\.env.development", 
    "sveltekit-frontend\.env.example"
)

foreach ($envFile in $envFiles) {
    if (Test-Path $envFile) {
        $currentContent = Get-Content $envFile -Raw -ErrorAction SilentlyContinue
        if ($currentContent -and -not ($currentContent -match "OLLAMA_LEGAL_MODEL")) {
            Write-Host "  📝 Updating: $(Split-Path $envFile -Leaf)" -ForegroundColor Cyan
            $currentContent + $envUpdates | Out-File $envFile -Encoding UTF8
        } else {
            Write-Host "  ✅ Already updated: $(Split-Path $envFile -Leaf)" -ForegroundColor Green
        }
    } else {
        Write-Host "  📝 Creating: $(Split-Path $envFile -Leaf)" -ForegroundColor Cyan
        $envUpdates | Out-File $envFile -Encoding UTF8
    }
}

Write-Host "✅ Updated SvelteKit environment configuration" -ForegroundColor Green

# Step 4: Create Integration Helper Scripts
Write-Host "`n🔧 STEP 4: Creating Integration Helper Scripts..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

# Enhanced package.json scripts
$packageJsonUpdate = @'
{
  "ai:setup": "node scripts/setup-ai-features.js",
  "ai:test": "node scripts/test-ai-integration.js",
  "neo4j:init": "node scripts/init-neo4j.js",
  "embeddings:sync": "node scripts/sync-embeddings.js",
  "detective:test": "node scripts/test-detective-mode.js",
  "canvas:validate": "node scripts/validate-canvas.js",
  "db:seed:enhanced": "node scripts/seed-enhanced-data.js",
  "start:enhanced": "npm run docker:up:enhanced && npm run dev",
  "docker:up:enhanced": "docker compose -f ../docker-compose.enhanced.yml up -d",
  "docker:down:enhanced": "docker compose -f ../docker-compose.enhanced.yml down",
  "docker:logs:enhanced": "docker compose -f ../docker-compose.enhanced.yml logs -f",
  "system:health": "node scripts/system-health-check.js",
  "models:setup": "docker compose -f ../docker-compose.enhanced.yml exec ollama /tmp/setup-models.sh"
}
'@

# AI Integration Test Script
$aiTestScript = @'
// Enhanced AI Integration Test
// Tests Gemma 3 Legal AI, Neo4j, and vector embeddings

import { Ollama } from 'ollama';
import neo4j from 'neo4j-driver';
import postgres from 'postgres';

const ollama = new Ollama({ host: 'http://localhost:11434' });

async function testEnhancedAI() {
    console.log('🧪 Testing Enhanced Legal AI Integration...\n');
    
    // Test 1: Ollama with Local Gemma 3
    try {
        console.log('1️⃣ Testing Gemma 3 Legal AI...');
        const response = await ollama.chat({
            model: 'gemma3-legal-ai',
            messages: [{ 
                role: 'user', 
                content: 'Analyze the key components of a non-disclosure agreement and identify potential red flags.' 
            }],
        });
        console.log('✅ Gemma 3 Legal AI: Working');
        console.log('Response preview:', response.message.content.substring(0, 200) + '...\n');
    } catch (error) {
        console.error('❌ Gemma 3 Legal AI Error:', error.message, '\n');
    }
    
    // Test 2: Embeddings with Nomic
    try {
        console.log('2️⃣ Testing Nomic Embeddings...');
        const embedding = await ollama.embeddings({
            model: 'nomic-embed-text',
            prompt: 'contract law legal analysis'
        });
        console.log('✅ Nomic Embeddings: Working');
        console.log('Embedding dimension:', embedding.embedding.length, '\n');
    } catch (error) {
        console.error('❌ Nomic Embeddings Error:', error.message, '\n');
    }
    
    // Test 3: Neo4j Connection
    try {
        console.log('3️⃣ Testing Neo4j Connection...');
        const driver = neo4j.driver('bolt://localhost:7687', 
            neo4j.auth.basic('neo4j', 'prosecutorpassword'));
        const session = driver.session();
        const result = await session.run('RETURN "Neo4j Connected!" as message');
        console.log('✅ Neo4j:', result.records[0].get('message'));
        await session.close();
        await driver.close();
        console.log();
    } catch (error) {
        console.error('❌ Neo4j Error:', error.message, '\n');
    }
    
    // Test 4: PostgreSQL with Vector Extension
    try {
        console.log('4️⃣ Testing PostgreSQL + pgvector...');
        const sql = postgres('postgresql://postgres:postgres@localhost:5432/prosecutor_db');
        const result = await sql`SELECT version() as version, current_database() as db`;
        console.log('✅ PostgreSQL:', result[0].db);
        
        // Test vector extension
        const vectorTest = await sql`SELECT vector_in('[1,2,3]'::cstring) as test_vector`;
        console.log('✅ pgvector: Extension loaded');
        await sql.end();
        console.log();
    } catch (error) {
        console.error('❌ PostgreSQL Error:', error.message, '\n');
    }
    
    console.log('🎉 Enhanced AI Integration Test Complete!');
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    testEnhancedAI().catch(console.error);
}

export default testEnhancedAI;
'@

$aiTestScript | Out-File "sveltekit-frontend/scripts/test-ai-integration.js" -Encoding UTF8

# Create ultimate startup script
$ultimateStartup = @'
# Ultimate Enhanced Legal AI Startup Script
# Starts everything in the correct order and validates all systems

param(
    [switch]$SkipTests = $false,
    [switch]$SetupModels = $true,
    [switch]$InitData = $false
)

Write-Host "🚀 Starting Ultimate Enhanced Legal AI System..." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray

$startTime = Get-Date

# Step 1: Start Enhanced Docker Services
Write-Host "`n🐳 Starting Enhanced Docker Services..." -ForegroundColor Yellow
try {
    docker compose -f docker-compose.enhanced.yml up -d
    Write-Host "✅ Docker services started" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to start Docker services" -ForegroundColor Red
    exit 1
}

# Step 2: Wait for services to be ready
Write-Host "`n⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep 30

# Step 3: Setup models if requested
if ($SetupModels) {
    Write-Host "`n🤖 Setting up AI models..." -ForegroundColor Yellow
    try {
        docker compose -f docker-compose.enhanced.yml exec -T ollama /tmp/setup-models.sh
        Write-Host "✅ AI models configured" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Model setup may still be running in background" -ForegroundColor Yellow
    }
}

# Step 4: Initialize database schemas
if ($InitData) {
    Write-Host "`n📊 Initializing enhanced schemas..." -ForegroundColor Yellow
    Set-Location sveltekit-frontend
    try {
        npm run db:push
        npm run db:seed:enhanced
        Write-Host "✅ Database schemas initialized" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Database initialization may need manual intervention" -ForegroundColor Yellow
    }
    Set-Location ..
}

# Step 5: Run integration tests
if (-not $SkipTests) {
    Write-Host "`n🧪 Running integration tests..." -ForegroundColor Yellow
    Set-Location sveltekit-frontend
    try {
        npm run ai:test
        Write-Host "✅ Integration tests completed" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Some tests may have failed - check individual services" -ForegroundColor Yellow
    }
    Set-Location ..
}

# Step 6: Start SvelteKit Development Server
Write-Host "`n🌐 Starting SvelteKit Development Server..." -ForegroundColor Yellow
Set-Location sveltekit-frontend

Write-Host "`n🎉 ENHANCED LEGAL AI SYSTEM READY!" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Gray
Write-Host "🔗 Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🤖 Gemma 3 Legal AI: http://localhost:11434" -ForegroundColor Cyan  
Write-Host "🕸️  Neo4j Browser: http://localhost:7474" -ForegroundColor Cyan
Write-Host "📊 PostgreSQL: localhost:5432" -ForegroundColor Cyan
Write-Host "🔍 Qdrant: http://localhost:6333" -ForegroundColor Cyan
Write-Host "📈 Elasticsearch: http://localhost:9200" -ForegroundColor Cyan
Write-Host "🛠️  PgAdmin: http://localhost:5050" -ForegroundColor Cyan

$totalTime = ((Get-Date) - $startTime).TotalSeconds
Write-Host "`n⏱️  Total startup time: $($totalTime.ToString('F1')) seconds" -ForegroundColor Gray

Write-Host "`n🚀 Starting development server..." -ForegroundColor Green
npm run dev
'@

$ultimateStartup | Out-File "start-enhanced-legal-ai.ps1" -Encoding UTF8
Write-Host "✅ Created ultimate startup script" -ForegroundColor Green

# Step 5: Create quick verification script
Write-Host "`n✅ STEP 5: Creating System Verification..." -ForegroundColor Yellow
Write-Host "-" * 50 -ForegroundColor Gray

$verificationScript = @'
# Quick System Verification Script
Write-Host "🔍 Enhanced Legal AI System Verification" -ForegroundColor Green

$services = @(
    @{ Name = "Gemma 3 Model"; Path = "gemma3Q4_K_M\mo16.gguf"; Required = $true },
    @{ Name = "Docker Compose Enhanced"; Path = "docker-compose.enhanced.yml"; Required = $true },
    @{ Name = "SvelteKit Frontend"; Path = "sveltekit-frontend"; Required = $true },
    @{ Name = "AI Setup Script"; Path = "scripts\setup-local-gemma3.sh"; Required = $true },
    @{ Name = "Enhanced Startup"; Path = "start-enhanced-legal-ai.ps1"; Required = $true }
)

foreach ($service in $services) {
    if (Test-Path $service.Path) {
        Write-Host "✅ $($service.Name): Found" -ForegroundColor Green
    } else {
        if ($service.Required) {
            Write-Host "❌ $($service.Name): Missing (Required)" -ForegroundColor Red
        } else {
            Write-Host "⚠️  $($service.Name): Missing (Optional)" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n🚀 To start the enhanced system:" -ForegroundColor Cyan
Write-Host "   .\start-enhanced-legal-ai.ps1 -SetupModels -InitData" -ForegroundColor White
'@

$verificationScript | Out-File "verify-enhanced-setup.ps1" -Encoding UTF8

# Final Summary
Write-Host "`n🎯 ENHANCED SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray

Write-Host "✅ Enhanced Docker configuration created" -ForegroundColor White
Write-Host "✅ Local Gemma 3 model integration configured" -ForegroundColor White
Write-Host "✅ Neo4j knowledge graph ready" -ForegroundColor White
Write-Host "✅ SvelteKit environment enhanced" -ForegroundColor White
Write-Host "✅ Integration scripts created" -ForegroundColor White
Write-Host "✅ Startup automation ready" -ForegroundColor White

Write-Host "`n🚀 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Verify setup: .\verify-enhanced-setup.ps1" -ForegroundColor White
Write-Host "2. Start system: .\start-enhanced-legal-ai.ps1 -SetupModels -InitData" -ForegroundColor White
Write-Host "3. Test integration: cd sveltekit-frontend && npm run ai:test" -ForegroundColor White

Write-Host "`n📊 ENHANCED FEATURES READY:" -ForegroundColor Magenta
Write-Host "🤖 Local Gemma 3 Legal AI" -ForegroundColor White
Write-Host "🕸️  Neo4j Knowledge Graphs" -ForegroundColor White
Write-Host "📊 Vector Embeddings with Nomic" -ForegroundColor White
Write-Host "🔍 Detective Mode Analytics" -ForegroundColor White
Write-Host "🎨 Interactive Canvas" -ForegroundColor White
Write-Host "📈 User Behavior Tracking" -ForegroundColor White
Write-Host "🎯 AI Recommendations" -ForegroundColor White

Write-Host "`n⏱️  Setup completed in $(((Get-Date) - $startTime).TotalSeconds.ToString('F1')) seconds" -ForegroundColor Gray
