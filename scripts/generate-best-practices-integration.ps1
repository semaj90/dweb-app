# Generate Best Practices Integration Script
# Merges all system components with security, performance, and monitoring best practices

param(
    [string]$OutputPath = "INTEGRATION_BEST_PRACTICES.md",
    [switch]$IncludeMetrics,
    [switch]$IncludeSecurityChecklist,
    [switch]$IncludePerformanceGuide,
    [switch]$GenerateConfigTemplates
)

function Write-Section {
    param($Title, $Content)
    return @"

## $Title

$Content

"@
}

function Write-CodeBlock {
    param($Language, $Code)
    return @"
``````$Language
$Code
``````
"@
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$integrationGuide = @"
# Legal AI System Integration Best Practices
Generated: $timestamp

# 🎯 Executive Summary

This document provides comprehensive integration best practices for the Legal AI System, covering SvelteKit 2, Go microservices, PostgreSQL with pgvector, Redis, Qdrant, MinIO, and Ollama integration with optimal security, performance, and monitoring.

# 🏗️ Architecture Overview

## Technology Stack Integration
- **Frontend**: SvelteKit 2 + Svelte 5 with TypeScript
- **Backend**: Go microservices with Gin framework
- **Database**: PostgreSQL 17 with pgvector extension
- **Vector DB**: Qdrant for semantic search
- **Cache**: Redis for session and application caching
- **Storage**: MinIO for file storage
- **AI/ML**: Ollama + Claude integration
- **Monitoring**: Prometheus + Grafana integration

## Service Communication Pattern
``````mermaid
graph TD
    A[SvelteKit Frontend] --> B[Go API Gateway]
    B --> C[PostgreSQL + pgvector]
    B --> D[Qdrant Vector DB]
    B --> E[Redis Cache]
    B --> F[MinIO Storage]
    B --> G[Ollama LLM]
    G --> H[Claude API]
``````

$(Write-Section "🔐 Security Integration Best Practices" @"
### API Security
- JWT tokens with RS256 algorithm
- Rate limiting: 100 req/min per IP, 1000 req/min per authenticated user
- CORS whitelist for specific origins
- Input validation with Go validator package
- SQL injection prevention with prepared statements

### Database Security
- Connection pooling with encrypted connections
- Row-level security (RLS) for multi-tenant data
- Audit logging for all data modifications
- Backup encryption with AES-256
- Regular security audits with automated scanning

### File Upload Security
``````go
// Secure file upload implementation
func SecureFileUpload(c *gin.Context) {
    file, header, err := c.Request.FormFile("file")
    if err != nil {
        c.JSON(400, gin.H{"error": "Invalid file"})
        return
    }
    
    // Validate file type
    if !isAllowedFileType(header.Filename) {
        c.JSON(400, gin.H{"error": "File type not allowed"})
        return
    }
    
    // Scan for malware
    if isInfected, err := scanFile(file); isInfected || err != nil {
        c.JSON(400, gin.H{"error": "File failed security scan"})
        return
    }
    
    // Store in MinIO with encryption
    objectName := generateSecureFileName(header.Filename)
    uploadInfo, err := minioClient.PutObject(ctx, bucketName, objectName, file, header.Size, minio.PutObjectOptions{
        ServerSideEncryption: encrypt.NewSSE(),
    })
}
``````

### Environment Security
``````bash
# Secure environment configuration
export DATABASE_URL="postgresql://user:\$\{SECURE_PASSWORD\}@localhost:5432/legal_ai_db?sslmode=require"
export JWT_SECRET="\$\{RANDOM_JWT_SECRET_256_BITS\}"
export MINIO_ROOT_PASSWORD="\$\{SECURE_MINIO_PASSWORD\}"
export ENCRYPTION_KEY="\$\{AES_256_ENCRYPTION_KEY\}"
``````
"@)

$(Write-Section "⚡ Performance Integration Best Practices" @"
### Database Performance
``````sql
-- Optimal indexes for legal AI queries
CREATE INDEX CONCURRENTLY idx_legal_documents_case_id ON legal_documents(case_id);
CREATE INDEX CONCURRENTLY idx_legal_documents_embedding ON legal_documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX CONCURRENTLY idx_legal_documents_fts ON legal_documents USING GIN (to_tsvector('english', content));
CREATE INDEX CONCURRENTLY idx_legal_documents_created_at ON legal_documents(created_at DESC);

-- Partitioning for large datasets
CREATE TABLE legal_documents_2024 PARTITION OF legal_documents
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
``````

### Caching Strategy
``````go
// Multi-layer caching implementation
type CacheManager struct {
    redis    *redis.Client
    memory   *cache.Cache
    qdrant   *qdrant.Client
}

func (cm *CacheManager) GetWithFallback(key string) (interface{}, error) {
    // L1: Memory cache (fastest)
    if val, found := cm.memory.Get(key); found {
        return val, nil
    }
    
    // L2: Redis cache (fast)
    if val, err := cm.redis.Get(key).Result(); err == nil {
        cm.memory.Set(key, val, 5*time.Minute)
        return val, nil
    }
    
    // L3: Database with vector similarity
    return cm.fetchFromDatabase(key)
}
``````

### Connection Pooling
``````go
// Optimal database connection configuration
config := pgxpool.Config{
    MaxConns:        25,              // Maximum connections
    MinConns:        5,               // Minimum connections
    MaxConnLifetime: time.Hour,       // Connection lifetime
    MaxConnIdleTime: time.Minute * 30, // Idle timeout
}
``````

### SvelteKit Performance
``````typescript
// Optimized SvelteKit load function
export const load: PageServerLoad = async ({ params, depends }) => {
    depends('case:analysis');
    
    // Parallel data loading
    const [caseData, evidence, analysis] = await Promise.all([
        getCaseData(params.id),
        getEvidenceData(params.id),
        getAIAnalysis(params.id)
    ]);
    
    return {
        case: caseData,
        evidence,
        analysis: analysis || null, // Allow graceful degradation
        streamed: {
            detailedAnalysis: getDetailedAnalysis(params.id) // Stream heavy computation
        }
    };
};
``````
"@)

$(Write-Section "🧠 AI/ML Integration Best Practices" @"
### Ollama Integration
``````go
// Robust Ollama client with fallback
type AIService struct {
    ollamaClient *ollama.Client
    claudeClient *claude.Client
    retryPolicy  *backoff.ExponentialBackOff
}

func (ai *AIService) GenerateResponse(prompt string) (string, error) {
    // Try local Ollama first
    if response, err := ai.ollamaClient.Generate(prompt); err == nil {
        return response, nil
    }
    
    // Fallback to Claude API
    return ai.claudeClient.Generate(prompt)
}
``````

### Vector Search Optimization
``````typescript
// Optimized vector search with Qdrant
export async function semanticSearch(
    query: string,
    options: {
        limit?: number;
        threshold?: number;
        filters?: Record<string, any>;
    } = {}
) {
    const embedding = await generateEmbedding(query);
    
    const searchParams = {
        vector: embedding,
        limit: options.limit || 10,
        score_threshold: options.threshold || 0.7,
        filter: options.filters,
        with_payload: true,
        with_vector: false // Optimize bandwidth
    };
    
    const results = await qdrantClient.search('legal_documents', searchParams);
    
    // Cache frequently accessed results
    await cacheResults(query, results);
    
    return results.map(result => ({
        id: result.id,
        score: result.score,
        content: result.payload?.content,
        metadata: result.payload?.metadata
    }));
}
``````

### Embedding Generation
``````go
// Optimized embedding generation with batching
func GenerateEmbeddings(texts []string) ([][]float32, error) {
    const batchSize = 32
    var allEmbeddings [][]float32
    
    for i := 0; i < len(texts); i += batchSize {
        end := i + batchSize
        if end > len(texts) {
            end = len(texts)
        }
        
        batch := texts[i:end]
        embeddings, err := ollamaClient.GenerateEmbeddings(batch)
        if err != nil {
            return nil, fmt.Errorf("embedding generation failed: %w", err)
        }
        
        allEmbeddings = append(allEmbeddings, embeddings...)
    }
    
    return allEmbeddings, nil
}
``````
"@)

$(Write-Section "📊 Monitoring Integration Best Practices" @"
### Metrics Collection
``````go
// Prometheus metrics integration
var (
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )
    
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "http_request_duration_seconds",
            Help: "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )
    
    databaseConnections = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "database_connections_active",
            Help: "Number of active database connections",
        },
        []string{"database"},
    )
)

// Middleware for metrics collection
func MetricsMiddleware() gin.HandlerFunc {
    return gin.HandlerFunc(func(c *gin.Context) {
        start := time.Now()
        
        c.Next()
        
        duration := time.Since(start).Seconds()
        status := strconv.Itoa(c.Writer.Status())
        
        httpRequestsTotal.WithLabelValues(c.Request.Method, c.FullPath(), status).Inc()
        httpRequestDuration.WithLabelValues(c.Request.Method, c.FullPath()).Observe(duration)
    })
}
``````

### Health Checks
``````go
// Comprehensive health check endpoint
func HealthCheck(c *gin.Context) {
    health := HealthStatus{
        Status:    "healthy",
        Timestamp: time.Now(),
        Services:  make(map[string]ServiceHealth),
    }
    
    // Check PostgreSQL
    if err := db.Ping(); err != nil {
        health.Services["postgresql"] = ServiceHealth{Status: "unhealthy", Error: err.Error()}
        health.Status = "degraded"
    } else {
        health.Services["postgresql"] = ServiceHealth{Status: "healthy"}
    }
    
    // Check Redis
    if _, err := redisClient.Ping().Result(); err != nil {
        health.Services["redis"] = ServiceHealth{Status: "unhealthy", Error: err.Error()}
        health.Status = "degraded"
    } else {
        health.Services["redis"] = ServiceHealth{Status: "healthy"}
    }
    
    // Check Qdrant
    if _, err := qdrantClient.HealthCheck(); err != nil {
        health.Services["qdrant"] = ServiceHealth{Status: "unhealthy", Error: err.Error()}
        health.Status = "degraded"
    } else {
        health.Services["qdrant"] = ServiceHealth{Status: "healthy"}
    }
    
    statusCode := 200
    if health.Status == "unhealthy" {
        statusCode = 503
    } else if health.Status == "degraded" {
        statusCode = 207
    }
    
    c.JSON(statusCode, health)
}
``````

### Logging Best Practices
``````go
// Structured logging with context
func LogWithContext(ctx context.Context, level string, message string, fields map[string]interface{}) {
    entry := log.WithContext(ctx).WithFields(logrus.Fields{
        "timestamp": time.Now().UTC(),
        "service":   "legal-ai-api",
        "version":   version,
    })
    
    for k, v := range fields {
        entry = entry.WithField(k, v)
    }
    
    switch level {
    case "debug":
        entry.Debug(message)
    case "info":
        entry.Info(message)
    case "warn":
        entry.Warn(message)
    case "error":
        entry.Error(message)
    }
}
``````
"@)

$(Write-Section "🚀 Deployment Integration Best Practices" @"
### Environment Configuration
``````yaml
# docker-compose.production.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: legal_ai_db
      POSTGRES_USER: legal_admin
      POSTGRES_PASSWORD: \$\{POSTGRES_PASSWORD\}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass \$\{REDIS_PASSWORD\}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    restart: unless-stopped
    
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: \$\{MINIO_ROOT_USER\}
      MINIO_ROOT_PASSWORD: \$\{MINIO_ROOT_PASSWORD\}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
  minio_data:
``````

### CI/CD Pipeline
``````yaml
# .github/workflows/deploy.yml
name: Deploy Legal AI System

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v3
        with:
          go-version: '1.21'
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Run Go tests
        run: |
          cd go-microservice
          go test ./...
      
      - name: Run SvelteKit tests
        run: |
          npm ci
          npm run test
      
      - name: Security scan
        run: |
          npm audit
          go mod download
          go list -json -m all | nancy sleuth

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Deployment steps
          docker-compose -f docker-compose.production.yml up -d
``````
"@)

$(Write-Section "📋 Integration Checklist" @"
### Pre-Production Checklist
- [ ] All services pass health checks
- [ ] Database migrations applied successfully
- [ ] Vector indexes created and optimized
- [ ] SSL/TLS certificates configured
- [ ] Backup and recovery tested
- [ ] Monitoring dashboards configured
- [ ] Security scan passed
- [ ] Performance benchmarks met
- [ ] Error handling tested
- [ ] Failover procedures documented

### Security Checklist
- [ ] JWT secrets rotated
- [ ] Database passwords secured
- [ ] File upload restrictions enforced
- [ ] Rate limiting configured
- [ ] Input validation implemented
- [ ] Audit logging enabled
- [ ] Backup encryption verified
- [ ] Access controls tested
- [ ] Network security configured
- [ ] Penetration testing completed

### Performance Checklist
- [ ] Database queries optimized
- [ ] Indexes created for all search columns
- [ ] Connection pooling configured
- [ ] Caching layers implemented
- [ ] CDN configured for static assets
- [ ] Compression enabled
- [ ] Load testing completed
- [ ] Memory usage optimized
- [ ] Response times < 2s (95th percentile)
- [ ] Horizontal scaling tested
"@)

Write-Host ""
Write-Host "=== Integration Best Practices Summary ===" -ForegroundColor Green
Write-Host ""
Write-Host "✅ Comprehensive security integration" -ForegroundColor Green
Write-Host "✅ Performance optimization patterns" -ForegroundColor Green
Write-Host "✅ AI/ML service integration" -ForegroundColor Green
Write-Host "✅ Monitoring and observability" -ForegroundColor Green
Write-Host "✅ Deployment and CI/CD patterns" -ForegroundColor Green
Write-Host "✅ Production readiness checklists" -ForegroundColor Green
Write-Host ""
"@

$integrationGuide | Out-File -FilePath $OutputPath -Encoding UTF8

Write-Host "✅ Integration best practices guide generated: $OutputPath" -ForegroundColor Green

if ($GenerateConfigTemplates) {
    Write-Host "📝 Generating configuration templates..." -ForegroundColor Cyan
    
    # Production environment template
    $prodEnvTemplate = @"
# Production Environment Configuration
# Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

# Database
DATABASE_URL=postgresql://legal_admin:CHANGE_THIS_PASSWORD@localhost:5432/legal_ai_db?sslmode=require
POSTGRES_PASSWORD=CHANGE_THIS_PASSWORD
DB_MAX_CONNECTIONS=25
DB_MIN_CONNECTIONS=5
DB_CONNECTION_TIMEOUT=30s

# Redis
REDIS_URL=redis://:CHANGE_THIS_PASSWORD@localhost:6379/0
REDIS_PASSWORD=CHANGE_THIS_PASSWORD
REDIS_MAX_CONNECTIONS=10

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=legal_documents
QDRANT_VECTOR_SIZE=384

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=CHANGE_THIS_PASSWORD
MINIO_BUCKET=legal-documents
MINIO_USE_SSL=false

# Security
JWT_SECRET=CHANGE_THIS_TO_RANDOM_256_BIT_SECRET
ENCRYPTION_KEY=CHANGE_THIS_TO_RANDOM_AES_256_KEY
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Performance
CACHE_TTL=300
MAX_FILE_SIZE=52428800
RATE_LIMIT_PER_MINUTE=100
WORKER_POOL_SIZE=10

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
LOG_LEVEL=info
HEALTH_CHECK_INTERVAL=30s

# AI Services
OLLAMA_URL=http://localhost:11434
CLAUDE_API_KEY=CHANGE_THIS_TO_YOUR_CLAUDE_KEY
OPENAI_API_KEY=CHANGE_THIS_TO_YOUR_OPENAI_KEY
AI_TIMEOUT=30s
AI_MAX_RETRIES=3
"@
    
    $prodEnvTemplate | Out-File -FilePath ".env.production.template" -Encoding UTF8
    Write-Host "✅ Production environment template: .env.production.template" -ForegroundColor Green
}

Write-Host ""
Write-Host "📚 Documentation generated successfully!" -ForegroundColor Green
Write-Host "📄 Main guide: $OutputPath" -ForegroundColor Cyan
if ($GenerateConfigTemplates) {
    Write-Host "📄 Config template: .env.production.template" -ForegroundColor Cyan
}
Write-Host ""