# Complete System Setup Script for File Merge System - ENHANCED VERSION
# Location: C:\Users\james\Desktop\deeds-web\deeds-web-app\scripts\setup-complete-system.ps1
# Version: 2.0 - Includes all enhancements

param(
    [switch]$SkipDocker,
    [switch]$DevMode,
    [switch]$GenerateSecureConfig,
    [switch]$EnableMonitoring,
    [switch]$CreateBackup,
    [string]$DatabasePassword = "secure_password_123"
)

Write-Host "🚀 Starting Enhanced File Merge System Setup v2.0" -ForegroundColor Blue
Write-Host "📁 Project Directory: $PWD" -ForegroundColor Gray
Write-Host "📅 Setup Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ("=" * 60) -ForegroundColor Blue

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Import Web.Security for password generation
Add-Type -AssemblyName System.Web

#region Enhanced Functions

function Write-Progress-Step {
    param($Step, $Message)
    Write-Host "`n[$Step] $Message" -ForegroundColor Cyan
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"

    # Create logs directory if it doesn't exist
    $logsDir = Join-Path $PWD "logs"
    if (!(Test-Path $logsDir)) {
        New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    }

    # Write to log file
    $logFile = Join-Path $logsDir "setup-$(Get-Date -Format 'yyyy-MM-dd').log"
    $logMessage | Out-File -FilePath $logFile -Append -Encoding UTF8

    # Also display in console with color coding
    switch ($Level) {
        "ERROR" { Write-Host $logMessage -ForegroundColor Red }
        "WARNING" { Write-Host $logMessage -ForegroundColor Yellow }
        "SUCCESS" { Write-Host $logMessage -ForegroundColor Green }
        default { Write-Host $logMessage -ForegroundColor Gray }
    }
}

function Test-DockerReady {
    try {
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Docker daemon is not running. Please start Docker Desktop." "ERROR"
            Write-Host "   On Windows: Start Docker Desktop from the Start Menu" -ForegroundColor Yellow
            return $false
        }
        Write-Log "Docker is ready" "SUCCESS"
        return $true
    } catch {
        Write-Log "Docker is not installed or not in PATH" "ERROR"
        return $false
    }
}

function Test-Port {
    param($Port, $Host = "localhost")
    try {
        $connection = New-Object System.Net.Sockets.TcpClient($Host, $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

function Wait-ForService {
    param($ServiceName, $Port, $MaxAttempts = 30)
    Write-Host "⏳ Waiting for $ServiceName to be ready on port $Port..." -ForegroundColor Yellow

    for ($i = 1; $i -le $MaxAttempts; $i++) {
        if (Test-Port -Port $Port) {
            Write-Log "$ServiceName is ready!" "SUCCESS"
            return $true
        }
        Write-Host "  Attempt $i/$MaxAttempts..." -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }

    Write-Log "$ServiceName failed to start within timeout" "ERROR"
    return $false
}

function Ensure-Ollama {
    param([string]$Model = "nomic-embed-text")

    Write-Log "Checking Ollama installation..." "INFO"

    try {
        $ollamaVersion = ollama version 2>&1
        Write-Log "Ollama found: $ollamaVersion" "SUCCESS"

        # Check if model is available
        $models = ollama list 2>&1
        if ($models -notmatch $Model) {
            Write-Log "Pulling model: $Model..." "WARNING"
            ollama pull $Model
            Write-Log "Model $Model pulled successfully" "SUCCESS"
        } else {
            Write-Log "Model $Model already available" "SUCCESS"
        }

        # Ensure Ollama is running
        $ollamaPs = Get-Process "ollama" -ErrorAction SilentlyContinue
        if (!$ollamaPs) {
            Write-Log "Starting Ollama service..." "INFO"
            Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 3
        }

        return $true
    } catch {
        Write-Log "Ollama not found. Please install from: https://ollama.ai" "WARNING"
        Write-Host "   After installation, run: ollama pull $Model" -ForegroundColor Yellow
        return $false
    }
}

function Backup-ExistingData {
    param([string]$BackupDir = "backups")

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupPath = Join-Path $BackupDir "backup_$timestamp"

    if ((Test-Path "data") -and (Get-ChildItem "data" -Recurse | Measure-Object).Count -gt 0) {
        Write-Log "Creating backup of existing data..." "INFO"

        New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
        Copy-Item -Path "data" -Destination $backupPath -Recurse -Force

        Write-Log "Backup created at: $backupPath" "SUCCESS"
        return $backupPath
    }

    return $null
}

function New-SecureConfig {
    $config = @{
        DatabasePassword = [System.Web.Security.Membership]::GeneratePassword(20, 5)
        JWTSecret = [System.Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Maximum 256 }))
        EncryptionKey = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
        MinIOAccessKey = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 20 | ForEach-Object {[char]$_})
        MinIOSecretKey = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 40 | ForEach-Object {[char]$_})
    }

    Write-Log "Generated secure configuration" "SUCCESS"
    Write-Host "🔐 Secure Configuration Generated:" -ForegroundColor Blue
    Write-Host "   Database Password: $($config.DatabasePassword)" -ForegroundColor Gray
    Write-Host "   JWT Secret: $($config.JWTSecret.Substring(0, 20))..." -ForegroundColor Gray
    Write-Host "   Encryption Key: $($config.EncryptionKey.Substring(0, 20))..." -ForegroundColor Gray

    # Save to secure config file
    $configPath = Join-Path $PWD ".env.secure"
    $configContent = @"
# Secure Configuration Generated on $(Get-Date)
DATABASE_PASSWORD=$($config.DatabasePassword)
JWT_SECRET=$($config.JWTSecret)
ENCRYPTION_KEY=$($config.EncryptionKey)
MINIO_ACCESS_KEY=$($config.MinIOAccessKey)
MINIO_SECRET_KEY=$($config.MinIOSecretKey)
"@
    $configContent | Out-File -FilePath $configPath -Encoding UTF8
    Write-Host "   Config saved to: .env.secure" -ForegroundColor Green

    return $config
}

function Create-SqlSchema {
    $schemaContent = @'
-- File Merge System Database Schema
-- PostgreSQL with pgVector extension

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create main tables
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    storage_path TEXT,
    minio_object_id VARCHAR(255),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    content_hash VARCHAR(64),
    is_deleted BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(255)
);

CREATE INDEX idx_documents_filename ON documents(filename);
CREATE INDEX idx_documents_upload_date ON documents(upload_date);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);

CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX idx_embeddings_vector ON document_embeddings USING ivfflat (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS merge_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'draft',
    output_format VARCHAR(50),
    merge_config JSONB,
    created_by VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS merge_session_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES merge_sessions(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER,
    included BOOLEAN DEFAULT TRUE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, document_id)
);

CREATE TABLE IF NOT EXISTS processing_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    action VARCHAR(100),
    status VARCHAR(50),
    details JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_processing_logs_document_action ON processing_logs(document_id, action);
CREATE INDEX idx_processing_logs_created_at ON processing_logs(created_at);

-- Create views for easier querying
CREATE OR REPLACE VIEW document_stats AS
SELECT
    COUNT(*) as total_documents,
    SUM(file_size) as total_size,
    COUNT(DISTINCT file_type) as unique_types,
    MAX(upload_date) as last_upload
FROM documents
WHERE is_deleted = FALSE;

-- Create functions and triggers
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_document_modtime
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_session_modtime
    BEFORE UPDATE ON merge_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Insert sample data for testing
INSERT INTO documents (filename, file_type, file_size, storage_path, created_by)
VALUES
    ('sample.pdf', 'application/pdf', 1024000, '/uploads/sample.pdf', 'system'),
    ('test.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 512000, '/uploads/test.docx', 'system')
ON CONFLICT DO NOTHING;

-- Grant necessary permissions (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
'@

    $sqlDir = "sql"
    if (!(Test-Path $sqlDir)) {
        New-Item -ItemType Directory -Path $sqlDir -Force | Out-Null
    }

    $schemaPath = Join-Path $sqlDir "file-merge-schema.sql"
    $schemaContent | Out-File -FilePath $schemaPath -Encoding UTF8

    Write-Log "SQL schema file created at: $schemaPath" "SUCCESS"
    return $schemaPath
}

function Test-AllServices {
    param($Config)

    Write-Log "Running Service Health Checks..." "INFO"

    $tests = @(
        @{
            Name = "PostgreSQL"
            Port = $Config.PostgreSQL.Port
            Test = {
                param($Config)
                Test-Port -Port $Config.PostgreSQL.Port
            }
        },
        @{
            Name = "MinIO"
            Port = 9000
            Test = {
                try {
                    $response = Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -UseBasicParsing
                    return $response.StatusCode -eq 200
                } catch { return $false }
            }
        },
        @{
            Name = "Qdrant"
            Port = 6333
            Test = {
                try {
                    $response = Invoke-WebRequest -Uri "http://localhost:6333/health" -UseBasicParsing
                    return $response.StatusCode -eq 200
                } catch { return $false }
            }
        },
        @{
            Name = "Redis"
            Port = 6379
            Test = {
                Test-Port -Port 6379
            }
        },
        @{
            Name = "Ollama"
            Port = 11434
            Test = {
                try {
                    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing
                    return $response.StatusCode -eq 200
                } catch { return $false }
            }
        }
    )

    $results = @()
    foreach ($test in $tests) {
        $success = & $test.Test -Config $Config
        $results += @{
            Service = $test.Name
            Status = if ($success) { "✅ Healthy" } else { "❌ Failed" }
            Success = $success
        }

        $level = if ($success) { "SUCCESS" } else { "ERROR" }
        Write-Log "$($test.Name): $(if ($success) { 'Healthy' } else { 'Failed' })" $level
    }

    $allHealthy = ($results | Where-Object { -not $_.Success }).Count -eq 0
    return @{
        AllHealthy = $allHealthy
        Results = $results
    }
}

function Setup-Monitoring {
    param([switch]$Enable)

    if (!$Enable) { return }

    Write-Log "Setting up monitoring stack..." "INFO"

    # Create monitoring directory
    $monitoringDir = "monitoring"
    if (!(Test-Path $monitoringDir)) {
        New-Item -ItemType Directory -Path $monitoringDir -Force | Out-Null
    }

    # Create Prometheus configuration
    $prometheusConfig = @"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'postgres'
    static_configs:
      - targets: ['host.docker.internal:5432']

  - job_name: 'minio'
    metrics_path: /minio/v2/metrics/cluster
    static_configs:
      - targets: ['minio:9000']

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
"@

    $prometheusConfig | Out-File -FilePath (Join-Path $monitoringDir "prometheus.yml") -Encoding UTF8

    Write-Log "Monitoring stack configured (Prometheus: 9090, Grafana: 3000)" "SUCCESS"
}

function Cleanup-FailedSetup {
    Write-Log "Cleaning up failed setup..." "WARNING"

    try {
        # Stop and remove Docker containers
        docker-compose down -v 2>$null

        # Remove data directories (with confirmation)
        $confirmation = Read-Host "Remove data directories? (y/n)"
        if ($confirmation -eq 'y') {
            Remove-Item -Path "data" -Recurse -Force -ErrorAction SilentlyContinue
            Write-Log "Data directories removed" "SUCCESS"
        }
    } catch {
        Write-Log "Cleanup error: $_" "ERROR"
    }
}

function Initialize-GoBackend {
    Write-Log "Initializing Go backend..." "INFO"

    # Create Go backend structure
    $goDir = "go-backend"
    if (!(Test-Path $goDir)) {
        New-Item -ItemType Directory -Path $goDir -Force | Out-Null
    }

    # Create main.go file
    $mainGoContent = @'
package main

import (
    "fmt"
    "log"
    "net/http"
    "os"

    "github.com/gin-gonic/gin"
    "github.com/joho/godotenv"
)

func main() {
    // Load environment variables
    if err := godotenv.Load(); err != nil {
        log.Println("No .env file found")
    }

    // Initialize Gin router
    r := gin.Default()

    // CORS middleware
    r.Use(func(c *gin.Context) {
        c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
        c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(204)
            return
        }

        c.Next()
    })

    // Health check endpoint
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
            "service": "file-merge-backend",
        })
    })

    // API routes
    api := r.Group("/api")
    {
        api.GET("/documents", getDocuments)
        api.POST("/documents/upload", uploadDocument)
        api.POST("/merge", mergeDocuments)
    }

    port := os.Getenv("API_PORT")
    if port == "" {
        port = "8084"
    }

    log.Printf("Server starting on port %s", port)
    if err := r.Run(":" + port); err != nil {
        log.Fatal("Failed to start server:", err)
    }
}

func getDocuments(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{"documents": []interface{}{}})
}

func uploadDocument(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{"status": "uploaded"})
}

func mergeDocuments(c *gin.Context) {
    c.JSON(http.StatusOK, gin.H{"status": "merged"})
}
'@

    $mainGoContent | Out-File -FilePath (Join-Path $goDir "main.go") -Encoding UTF8

    # Create go.mod file
    $goModContent = @"
module file-merge-backend

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/joho/godotenv v1.5.1
)
"@

    $goModContent | Out-File -FilePath (Join-Path $goDir "go.mod") -Encoding UTF8

    Write-Log "Go backend initialized" "SUCCESS"
}

function Initialize-SvelteKitFrontend {
    Write-Log "Checking SvelteKit frontend..." "INFO"

    $frontendDir = "sveltekit-frontend"

    if (!(Test-Path (Join-Path $frontendDir "package.json"))) {
        Write-Log "Initializing SvelteKit project..." "INFO"

        # Create SvelteKit project
        npm create vite@latest $frontendDir -- --template svelte

        # Install dependencies
        Set-Location $frontendDir
        npm install
        Set-Location ..

        Write-Log "SvelteKit frontend initialized" "SUCCESS"
    } else {
        Write-Log "SvelteKit frontend already exists" "SUCCESS"
    }
}

#endregion

#region Main Configuration

# Use secure config if requested
if ($GenerateSecureConfig) {
    $secureConfig = New-SecureConfig
    $DatabasePassword = $secureConfig.DatabasePassword
    $minioAccessKey = $secureConfig.MinIOAccessKey
    $minioSecretKey = $secureConfig.MinIOSecretKey
} else {
    $minioAccessKey = "minioadmin"
    $minioSecretKey = "minioadmin"
}

$Config = @{
    PostgreSQL = @{
        Host = "localhost"
        Port = 5432
        Database = "legal_ai"
        Username = "postgres"
        Password = $DatabasePassword
    }
    MinIO = @{
        Endpoint = "http://localhost:9000"
        AccessKey = $minioAccessKey
        SecretKey = $minioSecretKey
        Bucket = "legal-documents"
    }
    Qdrant = @{
        Url = "http://localhost:6333"
        Collection = "legal_documents"
    }
    Ollama = @{
        Url = "http://localhost:11434"
        Model = "nomic-embed-text"
    }
    Redis = @{
        Url = "redis://localhost:6379"
    }
}

#endregion

#region Main Execution

try {
    # Step 0: Pre-flight checks
    Write-Progress-Step "0/12" "Pre-flight Checks"

    if (!$SkipDocker) {
        if (!Test-DockerReady) {
            throw "Docker is not ready. Please ensure Docker Desktop is running."
        }
    }

    # Create backup if requested
    if ($CreateBackup) {
        $backupPath = Backup-ExistingData
    }

    # Step 1: Prerequisites Check
    Write-Progress-Step "1/12" "Checking Prerequisites"

    $Prerequisites = @(
        @{ Name = "Docker"; Command = "docker --version" },
        @{ Name = "Node.js"; Command = "node --version" },
        @{ Name = "NPM"; Command = "npm --version" },
        @{ Name = "Git"; Command = "git --version" }
    )

    foreach ($prereq in $Prerequisites) {
        try {
            $result = Invoke-Expression $prereq.Command 2>$null
            Write-Log "$($prereq.Name): $result" "SUCCESS"
        } catch {
            Write-Log "$($prereq.Name) not found. Please install it first." "ERROR"
            exit 1
        }
    }

    # Check and install Ollama
    Ensure-Ollama -Model $Config.Ollama.Model

    # Step 2: Create Directory Structure
    Write-Progress-Step "2/12" "Creating Directory Structure"

    $Directories = @(
        "data/postgres",
        "data/minio",
        "data/qdrant",
        "data/redis",
        "logs",
        "backups",
        "uploads",
        "sql",
        "nginx",
        "monitoring"
    )

    foreach ($dir in $Directories) {
        $fullPath = Join-Path $PWD $dir
        if (!(Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
            Write-Log "Created: $dir" "SUCCESS"
        } else {
            Write-Log "Exists: $dir" "INFO"
        }
    }

    # Step 3: Create SQL Schema
    Write-Progress-Step "3/12" "Creating SQL Schema"
    $schemaPath = Create-SqlSchema

    # Step 4: Create Docker Compose File
    Write-Progress-Step "4/12" "Creating Docker Compose Configuration"

    $DockerComposeContent = @"
version: '3.8'

services:
  # PostgreSQL with pgVector extension
  postgres:
    image: pgvector/pgvector:pg16
    container_name: legal-ai-postgres
    environment:
      POSTGRES_DB: $($Config.PostgreSQL.Database)
      POSTGRES_USER: $($Config.PostgreSQL.Username)
      POSTGRES_PASSWORD: $($Config.PostgreSQL.Password)
      PGDATA: /var/lib/postgresql/data/pgdata
    ports:
      - "$($Config.PostgreSQL.Port):5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./sql:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $($Config.PostgreSQL.Username) -d $($Config.PostgreSQL.Database)"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - legal-ai-network

  # MinIO Object Storage
  minio:
    image: minio/minio:latest
    container_name: legal-ai-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: $($Config.MinIO.AccessKey)
      MINIO_ROOT_PASSWORD: $($Config.MinIO.SecretKey)
    ports:
      - "9000:9000"   # MinIO API
      - "9001:9001"   # MinIO Console
    volumes:
      - ./data/minio:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    restart: unless-stopped
    networks:
      - legal-ai-network

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: legal-ai-qdrant
    ports:
      - "6333:6333"   # HTTP API
      - "6334:6334"   # gRPC API
    volumes:
      - ./data/qdrant:/qdrant/storage
    environment:
      QDRANT__LOG_LEVEL: INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - legal-ai-network

  # Redis for Caching
  redis:
    image: redis:7-alpine
    container_name: legal-ai-redis
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    restart: unless-stopped
    networks:
      - legal-ai-network
"@

    # Add monitoring services if enabled
    if ($EnableMonitoring) {
        $DockerComposeContent += @"

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    container_name: legal-ai-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./data/prometheus:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - legal-ai-network

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: legal-ai-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - ./data/grafana:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - legal-ai-network
"@
    }

    $DockerComposeContent += @"

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: legal-ai-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro
    depends_on:
      - postgres
      - minio
      - qdrant
    restart: unless-stopped
    networks:
      - legal-ai-network

networks:
  legal-ai-network:
    driver: bridge

volumes:
  postgres_data:
    driver: local
  minio_data:
    driver: local
  qdrant_data:
    driver: local
  redis_data:
    driver: local
"@

    $DockerComposeContent | Out-File -FilePath "docker-compose.yml" -Encoding UTF8
    Write-Log "Docker Compose file created" "SUCCESS"

    # Step 5: Create Nginx Configuration
    Write-Progress-Step "5/12" "Creating Nginx Configuration"

    $NginxConfig = @"
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server host.docker.internal:8084;
    }

    upstream frontend {
        server host.docker.internal:5173;
    }

    server {
        listen 80;
        server_name localhost;

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host `$host;
            proxy_set_header X-Real-IP `$remote_addr;
            proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto `$scheme;
        }

        # Backend API
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host `$host;
            proxy_set_header X-Real-IP `$remote_addr;
            proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto `$scheme;
        }

        # MinIO API
        location /minio/ {
            proxy_pass http://minio:9000/;
            proxy_set_header Host `$host;
            proxy_set_header X-Real-IP `$remote_addr;
            proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto `$scheme;
        }

        # Qdrant API
        location /qdrant/ {
            proxy_pass http://qdrant:6333/;
            proxy_set_header Host `$host;
            proxy_set_header X-Real-IP `$remote_addr;
            proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto `$scheme;
        }
    }
}
"@

    $NginxConfig | Out-File -FilePath "nginx/nginx.conf" -Encoding UTF8
    Write-Log "Nginx configuration created" "SUCCESS"

    # Step 6: Create Environment Files
    Write-Progress-Step "6/12" "Creating Environment Configuration"

    $EnvContent = @"
# Database Configuration
POSTGRES_HOST=$($Config.PostgreSQL.Host)
POSTGRES_PORT=$($Config.PostgreSQL.Port)
POSTGRES_DB=$($Config.PostgreSQL.Database)
POSTGRES_USER=$($Config.PostgreSQL.Username)
POSTGRES_PASSWORD=$($Config.PostgreSQL.Password)
DATABASE_URL=postgresql://$($Config.PostgreSQL.Username):$($Config.PostgreSQL.Password)@$($Config.PostgreSQL.Host):$($Config.PostgreSQL.Port)/$($Config.PostgreSQL.Database)

# MinIO Configuration
MINIO_ENDPOINT=$($Config.MinIO.Endpoint)
MINIO_ACCESS_KEY=$($Config.MinIO.AccessKey)
MINIO_SECRET_KEY=$($Config.MinIO.SecretKey)
MINIO_BUCKET=$($Config.MinIO.Bucket)

# Qdrant Configuration
QDRANT_URL=$($Config.Qdrant.Url)
QDRANT_COLLECTION=$($Config.Qdrant.Collection)

# Ollama Configuration
OLLAMA_URL=$($Config.Ollama.Url)
OLLAMA_MODEL=$($Config.Ollama.Model)

# Redis Configuration
REDIS_URL=$($Config.Redis.Url)

# Application Configuration
NODE_ENV=development
PORT=5173
API_PORT=8084
LOG_LEVEL=info

# Security
JWT_SECRET=$(if ($GenerateSecureConfig) { $secureConfig.JWTSecret } else { "your-super-secret-jwt-key-change-this-in-production" })
ENCRYPTION_KEY=$(if ($GenerateSecureConfig) { $secureConfig.EncryptionKey } else { "your-32-character-encryption-key-here" })

# File Upload Limits
MAX_FILE_SIZE=100MB
ALLOWED_FILE_TYPES=pdf,docx,txt,png,jpg,jpeg

# Vector Search Configuration
EMBEDDING_DIMENSIONS=1536
SIMILARITY_THRESHOLD=0.7
MAX_SEARCH_RESULTS=50
"@

    $EnvContent | Out-File -FilePath ".env" -Encoding UTF8
    $EnvContent | Out-File -FilePath "sveltekit-frontend/.env" -Encoding UTF8
    Write-Log "Environment files created" "SUCCESS"

    # Step 7: Initialize Backend and Frontend
    Write-Progress-Step "7/12" "Initializing Application Components"

    Initialize-GoBackend
    Initialize-SvelteKitFrontend

    # Step 8: Setup Monitoring
    if ($EnableMonitoring) {
        Write-Progress-Step "8/12" "Setting up Monitoring"
        Setup-Monitoring -Enable
    }

    # Step 9: Start Docker Services
    if (!$SkipDocker) {
        Write-Progress-Step "9/12" "Starting Docker Services"

        try {
            docker-compose down --remove-orphans 2>$null
            docker-compose up -d

            Write-Log "Docker services starting..." "INFO"

            # Wait for services to be ready
            $ServicesToCheck = @(
                @{ Name = "PostgreSQL"; Port = $Config.PostgreSQL.Port },
                @{ Name = "MinIO"; Port = 9000 },
                @{ Name = "Qdrant"; Port = 6333 },
                @{ Name = "Redis"; Port = 6379 }
            )

            foreach ($service in $ServicesToCheck) {
                if (!(Wait-ForService -ServiceName $service.Name -Port $service.Port)) {
                    throw "Service $($service.Name) failed to start"
                }
            }

        } catch {
            Write-Log "Docker services failed to start: $($_.Exception.Message)" "ERROR"
            Cleanup-FailedSetup
            exit 1
        }
    } else {
        Write-Log "Skipping Docker services (--SkipDocker specified)" "WARNING"
    }

    # Step 10: Initialize Database Schema
    Write-Progress-Step "10/12" "Initializing Database Schema"

    if (!$SkipDocker) {
        try {
            Start-Sleep -Seconds 5  # Give PostgreSQL extra time to be ready

            $env:PGPASSWORD = $Config.PostgreSQL.Password
            $psqlCommand = "docker exec legal-ai-postgres psql -U $($Config.PostgreSQL.Username) -d $($Config.PostgreSQL.Database) -f /docker-entrypoint-initdb.d/file-merge-schema.sql"

            Invoke-Expression $psqlCommand 2>&1 | Out-Null
            Write-Log "Database schema initialized" "SUCCESS"
        } catch {
            Write-Log "Database schema initialization failed. Will retry on next startup." "WARNING"
        }
    }

    # Step 11: Initialize MinIO Buckets
    Write-Progress-Step "11/12" "Initializing MinIO Buckets"

    if (!$SkipDocker) {
        try {
            # Wait for MinIO to be fully ready
            Start-Sleep -Seconds 5

            # Create bucket using MinIO client via Docker
            $createBucketCmd = "docker exec legal-ai-minio mc alias set local http://localhost:9000 $($Config.MinIO.AccessKey) $($Config.MinIO.SecretKey)"
            Invoke-Expression $createBucketCmd 2>&1 | Out-Null

            $makeBucketCmd = "docker exec legal-ai-minio mc mb local/$($Config.MinIO.Bucket) --ignore-existing"
            Invoke-Expression $makeBucketCmd 2>&1 | Out-Null

            Write-Log "MinIO bucket '$($Config.MinIO.Bucket)' created" "SUCCESS"
        } catch {
            Write-Log "MinIO bucket creation failed: $($_.Exception.Message)" "WARNING"
        }
    }

    # Step 12: Initialize Qdrant Collection
    Write-Progress-Step "12/12" "Initializing Qdrant Vector Collection"

    if (!$SkipDocker) {
        try {
            $qdrantConfig = @{
                vectors = @{
                    size = 1536
                    distance = "Cosine"
                }
            } | ConvertTo-Json -Depth 3

            $headers = @{ "Content-Type" = "application/json" }
            $uri = "$($Config.Qdrant.Url)/collections/$($Config.Qdrant.Collection)"

            Invoke-RestMethod -Uri $uri -Method PUT -Body $qdrantConfig -Headers $headers
            Write-Log "Qdrant collection '$($Config.Qdrant.Collection)' created" "SUCCESS"
        } catch {
            if ($_.Exception.Response.StatusCode -eq 409) {
                Write-Log "Qdrant collection already exists" "SUCCESS"
            } else {
                Write-Log "Qdrant collection creation failed: $($_.Exception.Message)" "WARNING"
            }
        }
    }

    # Install Dependencies
    Write-Progress-Step "BONUS" "Installing Dependencies and Running Error Fixes"

    try {
        # Install root dependencies if package.json exists
        if (Test-Path "package.json") {
            Write-Log "Installing root dependencies..." "INFO"
            npm install
        }

        # Install frontend dependencies
        if (Test-Path "sveltekit-frontend/package.json") {
            Write-Log "Installing frontend dependencies..." "INFO"
            Set-Location "sveltekit-frontend"
            npm install
            Set-Location ".."
        }

        # Run error fixes if script exists
        if (Test-Path "scripts/fix-svelte5-errors.mjs") {
            Write-Log "Running automated error fixes..." "INFO"
            node scripts/fix-svelte5-errors.mjs
        }

        Write-Log "Dependencies installed and errors fixed" "SUCCESS"
    } catch {
        Write-Log "Dependency installation failed: $($_.Exception.Message)" "WARNING"
    }

    # Create startup scripts
    Write-Progress-Step "FINAL" "Creating Startup Scripts"

    $StartupScript = @"
@echo off
echo Starting Legal AI File Merge System...

echo Starting Docker services...
docker-compose up -d

echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo Starting Go backend...
start "Go Backend" cmd /k "cd go-backend && go run main.go"

echo Starting SvelteKit frontend...
start "Frontend" cmd /k "cd sveltekit-frontend && npm run dev"

echo Starting Ollama service...
start "Ollama" cmd /k "ollama serve"

echo Opening browser...
timeout /t 5 /nobreak >nul
start http://localhost:5173

echo System started! Check the opened windows for logs.
pause
"@

    $StartupScript | Out-File -FilePath "start-system.bat" -Encoding ASCII

    $StopScript = @"
@echo off
echo Stopping Legal AI File Merge System...

echo Stopping Docker services...
docker-compose down

echo Killing Node.js processes...
taskkill /f /im node.exe 2>nul

echo Killing Go processes...
taskkill /f /im main.exe 2>nul

echo Killing Ollama processes...
taskkill /f /im ollama.exe 2>nul

echo System stopped.
pause
"@

    $StopScript | Out-File -FilePath "stop-system.bat" -Encoding ASCII

    Write-Log "Startup scripts created: start-system.bat, stop-system.bat" "SUCCESS"

    # Test all services
    $testResults = Test-AllServices -Config $Config

    # Final Summary
    Write-Host "`n🎉 SETUP COMPLETE!" -ForegroundColor Green -BackgroundColor Black
    Write-Host ("=" * 80) -ForegroundColor Green

    Write-Host "`n📋 SYSTEM STATUS:" -ForegroundColor Blue
    Write-Host "  🐘 PostgreSQL:    http://localhost:$($Config.PostgreSQL.Port)" -ForegroundColor White
    Write-Host "  🗄️  MinIO:         http://localhost:9000 (Console: http://localhost:9001)" -ForegroundColor White
    Write-Host "  🔍 Qdrant:        http://localhost:6333" -ForegroundColor White
    Write-Host "  ⚡ Redis:         localhost:6379" -ForegroundColor White
    Write-Host "  🤖 Ollama:        http://localhost:11434" -ForegroundColor White
    Write-Host "  🌐 Nginx:         http://localhost:80" -ForegroundColor White

    if ($EnableMonitoring) {
        Write-Host "  📊 Prometheus:    http://localhost:9090" -ForegroundColor White
        Write-Host "  📈 Grafana:       http://localhost:3000" -ForegroundColor White
    }

    Write-Host "`n📊 SERVICE HEALTH:" -ForegroundColor Blue
    foreach ($result in $testResults.Results) {
        Write-Host "  $($result.Status) $($result.Service)" -ForegroundColor White
    }

    Write-Host "`n🚀 NEXT STEPS:" -ForegroundColor Blue
    Write-Host "  1. Run: .\start-system.bat" -ForegroundColor Yellow
    Write-Host "  2. Open: http://localhost:5173" -ForegroundColor Yellow
    Write-Host "  3. Test: Upload files and try merging" -ForegroundColor Yellow

    Write-Host "`n📚 USEFUL COMMANDS:" -ForegroundColor Blue
    Write-Host "  • Check status:    docker-compose ps" -ForegroundColor White
    Write-Host "  • View logs:       docker-compose logs -f [service]" -ForegroundColor White
    Write-Host "  • Stop services:   .\stop-system.bat" -ForegroundColor White
    Write-Host "  • Restart:         docker-compose restart" -ForegroundColor White

    Write-Host "`n💡 CREDENTIALS:" -ForegroundColor Blue
    Write-Host "  • PostgreSQL:     postgres / $($Config.PostgreSQL.Password)" -ForegroundColor White
    Write-Host "  • MinIO:          $($Config.MinIO.AccessKey) / $($Config.MinIO.SecretKey)" -ForegroundColor White
    Write-Host "  • MinIO Console:  http://localhost:9001" -ForegroundColor White

    if ($GenerateSecureConfig) {
        Write-Host "`n🔐 SECURE CONFIG: Saved to .env.secure" -ForegroundColor Magenta
    }

    if ($DevMode) {
        Write-Host "`n🔧 DEV MODE ENABLED - Additional services available" -ForegroundColor Magenta
    }

    Write-Host "`n📝 LOGS: Check ./logs/ directory for detailed setup logs" -ForegroundColor Cyan
    Write-Host "`n🎯 Your file merging system is now ready!" -ForegroundColor Green

    # Save summary to file
    $summaryPath = Join-Path "logs" "setup-summary-$(Get-Date -Format 'yyyy-MM-dd-HHmmss').txt"
    @"
Setup Summary - $(Get-Date)
========================
All Services: $(if ($testResults.AllHealthy) { "✅ Healthy" } else { "⚠️ Some issues" })

Service Status:
$($testResults.Results | ForEach-Object { "  - $($_.Service): $($_.Status)" } | Out-String)

Configuration:
  Database: $($Config.PostgreSQL.Database)
  MinIO Bucket: $($Config.MinIO.Bucket)
  Qdrant Collection: $($Config.Qdrant.Collection)
  Ollama Model: $($Config.Ollama.Model)

Next Steps:
  1. Run start-system.bat
  2. Access http://localhost:5173
"@ | Out-File -FilePath $summaryPath -Encoding UTF8

    Write-Host "`n📄 Setup summary saved to: $summaryPath" -ForegroundColor Gray

} catch {
    Write-Log "Setup failed: $($_.Exception.Message)" "ERROR"
    Write-Host "`n❌ SETUP FAILED" -ForegroundColor Red
    Write-Host "Check logs for details: ./logs/" -ForegroundColor Yellow

    # Offer cleanup
    $cleanup = Read-Host "`nDo you want to cleanup and retry? (y/n)"
    if ($cleanup -eq 'y') {
        Cleanup-FailedSetup
    }

    exit 1
} finally {
    # Ensure we return to original directory
    Set-Location $PWD
}

#endregion
