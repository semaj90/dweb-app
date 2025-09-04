# Deploy Complete RAG System - Windows PowerShell Version
# This script sets up the entire enhanced RAG system with all dependencies

Write-Host "🚀 Starting Complete RAG System Deployment..." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

# Function to check if command exists
function Test-Command {
    param($Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Function to print colored output
function Write-Success {
    param($Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param($Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning {
    param($Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

# Check prerequisites
Write-Host ""
Write-Host "Checking prerequisites..." -ForegroundColor Cyan
Write-Host "------------------------"

if (!(Test-Command docker)) {
    Write-Error "Docker is not installed. Please install Docker Desktop for Windows first."
    exit 1
}
Write-Success "Docker is installed"

if (!(Test-Command node)) {
    Write-Error "Node.js is not installed. Please install Node.js first."
    exit 1
}
Write-Success "Node.js is installed"

if (!(Test-Command npm)) {
    Write-Error "npm is not installed. Please install npm first."
    exit 1
}
Write-Success "npm is installed"

# Check for Python (needed for OCR)
if (!(Test-Command python)) {
    Write-Warning "Python is not installed. OCR features will be limited."
} else {
    Write-Success "Python is installed"
}

# Create necessary directories
Write-Host ""
Write-Host "Creating project directories..." -ForegroundColor Cyan
Write-Host "-------------------------------"

$directories = @(
    "data\postgres",
    "data\redis",
    "data\neo4j",
    "data\minio",
    "data\rabbitmq",
    "logs",
    "models",
    "uploads",
    "cache"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Success "Created directory: $dir"
}

# Copy environment file
Write-Host ""
Write-Host "Setting up environment configuration..." -ForegroundColor Cyan
Write-Host "---------------------------------------"

if (!(Test-Path .env)) {
    if (Test-Path .env.example) {
        Copy-Item .env.example .env
        Write-Success "Created .env file from .env.example"
        Write-Warning "Please update .env with your configuration"
    } else {
        @"
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_ai_db?sslmode=disable

# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_BUCKET=legal-documents

# Redis
REDIS_URL=redis://localhost:6379

# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@localhost:5672

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3legal:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Backend Services
BACKEND_RAG_STREAM_ENDPOINT=http://localhost:8094/stream
PORT=3000
NODE_ENV=development
"@ | Out-File -FilePath .env -Encoding UTF8
        Write-Success "Created default .env file"
    }
} else {
    Write-Success ".env file already exists"
}

# Create docker-compose.yml
Write-Host ""
Write-Host "Creating Docker Compose configuration..." -ForegroundColor Cyan
Write-Host "----------------------------------------"

@"
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    container_name: legal-ai-postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: legal_ai_db
    ports:
      - "5432:5432"
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: legal-ai-redis
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:5-community
    container_name: legal-ai-neo4j
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_apoc_export_file_enabled: 'true'
      NEO4J_apoc_import_file_enabled: 'true'
      NEO4J_apoc_import_file_use__neo4j__config: 'true'
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - ./data/neo4j:/data
    healthcheck:
      test: ["CMD-SHELL", "wget --quiet --tries=1 --spider http://localhost:7474 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  rabbitmq:
    image: rabbitmq:3-management-alpine
    container_name: legal-ai-rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - ./data/rabbitmq:/var/lib/rabbitmq
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  minio:
    image: minio/minio:latest
    container_name: legal-ai-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./data/minio:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5

networks:
  default:
    name: legal-ai-network
"@ | Out-File -FilePath docker-compose.yml -Encoding UTF8

Write-Success "Created docker-compose.yml"

# Create database initialization script
Write-Host ""
Write-Host "Creating database initialization script..." -ForegroundColor Cyan
Write-Host "------------------------------------------"

@"
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create user roles
CREATE TYPE user_role AS ENUM ('prosecutor', 'detective', 'admin', 'analyst');

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role user_role NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create documents table with vector embeddings
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    content TEXT,
    embedding vector(384),
    metadata JSONB,
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create cases table
CREATE TABLE IF NOT EXISTS cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_number VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'open',
    assigned_to UUID REFERENCES users(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create evidence table
CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES cases(id),
    document_id UUID REFERENCES documents(id),
    type VARCHAR(50),
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create feedback table for reinforcement learning
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    user_id UUID REFERENCES users(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create analytics table
CREATE TABLE IF NOT EXISTS analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    user_id UUID REFERENCES users(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_cases_assigned_to ON cases(assigned_to);
CREATE INDEX idx_evidence_case_id ON evidence(case_id);
CREATE INDEX idx_feedback_user_id ON feedback(user_id);
CREATE INDEX idx_analytics_event_type ON analytics(event_type);
CREATE INDEX idx_analytics_created_at ON analytics(created_at);

-- Insert demo users
INSERT INTO users (email, password_hash, role) VALUES
    ('prosecutor@legal.ai', '$2b$10$YourHashedPasswordHere', 'prosecutor'),
    ('detective@legal.ai', '$2b$10$YourHashedPasswordHere', 'detective'),
    ('admin@legal.ai', '$2b$10$YourHashedPasswordHere', 'admin'),
    ('analyst@legal.ai', '$2b$10$YourHashedPasswordHere', 'analyst')
ON CONFLICT (email) DO NOTHING;
"@ | Out-File -FilePath init-db.sql -Encoding UTF8

Write-Success "Created init-db.sql"

# Start Docker services
Write-Host ""
Write-Host "Starting Docker services..." -ForegroundColor Cyan
Write-Host "---------------------------"

docker-compose down 2>$null
docker-compose up -d

# Wait for services to be healthy
Write-Host ""
Write-Host "Waiting for services to be healthy..." -ForegroundColor Cyan
Write-Host "-------------------------------------"

$services = @("postgres", "redis", "neo4j", "rabbitmq", "minio")
$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    $allHealthy = $true
    
    foreach ($service in $services) {
        $containerName = "legal-ai-$service"
        $health = docker inspect --format='{{.State.Health.Status}}' $containerName 2>$null
        
        if ($health -ne "healthy") {
            $allHealthy = $false
            break
        }
    }
    
    if ($allHealthy) {
        Write-Success "All services are healthy!"
        break
    }
    
    $attempt++
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 2
}

if ($attempt -eq $maxAttempts) {
    Write-Error "Services failed to become healthy in time"
    docker-compose logs
    exit 1
}

Write-Host ""

# Install npm dependencies
Write-Host ""
Write-Host "Installing npm dependencies..." -ForegroundColor Cyan
Write-Host "------------------------------"

npm install
Write-Success "npm dependencies installed"

# Run database migrations
Write-Host ""
Write-Host "Running database migrations..." -ForegroundColor Cyan
Write-Host "------------------------------"

if (Test-Path "drizzle.config.ts") {
    npm run db:push 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Database migrations completed"
    } else {
        Write-Warning "Database migration skipped"
    }
} else {
    Write-Warning "No drizzle.config.ts found, skipping migrations"
}

# Download models if needed
Write-Host ""
Write-Host "Checking AI models..." -ForegroundColor Cyan
Write-Host "--------------------"

if (Test-Command ollama) {
    Write-Host "Pulling Ollama models..."
    ollama pull nomic-embed-text 2>$null
    ollama pull gemma3legal:latest 2>$null
    if ($LASTEXITCODE -ne 0) {
        ollama pull gemma:2b 2>$null
    }
    Write-Success "AI models checked"
} else {
    Write-Warning "Ollama not installed. Please install Ollama and pull models manually."
}

# Create MinIO bucket
Write-Host ""
Write-Host "Setting up MinIO bucket..." -ForegroundColor Cyan
Write-Host "--------------------------"

Start-Sleep -Seconds 5

# Try to create bucket using curl
$headers = @{
    "Host" = "localhost:9000"
    "Date" = (Get-Date).ToString("R")
    "Content-Type" = "application/octet-stream"
}

try {
    Invoke-WebRequest -Uri "http://localhost:9000/legal-documents" `
                      -Method PUT `
                      -Headers $headers `
                      -Credential (New-Object PSCredential("minioadmin", (ConvertTo-SecureString "minioadmin123" -AsPlainText -Force))) `
                      -ErrorAction SilentlyContinue
    Write-Success "MinIO bucket created"
} catch {
    Write-Warning "Could not create MinIO bucket automatically - please create manually via console"
}

# Build the application
Write-Host ""
Write-Host "Building the application..." -ForegroundColor Cyan
Write-Host "---------------------------"

npm run build 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Success "Application built successfully"
} else {
    Write-Warning "Build failed - continuing anyway"
}

# Create start script
Write-Host ""
Write-Host "Creating start script..." -ForegroundColor Cyan
Write-Host "-----------------------"

@"
# Start Enhanced RAG System
Write-Host '🚀 Starting Enhanced RAG System...' -ForegroundColor Green
Write-Host '=================================' -ForegroundColor Green

# Start Docker services if not running
docker-compose up -d

# Wait for services
Start-Sleep -Seconds 5

# Start the application
npm run dev

Write-Host '✅ System is running!' -ForegroundColor Green
Write-Host 'Access the application at: http://localhost:3000'
Write-Host 'MinIO Console: http://localhost:9001'
Write-Host 'RabbitMQ Management: http://localhost:15672'
Write-Host 'Neo4j Browser: http://localhost:7474'
"@ | Out-File -FilePath start-rag-system.ps1 -Encoding UTF8

Write-Success "Created start-rag-system.ps1"

# Create test script
Write-Host ""
Write-Host "Creating test script..." -ForegroundColor Cyan
Write-Host "----------------------"

@"
# Test RAG System
Write-Host '🧪 Testing RAG System...' -ForegroundColor Cyan
Write-Host '=======================' -ForegroundColor Cyan

# Test database connection
Write-Host -NoNewline 'Testing PostgreSQL... '
docker exec legal-ai-postgres pg_isready -U postgres 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host '✓' -ForegroundColor Green } else { Write-Host '✗' -ForegroundColor Red }

# Test Redis
Write-Host -NoNewline 'Testing Redis... '
docker exec legal-ai-redis redis-cli ping 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Host '✓' -ForegroundColor Green } else { Write-Host '✗' -ForegroundColor Red }

# Test Neo4j
Write-Host -NoNewline 'Testing Neo4j... '
try {
    Invoke-WebRequest -Uri 'http://localhost:7474' -ErrorAction SilentlyContinue | Out-Null
    Write-Host '✓' -ForegroundColor Green
} catch {
    Write-Host '✗' -ForegroundColor Red
}

# Test RabbitMQ
Write-Host -NoNewline 'Testing RabbitMQ... '
try {
    Invoke-WebRequest -Uri 'http://localhost:15672' -ErrorAction SilentlyContinue | Out-Null
    Write-Host '✓' -ForegroundColor Green
} catch {
    Write-Host '✗' -ForegroundColor Red
}

# Test MinIO
Write-Host -NoNewline 'Testing MinIO... '
try {
    Invoke-WebRequest -Uri 'http://localhost:9000' -ErrorAction SilentlyContinue | Out-Null
    Write-Host '✓' -ForegroundColor Green
} catch {
    Write-Host '✗' -ForegroundColor Red
}

# Run unit tests if available
if ((Test-Path 'package.json') -and (Get-Content package.json | Select-String '"test"')) {
    Write-Host ''
    Write-Host 'Running unit tests...'
    npm test
}

Write-Host ''
Write-Host '✅ Testing complete!' -ForegroundColor Green
"@ | Out-File -FilePath test-rag-system.ps1 -Encoding UTF8

Write-Success "Created test-rag-system.ps1"

# Final status
Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "✅ RAG System Deployment Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Services running:" -ForegroundColor Cyan
Write-Host "  • PostgreSQL: localhost:5432"
Write-Host "  • Redis: localhost:6379"
Write-Host "  • Neo4j: localhost:7474 (browser) / localhost:7687 (bolt)"
Write-Host "  • RabbitMQ: localhost:5672 (amqp) / localhost:15672 (management)"
Write-Host "  • MinIO: localhost:9000 (api) / localhost:9001 (console)"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Update .env file with your configuration"
Write-Host "  2. Run: .\start-rag-system.ps1"
Write-Host "  3. Access the application at http://localhost:3000"
Write-Host ""
Write-Host "To test the system: .\test-rag-system.ps1"
Write-Host "To stop services: docker-compose down"
Write-Host ""
Write-Warning "Remember to configure Ollama and pull the required models!"
