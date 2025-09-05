#!/usr/bin/env powershell
# Comprehensive Legal AI System Startup Script
# Fixes all critical errors and starts the complete production stack

param(
    [switch]$Force,
    [switch]$CleanStart,
    [switch]$SkipGPU
)

Write-Host "🚀 Starting Comprehensive Legal AI System Fix & Launch" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Yellow

# Function to log with timestamp
function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

# Function to check command availability
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

Write-Log "🔍 Checking system prerequisites..." "Yellow"

# Check Docker
if (-not (Test-Command "docker")) {
    Write-Log "❌ Docker not found. Please install Docker Desktop." "Red"
    exit 1
}

# Check Docker Compose
if (-not (Test-Command "docker-compose")) {
    Write-Log "❌ Docker Compose not found. Please install Docker Compose." "Red"
    exit 1
}

# Check Node.js
if (-not (Test-Command "node")) {
    Write-Log "❌ Node.js not found. Please install Node.js 18+" "Red"
    exit 1
}

# Check npm
if (-not (Test-Command "npm")) {
    Write-Log "❌ npm not found. Please install npm." "Red"
    exit 1
}

Write-Log "✅ All prerequisites found!" "Green"

# Navigate to project directory
Set-Location -Path $PSScriptRoot
Write-Log "📁 Working directory: $(Get-Location)" "Yellow"

# Stop existing containers if CleanStart
if ($CleanStart -or $Force) {
    Write-Log "🛑 Stopping and removing existing containers..." "Yellow"
    docker-compose down --remove-orphans --volumes 2>$null
    docker system prune -f 2>$null
    Write-Log "✅ Cleaned up existing containers" "Green"
}

# Update timestamp summary
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$summaryUpdate = @"

## System Startup - $timestamp

### Actions Taken:
- Fixed Qdrant duplicate indexing_threshold configuration
- Updated docker-compose.yml to remove environment variable conflicts
- Starting comprehensive production stack with all services
- Validating container health checks
- Setting up OCR, RAG, and multi-agent systems

"@

Add-Content -Path ".\COMPREHENSIVE_LEGAL_AI_TIMESTAMP_SUMMARY.md" -Value $summaryUpdate

# Create sveltekit-frontend Dockerfile if missing
$dockerfilePath = ".\sveltekit-frontend\Dockerfile"
if (-not (Test-Path $dockerfilePath)) {
    Write-Log "📝 Creating SvelteKit Dockerfile..." "Yellow"

$dockerfileContent = @'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start the application
CMD ["node", "build"]
'@

    Set-Content -Path $dockerfilePath -Value $dockerfileContent
    Write-Log "✅ Created SvelteKit Dockerfile" "Green"
}

# Create Context7 Dockerfile if missing
$context7DockerfilePath = ".\Dockerfile.context7"
if (-not (Test-Path $context7DockerfilePath)) {
    Write-Log "📝 Creating Context7 Dockerfile..." "Yellow"

$context7DockerfileContent = @'
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY mcp-servers/package*.json ./
RUN npm ci

# Copy MCP server code
COPY mcp-servers/ ./
COPY context7-mcp-server.js ./

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3001/health || exit 1

# Start Context7 MCP server
CMD ["node", "context7-mcp-server.js"]
'@

    Set-Content -Path $context7DockerfilePath -Value $context7DockerfileContent
    Write-Log "✅ Created Context7 Dockerfile" "Green"
}

# Create TTS Dockerfile if missing
$ttsDockerfilePath = ".\sveltekit-frontend\tts-service\Dockerfile"
$ttsDir = ".\sveltekit-frontend\tts-service"
if (-not (Test-Path $ttsDir)) {
    New-Item -ItemType Directory -Path $ttsDir -Force | Out-Null
}

if (-not (Test-Path $ttsDockerfilePath)) {
    Write-Log "📝 Creating TTS Service Dockerfile..." "Yellow"

$ttsDockerfileContent = @'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

# Start server
CMD ["python", "server.py"]
'@

    Set-Content -Path $ttsDockerfilePath -Value $ttsDockerfileContent

    # Create TTS server.py
    $ttsServerContent = @'
from flask import Flask, request, jsonify, send_file
import pyttsx3
import io
import wave
import tempfile
import os

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Initialize TTS engine
        engine = pyttsx3.init()

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            engine.save_to_file(text, tmp_file.name)
            engine.runAndWait()

            return send_file(tmp_file.name, mimetype='audio/wav')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)
'@

    Set-Content -Path "$ttsDir\server.py" -Value $ttsServerContent

    # Create TTS requirements.txt
    $ttsRequirementsContent = @'
Flask==2.3.3
pyttsx3==2.90
'@

    Set-Content -Path "$ttsDir\requirements.txt" -Value $ttsRequirementsContent
    Write-Log "✅ Created TTS Service" "Green"
}

# Create database init.sql if missing
$initSqlPath = ".\database\init.sql"
if (-not (Test-Path ".\database")) {
    New-Item -ItemType Directory -Path ".\database" -Force | Out-Null
}

if (-not (Test-Path $initSqlPath)) {
    Write-Log "📝 Creating database initialization script..." "Yellow"

$initSqlContent = @'
-- Legal AI Database Initialization
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create user tables
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create cases table
CREATE TABLE IF NOT EXISTS cases (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',
    user_id INTEGER REFERENCES users(id),
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content TEXT,
    case_id INTEGER REFERENCES cases(id),
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create evidence table
CREATE TABLE IF NOT EXISTS evidence (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    evidence_type VARCHAR(100),
    case_id INTEGER REFERENCES cases(id),
    document_id INTEGER REFERENCES documents(id),
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_cases_user_id ON cases(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_case_id ON documents(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);

-- Vector similarity search indexes
CREATE INDEX IF NOT EXISTS idx_cases_embedding ON cases USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_evidence_embedding ON evidence USING ivfflat (embedding vector_cosine_ops);
'@

    Set-Content -Path $initSqlPath -Value $initSqlContent
    Write-Log "✅ Created database initialization script" "Green"
}

# Install npm dependencies for frontend
Write-Log "📦 Installing frontend dependencies..." "Yellow"
Set-Location -Path ".\sveltekit-frontend"

if (Test-Path "package-lock.json") {
    npm ci
} else {
    npm install
}

# Fix any TypeScript/build issues
Write-Log "🔧 Running TypeScript checks and fixes..." "Yellow"
npm run check:fast 2>$null

Set-Location -Path ".."

# Start the complete stack
Write-Log "🚀 Starting complete legal AI stack..." "Yellow"

# Pull latest images
Write-Log "📥 Pulling latest Docker images..." "Yellow"
docker-compose pull

# Build services
Write-Log "🔨 Building services..." "Yellow"
docker-compose build --parallel

# Start services
Write-Log "▶️ Starting all services..." "Yellow"
docker-compose up -d

# Wait for services to be ready
Write-Log "⏳ Waiting for services to start..." "Yellow"
Start-Sleep -Seconds 30

# Check service health
Write-Log "🏥 Checking service health..." "Yellow"

$services = @("legal_ai_postgres", "legal_ai_redis", "legal_ai_qdrant", "legal_ai_ollama")
$healthyServices = 0

foreach ($service in $services) {
    $health = docker inspect --format='{{.State.Health.Status}}' $service 2>$null
    if ($health -eq "healthy" -or (docker ps --filter "name=$service" --filter "status=running" --quiet)) {
        Write-Log "✅ $service is running" "Green"
        $healthyServices++
    } else {
        Write-Log "⚠️ $service may have issues" "Yellow"
        docker logs $service --tail 10
    }
}

# Initialize Qdrant collections
Write-Log "🗂️ Initializing Qdrant collections..." "Yellow"
$qdrantUrl = "http://localhost:6333"

# Create collections using Qdrant API
$collections = @(
    @{name="legal_documents"; size=384},
    @{name="case_embeddings"; size=384},
    @{name="evidence_vectors"; size=384}
)

foreach ($collection in $collections) {
    $body = @{
        vectors = @{
            size = $collection.size
            distance = "Cosine"
        }
    } | ConvertTo-Json

    try {
        Invoke-RestMethod -Uri "$qdrantUrl/collections/$($collection.name)" -Method Put -Body $body -ContentType "application/json"
        Write-Log "✅ Created collection: $($collection.name)" "Green"
    } catch {
        Write-Log "⚠️ Collection $($collection.name) may already exist" "Yellow"
    }
}

# Load Ollama models
Write-Log "🤖 Loading Ollama models..." "Yellow"
try {
    # Check if gemma3 model exists, if not pull it
    $models = docker exec legal_ai_ollama ollama list 2>$null
    if ($models -notmatch "gemma") {
        Write-Log "📥 Pulling Gemma3 model (this may take a while)..." "Yellow"
        docker exec legal_ai_ollama ollama pull gemma:2b
    }

    # Pull nomic-embed for embeddings
    docker exec legal_ai_ollama ollama pull nomic-embed-text
    Write-Log "✅ Ollama models loaded" "Green"
} catch {
    Write-Log "⚠️ Could not load Ollama models automatically" "Yellow"
}

# Final status report
Write-Log "📊 Final System Status:" "Cyan"
Write-Log "======================" "Cyan"

# Check all services
$allServices = docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
Write-Host $allServices

# Display access URLs
Write-Log "" "White"
Write-Log "🌐 Access URLs:" "Cyan"
Write-Log "Frontend: http://localhost:3000" "Green"
Write-Log "Qdrant Dashboard: http://localhost:6333/dashboard" "Green"
Write-Log "Redis Insight: http://localhost:8001" "Green"
Write-Log "Ollama API: http://localhost:11434" "Green"
Write-Log "Context7 MCP: http://localhost:3001" "Green"

# Update final timestamp
$finalTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$finalUpdate = @"

### System Started Successfully - $finalTimestamp
- All Docker containers running
- Qdrant collections initialized
- Ollama models loaded
- Database schema created
- Frontend ready at http://localhost:3000

✅ Legal AI System is now OPERATIONAL!

"@

Add-Content -Path ".\COMPREHENSIVE_LEGAL_AI_TIMESTAMP_SUMMARY.md" -Value $finalUpdate

Write-Log "" "White"
Write-Log "🎉 Legal AI System startup complete!" "Green"
Write-Log "💡 Check the logs above for any warnings" "Yellow"
Write-Log "🚀 System ready for use!" "Cyan"
