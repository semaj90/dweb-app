# Comprehensive Fix Script for Web App with Docker, Ollama, and SvelteKit
Write-Host "=== Starting Comprehensive Fix for Web App ===" -ForegroundColor Cyan

# Set error action preference
$ErrorActionPreference = "Continue"

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker info 2>&1 | Out-Null
        return $?
    } catch {
        return $false
    }
}

# Function to check if command exists
function Test-CommandExists {
    param($command)
    try {
        Get-Command $command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Step 1: Check prerequisites
Write-Host "`n[1/10] Checking prerequisites..." -ForegroundColor Yellow

if (!(Test-CommandExists "docker")) {
    Write-Host "Error: Docker is not installed. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

if (!(Test-DockerRunning)) {
    Write-Host "Docker is not running. Starting Docker Desktop..." -ForegroundColor Yellow
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -WindowStyle Hidden
    Write-Host "Waiting for Docker to start (60 seconds)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 60
}

if (!(Test-CommandExists "npm")) {
    Write-Host "Error: Node.js/npm is not installed. Please install Node.js." -ForegroundColor Red
    exit 1
}

# Step 2: Create necessary directories
Write-Host "`n[2/10] Creating necessary directories..." -ForegroundColor Yellow
$directories = @(
    ".\scripts",
    ".\logs",
    ".\uploads",
    ".\sveltekit-frontend\src\lib\server\db",
    ".\sveltekit-frontend\drizzle"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# Step 3: Create Ollama models file
Write-Host "`n[3/10] Creating Ollama models configuration..." -ForegroundColor Yellow
@"
gemma3
gemma:2b
nomic-embed-text
llama3.2
mistral
"@ | Set-Content -Path ".\scripts\ollama-models.txt" -Force

# Step 4: Create database initialization scripts
Write-Host "`n[4/10] Creating database initialization scripts..." -ForegroundColor Yellow

# Create pgvector initialization script
@"
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE user_role AS ENUM ('admin', 'investigator', 'analyst', 'viewer');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE case_status AS ENUM ('open', 'active', 'closed', 'archived');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE evidence_type AS ENUM ('document', 'image', 'video', 'audio', 'other');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE prosecutor_db TO postgres;
"@ | Set-Content -Path ".\scripts\init-pgvector.sql" -Force

# Create schema initialization script
@"
-- Schema initialization will be handled by Drizzle migrations
-- This file is a placeholder for any custom SQL needed
"@ | Set-Content -Path ".\scripts\init-schema.sql" -Force

# Step 5: Fix TypeScript configuration
Write-Host "`n[5/10] Fixing TypeScript configuration..." -ForegroundColor Yellow

# Fix missing exports in database index
$dbIndexPath = ".\sveltekit-frontend\src\lib\server\db\index.ts"
if (Test-Path $dbIndexPath) {
    $dbIndexContent = Get-Content $dbIndexPath -Raw
    if ($dbIndexContent -notmatch "export.*isPostgreSQL") {
        $dbIndexContent += "`n`n// Database type check`nexport const isPostgreSQL = true;"
        Set-Content -Path $dbIndexPath -Value $dbIndexContent -Force
        Write-Host "Fixed missing isPostgreSQL export" -ForegroundColor Green
    }
}

# Step 6: Fix package.json dependencies
Write-Host "`n[6/10] Updating package dependencies..." -ForegroundColor Yellow

# Update root package.json
$rootPackageJson = Get-Content ".\package.json" -Raw | ConvertFrom-Json

# Ensure all required dependencies
$requiredDeps = @{
    "@tauri-apps/api" = "^2.5.0"
    "bcryptjs" = "^2.4.3"
    "dotenv" = "^16.5.0"
    "pg" = "^8.16.2"
    "postgres" = "^3.4.4"
    "@qdrant/js-client-rest" = "^1.9.0"
    "redis" = "^4.6.13"
    "ollama" = "^0.5.11"
}

foreach ($dep in $requiredDeps.Keys) {
    if (!$rootPackageJson.dependencies.$dep) {
        $rootPackageJson.dependencies | Add-Member -MemberType NoteProperty -Name $dep -Value $requiredDeps[$dep] -Force
    }
}

$rootPackageJson | ConvertTo-Json -Depth 10 | Set-Content ".\package.json" -Force

# Update frontend package.json
$frontendPackageJson = Get-Content ".\sveltekit-frontend\package.json" -Raw | ConvertFrom-Json

# Ensure frontend dependencies
$frontendDeps = @{
    "@tiptap/core" = "^2.1.13"
    "@tiptap/starter-kit" = "^2.1.13"
    "bits-ui" = "^0.21.19"
    "lucide-svelte" = "^0.378.0"
    "drizzle-orm" = "^0.44.2"
}

foreach ($dep in $frontendDeps.Keys) {
    if (!$frontendPackageJson.dependencies.$dep) {
        $frontendPackageJson.dependencies | Add-Member -MemberType NoteProperty -Name $dep -Value $frontendDeps[$dep] -Force
    }
}

$frontendPackageJson | ConvertTo-Json -Depth 10 | Set-Content ".\sveltekit-frontend\package.json" -Force

# Step 7: Fix environment files
Write-Host "`n[7/10] Fixing environment configuration..." -ForegroundColor Yellow

# Ensure frontend .env matches root .env
Copy-Item -Path ".\.env" -Destination ".\sveltekit-frontend\.env" -Force

# Create docker-compose override for development
@"
version: '3.8'

services:
  ollama:
    environment:
      - OLLAMA_MODELS=gemma3,nomic-embed-text
    volumes:
      - ./ollama_models:/root/.ollama
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
"@ | Set-Content -Path ".\docker-compose.override.yml" -Force

# Step 8: Stop existing containers and clean up
Write-Host "`n[8/10] Cleaning up existing Docker containers..." -ForegroundColor Yellow
docker-compose down -v 2>&1 | Out-Null
docker system prune -f 2>&1 | Out-Null

# Step 9: Start Docker services
Write-Host "`n[9/10] Starting Docker services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to be ready
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    $postgresReady = docker exec prosecutor_postgres pg_isready -U postgres 2>&1 | Select-String "accepting connections"
    $ollamaReady = (Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -ErrorAction SilentlyContinue) -ne $null
    
    if ($postgresReady -and $ollamaReady) {
        Write-Host "All services are ready!" -ForegroundColor Green
        break
    }
    
    $attempt++
    Write-Host "Waiting for services... ($attempt/$maxAttempts)" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

# Step 10: Install dependencies and run fixes
Write-Host "`n[10/10] Installing dependencies and running fixes..." -ForegroundColor Yellow

# Install root dependencies
Write-Host "Installing root dependencies..." -ForegroundColor Yellow
npm install

# Install frontend dependencies
Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
Set-Location ".\sveltekit-frontend"
npm install

# Create TypeScript fix script
@"
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';

// Fix vector-search.ts
const vectorSearchPath = join('src', 'lib', 'server', 'search', 'vector-search.ts');
if (existsSync(vectorSearchPath)) {
    let content = readFileSync(vectorSearchPath, 'utf-8');
    
    // Fix cache.get type parameter
    content = content.replace(
        /cache\.get<([^>]+)>\(/g,
        'cache.get('
    );
    
    writeFileSync(vectorSearchPath, content);
    console.log('Fixed vector-search.ts');
}

// Fix embedding-service.ts
const embeddingServicePath = join('src', 'lib', 'server', 'services', 'embedding-service.ts');
if (existsSync(embeddingServicePath)) {
    let content = readFileSync(embeddingServicePath, 'utf-8');
    
    // Fix error type
    content = content.replace(
        /catch \(error\) {/g,
        'catch (error: any) {'
    );
    
    writeFileSync(embeddingServicePath, content);
    console.log('Fixed embedding-service.ts');
}

// Fix vector-service.ts
const vectorServicePath = join('src', 'lib', 'server', 'services', 'vector-service.ts');
if (existsSync(vectorServicePath)) {
    let content = readFileSync(vectorServicePath, 'utf-8');
    
    // Fix metadata expression
    content = content.replace(
        /metadata:\s*{[^}]+}\s*\|\|\s*{}/g,
        'metadata: {
                    contentType: contentType,
                    caseId: options.caseId,
                    ...options.metadata,
                }'
    );
    
    writeFileSync(vectorServicePath, content);
    console.log('Fixed vector-service.ts');
}

console.log('TypeScript fixes completed!');
"@ | Set-Content -Path "fix-typescript-errors.mjs" -Force

# Run TypeScript fixes
Write-Host "Running TypeScript fixes..." -ForegroundColor Yellow
node fix-typescript-errors.mjs

# Run database migrations
Write-Host "Running database migrations..." -ForegroundColor Yellow
npm run db:migrate

# Return to root directory
Set-Location ..

# Final status check
Write-Host "`n=== Fix Complete ===" -ForegroundColor Green
Write-Host "`nServices Status:" -ForegroundColor Cyan
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "1. Run 'npm run dev' to start the development server" -ForegroundColor White
Write-Host "2. Access the application at http://localhost:5173" -ForegroundColor White
Write-Host "3. Ollama API is available at http://localhost:11434" -ForegroundColor White
Write-Host "4. PostgreSQL is available at localhost:5432" -ForegroundColor White

Write-Host "`nTo test Ollama integration:" -ForegroundColor Cyan
Write-Host "curl http://localhost:11434/api/tags" -ForegroundColor White
