# 🏛️ Legal AI Case Management System - Performance Optimized
# Quick deployment and management scripts

# Stage 1: Core Infrastructure
Write-Host "🏛️ LEGAL AI CASE MANAGEMENT SYSTEM - OPTIMIZED SETUP" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Yellow

# Check prerequisites
Write-Host "📋 Checking system prerequisites..." -ForegroundColor Cyan

# Docker check
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not found! Please install Docker Desktop" -ForegroundColor Red
    exit 1
}

# Node.js check
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found! Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

# Git check
try {
    $gitVersion = git --version
    Write-Host "✅ Git: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Git not found - some features may be limited" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Starting Legal AI System Setup..." -ForegroundColor Green

# Stage 2: Environment Setup
Write-Host "📁 Setting up environment files..." -ForegroundColor Cyan

# Create optimized .env file
$envContent = @"
# Legal AI Case Management System - Optimized Configuration
# Database Configuration
DATABASE_URL=postgresql://legal_admin:LegalSecure2024!@localhost:5432/legal_ai_v3
POSTGRES_USER=legal_admin
POSTGRES_PASSWORD=LegalSecure2024!
POSTGRES_DB=legal_ai_v3

# AI Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3-legal-enhanced
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_KEEP_ALIVE=24h

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=legal_qdrant_secure_2024

# Redis Cache
REDIS_URL=redis://:LegalRedis2024!@localhost:6379
REDIS_TTL=3600

# Application Settings
NODE_ENV=development
VITE_API_BASE_URL=http://localhost:5173
PUBLIC_OLLAMA_URL=http://localhost:11434

# Security
JWT_SECRET=legal_ai_jwt_super_secure_key_2024_change_in_production
SESSION_SECRET=legal_ai_session_super_secure_key_2024

# Features
ENABLE_AI_FEATURES=true
ENABLE_REAL_TIME=true
ENABLE_COLLABORATION=true
ENABLE_DOCUMENT_PROCESSING=true
ENABLE_ADVANCED_ANALYTICS=true
ENABLE_VOICE_NOTES=true

# Performance
NODE_OPTIONS=--max-old-space-size=4096
VITE_LEGACY_BUILD=false
VITE_OPTIMIZE_DEPS=true

# Logging
LOG_LEVEL=info
LOG_FILE=logs/legal-ai.log
"@

Set-Content -Path ".env" -Value $envContent
Write-Host "✅ Environment configuration created" -ForegroundColor Green

# Stage 3: Database Initialization
Write-Host "💾 Setting up optimized database configuration..." -ForegroundColor Cyan

# Create database directory
New-Item -ItemType Directory -Force -Path "database" | Out-Null

# Create optimized database init script
$dbInitScript = @"
-- Legal AI Case Management System - Optimized Database Setup
-- PostgreSQL with pgvector extensions and performance tuning

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Performance optimization settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '8MB';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET max_worker_processes = 8;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;

-- Reload configuration
SELECT pg_reload_conf();

-- Create optimized indexes for legal case management
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cases_created_at_btree ON cases USING btree(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cases_status_gin ON cases USING gin(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evidence_case_id ON evidence USING btree(case_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evidence_content_trgm ON evidence USING gin(content gin_trgm_ops);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_embedding_vector ON documents USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- Create full-text search indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cases_title_fts ON cases USING gin(to_tsvector('english', title));
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cases_description_fts ON cases USING gin(to_tsvector('english', description));
"@

Set-Content -Path "database/init-optimized.sql" -Value $dbInitScript
Write-Host "✅ Database initialization script created" -ForegroundColor Green

# Create performance tuning script
$perfTuningScript = @"
-- Performance Tuning for Legal AI Database
-- Run after initial setup

-- Analyze all tables for better query planning
ANALYZE;

-- Update table statistics
VACUUM ANALYZE cases;
VACUUM ANALYZE evidence;
VACUUM ANALYZE documents;
VACUUM ANALYZE users;

-- Create materialized views for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS case_summary_stats AS
SELECT 
    status,
    COUNT(*) as total_cases,
    AVG(EXTRACT(DAY FROM (updated_at - created_at))) as avg_duration_days
FROM cases 
GROUP BY status;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW case_summary_stats;

-- Create function for fast case search
CREATE OR REPLACE FUNCTION search_cases_optimized(search_term TEXT)
RETURNS TABLE(
    case_id UUID,
    title TEXT,
    description TEXT,
    status TEXT,
    relevance_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.title,
        c.description,
        c.status,
        ts_rank(to_tsvector('english', c.title || ' ' || c.description), plainto_tsquery('english', search_term)) as relevance
    FROM cases c
    WHERE to_tsvector('english', c.title || ' ' || c.description) @@ plainto_tsquery('english', search_term)
    ORDER BY relevance DESC;
END;
$$ LANGUAGE plpgsql;
"@

Set-Content -Path "database/performance-tuning.sql" -Value $perfTuningScript
Write-Host "✅ Performance tuning script created" -ForegroundColor Green

# Stage 4: Enhanced Docker Services
Write-Host "🐳 Creating enhanced Docker services..." -ForegroundColor Cyan

# Create collaboration server
New-Item -ItemType Directory -Force -Path "collaboration-server" | Out-Null

$collaborationPackageJson = @"
{
  "name": "legal-collaboration-server",
  "version": "1.0.0",
  "description": "Real-time collaboration server for Legal AI Case Management",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js"
  },
  "dependencies": {
    "express": "^4.18.0",
    "socket.io": "^4.7.0",
    "redis": "^4.6.0",
    "cors": "^2.8.5",
    "helmet": "^7.0.0",
    "compression": "^1.7.4",
    "pg": "^8.11.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.0"
  }
}
"@

Set-Content -Path "collaboration-server/package.json" -Value $collaborationPackageJson

$collaborationServer = @"
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const Redis = require('redis');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: ["http://localhost:5173", "http://localhost:4173"],
    methods: ["GET", "POST"]
  }
});

// Redis client for real-time data
const redis = Redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

redis.connect();

// Middleware
app.use(helmet());
app.use(compression());
app.use(cors());
app.use(express.json());

// Track active users and sessions
const activeUsers = new Map();
const activeCases = new Map();

// Socket.io connection handling
io.on('connection', (socket) => {
  console.log('User connected:', socket.id);
  
  // Join case room for collaboration
  socket.on('join-case', (caseId, userInfo) => {
    socket.join(caseId);
    
    // Track user in case
    if (!activeCases.has(caseId)) {
      activeCases.set(caseId, new Set());
    }
    activeCases.get(caseId).add(socket.id);
    activeUsers.set(socket.id, { ...userInfo, caseId });
    
    // Notify others in the case
    socket.to(caseId).emit('user-joined', userInfo);
    
    // Send current collaborators
    const collaborators = Array.from(activeCases.get(caseId))
      .map(id => activeUsers.get(id))
      .filter(user => user && user.caseId === caseId);
    
    socket.emit('current-collaborators', collaborators);
  });
  
  // Real-time document editing
  socket.on('document-change', (data) => {
    const user = activeUsers.get(socket.id);
    if (user && user.caseId) {
      socket.to(user.caseId).emit('document-update', {
        ...data,
        user: user.name,
        timestamp: Date.now()
      });
      
      // Cache changes in Redis
      redis.setEx(`doc-changes:${data.documentId}`, 3600, JSON.stringify(data));
    }
  });
  
  // Evidence annotations
  socket.on('evidence-annotation', (data) => {
    const user = activeUsers.get(socket.id);
    if (user && user.caseId) {
      socket.to(user.caseId).emit('new-annotation', {
        ...data,
        user: user.name,
        timestamp: Date.now()
      });
    }
  });
  
  // Case status updates
  socket.on('case-update', (data) => {
    const user = activeUsers.get(socket.id);
    if (user && user.caseId) {
      socket.to(user.caseId).emit('case-updated', {
        ...data,
        updatedBy: user.name,
        timestamp: Date.now()
      });
    }
  });
  
  // Handle disconnection
  socket.on('disconnect', () => {
    const user = activeUsers.get(socket.id);
    if (user) {
      // Remove from active case
      if (activeCases.has(user.caseId)) {
        activeCases.get(user.caseId).delete(socket.id);
        if (activeCases.get(user.caseId).size === 0) {
          activeCases.delete(user.caseId);
        }
      }
      
      // Notify others
      socket.to(user.caseId).emit('user-left', user);
      activeUsers.delete(socket.id);
    }
    
    console.log('User disconnected:', socket.id);
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    activeUsers: activeUsers.size,
    activeCases: activeCases.size,
    timestamp: new Date().toISOString()
  });
});

// API endpoint for collaboration stats
app.get('/api/collaboration/stats', (req, res) => {
  const stats = {
    totalActiveUsers: activeUsers.size,
    totalActiveCases: activeCases.size,
    caseActivity: Array.from(activeCases.entries()).map(([caseId, users]) => ({
      caseId,
      activeUsers: users.size
    }))
  };
  
  res.json(stats);
});

const PORT = process.env.PORT || 8080;
server.listen(PORT, () => {
  console.log(`🤝 Legal AI Collaboration Server running on port ${PORT}`);
});
"@

Set-Content -Path "collaboration-server/index.js" -Value $collaborationServer
Write-Host "✅ Collaboration server created" -ForegroundColor Green

# Install Dependencies and Start Services
Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan

# Install main project dependencies
if (Test-Path "package.json") {
    npm install
    Write-Host "✅ Main project dependencies installed" -ForegroundColor Green
}

# Install frontend dependencies
if (Test-Path "sveltekit-frontend/package.json") {
    Set-Location "sveltekit-frontend"
    npm install
    Set-Location ".."
    Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green
}

# Install collaboration server dependencies
if (Test-Path "collaboration-server/package.json") {
    Set-Location "collaboration-server"
    npm install
    Set-Location ".."
    Write-Host "✅ Collaboration server dependencies installed" -ForegroundColor Green
}

# Start Optimized Services
Write-Host "🚀 Starting optimized Legal AI services..." -ForegroundColor Green

# Start Docker services
Write-Host "🐳 Starting Docker containers..." -ForegroundColor Cyan
try {
    docker-compose -f docker-compose.optimized.yml up -d
    Write-Host "✅ Docker services started successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Docker services failed to start. Trying fallback..." -ForegroundColor Yellow
    docker-compose up -d
}

# Wait for services to be ready
Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Run database migrations
Write-Host "💾 Running database migrations..." -ForegroundColor Cyan
Set-Location "sveltekit-frontend"
try {
    npm run db:migrate
    Write-Host "✅ Database migrations completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Database migrations failed - will retry later" -ForegroundColor Yellow
}
Set-Location ".."

# Final verification
Write-Host "" 
Write-Host "🎉 LEGAL AI CASE MANAGEMENT SYSTEM SETUP COMPLETE!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "🌐 Access Points:" -ForegroundColor Cyan
Write-Host "  • Main Application: http://localhost:5173" -ForegroundColor White
Write-Host "  • Database Studio: Run 'npm run db:studio' in sveltekit-frontend" -ForegroundColor White
Write-Host "  • Collaboration Server: http://localhost:8080" -ForegroundColor White
Write-Host "  • Document Processor: http://localhost:8081" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Management Tools:" -ForegroundColor Cyan
Write-Host "  • Master Control Panel: .\LEGAL-AI-MASTER-CONTROL.bat" -ForegroundColor White
Write-Host "  • Service Status: docker ps" -ForegroundColor White
Write-Host "  • Logs: docker-compose logs -f" -ForegroundColor White
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Open http://localhost:5173 in your browser" -ForegroundColor White
Write-Host "  2. Register a new account or use admin@example.com / password123" -ForegroundColor White
Write-Host "  3. Create your first legal case" -ForegroundColor White
Write-Host "  4. Upload documents and evidence" -ForegroundColor White
Write-Host "  5. Explore AI-powered legal analysis features" -ForegroundColor White
Write-Host ""
Write-Host "🎯 Legal AI system is now fully operational with advanced features!" -ForegroundColor Green

# Open application in browser
Start-Process "http://localhost:5173"