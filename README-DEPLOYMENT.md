# Legal AI System - Production Deployment Guide

## 🚀 Quick Start

You already have the `legal_ai_db` database set up in pgAdmin, and you're using **Gemma3:legal** and **nomic-embed-text** models. Here's how to start everything:

### 1. Install and Setup Ollama Models
```bash
# Start Ollama service
ollama serve

# Install your models
ollama pull gemma3:legal
ollama pull nomic-embed-text
```

### 2. Start the Complete System
```bash
npm run start:production
# OR
START-PRODUCTION-COMPLETE.bat
```

### 3. Validate System Health
```bash
npm run test:validate
# OR
node validate-system.mjs
```

### 4. Run Complete Tests
```bash
npm run test:complete
# OR
RUN-TESTS.bat
```

## 📋 Prerequisites

✅ **Already Set Up (You have these)**:
- PostgreSQL with pgAdmin
- Database: `legal_ai_db`
- Node.js and npm
- Gemma3:legal model
- nomic-embed-text model

**Still Need (Optional)**:
- Redis (optional but recommended)
- Qdrant vector database (optional)
- MinIO object storage (optional)

## 🛠️ Quick Setup Commands

### Install Ollama (Required for AI features)
1. Download from: https://ollama.ai/download
2. Install and run: `ollama serve`
3. Install your models:
   ```bash
   ollama pull gemma3:legal
   ollama pull nomic-embed-text
   ```

### AI Model Configuration
- **Chat/Generation**: `gemma3:legal:latest`
- **Embeddings**: `nomic-embed-text` (768 dimensions)
- **Analysis**: `gemma3:legal:latest`
- **Summary**: `gemma3:legal:latest`

### Install Optional Services (Windows Native - No Docker)

**Redis (for caching)**:
```bash
# Download Redis for Windows from:
# https://redis.io/download
# Or use Windows package manager:
winget install Redis.Redis

# Start Redis
redis-server --port 6379
```

**Qdrant (for vector search)**:
```bash
# Download from: https://qdrant.tech/documentation/quick-start/
# Extract and run:
qdrant
```

**MinIO (for file storage)**:
```bash
# Download from: https://min.io/download
# Run with:
minio server ./minio-data --console-address :9001
```

## 🔧 System Architecture

### Database Setup
- **Database**: `legal_ai_db` (already created in pgAdmin)
- **Extensions**: pgvector, pg_trgm, uuid-ossp
- **Connection**: `postgresql://postgres:postgres@localhost:5432/legal_ai_db`

### Service Ports
- **Frontend**: http://localhost:5173
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **Ollama**: http://localhost:11434
- **Qdrant**: http://localhost:6333
- **MinIO**: http://localhost:9000

### Microservices
- **Enhanced RAG**: http://localhost:8094
- **GPU Orchestrator**: http://localhost:8095
- **Vector Processor**: http://localhost:8096
- **Document Analyzer**: http://localhost:8097

## 🧪 Testing

### System Validation
The validation script checks all connections:
```bash
npm run test:validate
```

### Full Integration Tests
Runs Playwright tests and system validation:
```bash
npm run test:complete
```

## 🐛 Troubleshooting

### Database Issues
```bash
# Check if database exists
psql -U postgres -l | findstr legal_ai_db

# Run migrations manually
psql -U postgres -d legal_ai_db -f production-migration.sql
```

### Service Issues
```bash
# Check what's running on ports
netstat -an | findstr :5432
netstat -an | findstr :11434

# Kill conflicting processes
taskkill /F /PID <process_id>
```

### AI Model Issues
```bash
# Check Ollama status and models
ollama list

# Install your legal models
ollama pull gemma3:legal
ollama pull nomic-embed-text

# Test model
ollama run gemma3:legal "Test legal query"
```

## 📊 System Status

After starting, check these URLs:
- **Application**: http://localhost:5173
- **API Health**: http://localhost:5173/api/health
- **MinIO Console**: http://localhost:9001 (if installed)
- **Qdrant Dashboard**: http://localhost:6333/dashboard (if installed)

## 🔐 Default Credentials

**Admin Login**:
- Email: `admin@legalai.com`
- Password: `admin123`

**MinIO** (if used):
- Username: `minioadmin`
- Password: `minioadmin`

## 📝 File Structure

```
C:\Users\james\Desktop\deeds-web\deeds-web-app\
├── START-PRODUCTION-COMPLETE.bat    # Main startup script
├── RUN-TESTS.bat                    # Test runner
├── validate-system.mjs              # System validation
├── production-migration.sql         # Database schema
├── .env                             # Environment variables
└── README-DEPLOYMENT.md             # This file
```

## 🔄 Development vs Production

**Development**:
```bash
npm run dev
```

**Production**:
```bash
npm run start:production
```

## 📞 Support

If you encounter issues:
1. Run `npm run test:validate` to check system status
2. Check logs in the `logs/` directory
3. Verify all services are running with the validation script

The system is designed to be fault-tolerant - core features work with just PostgreSQL and Ollama, while optional services enhance functionality.

## 🎯 Windows Native Services

This deployment uses **Windows native services only** - no Docker required:
- **PostgreSQL**: Windows service or manual install
- **Redis**: Windows native Redis
- **Ollama**: Native Windows application
- **Qdrant**: Native Windows binary
- **MinIO**: Native Windows binary
- **RabbitMQ**: Windows service (optional)

All services run as native Windows processes for maximum performance and compatibility.
