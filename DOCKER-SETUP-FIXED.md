# Legal AI Assistant - Fixed Docker Setup

This is the **FIXED** version of your Docker Compose setup that resolves all the issues with duplicate keys, missing services, and WSL/Docker Desktop integration.

## 🚀 Quick Start

### Option 1: Windows Batch File (Recommended)
```bash
# Double-click this file or run from Command Prompt
START-LEGAL-AI-FIXED.bat
```

### Option 2: Manual Setup
```bash
# 1. Pull and build images
npm run setup

# 2. Start all services
npm run docker:up

# 3. Check health
npm run health

# 4. Access your app
# Open http://localhost:5173
```

## 🔧 Fixed Issues

### ✅ **No Duplicate Keys**
- Removed duplicate `reservations` keys in Docker Compose
- Fixed YAML syntax issues

### ✅ **Complete Service Stack**
- PostgreSQL + pgvector (Database)
- Ollama + GPU support (AI)
- Qdrant (Vector Search)
- Neo4j (Graph Database)
- RabbitMQ (Message Queue)
- Redis (Cache)
- SvelteKit (Frontend)

### ✅ **Named Volumes**
- All data persists across container restarts
- No data loss when containers are deleted

### ✅ **Proper Dependencies**
- Services start in correct order
- Health checks for all critical services
- Automatic dependency management

### ✅ **Environment Variables**
- All services properly connected
- Database URLs, API endpoints configured
- Production-ready environment setup

## 📱 Service Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| SvelteKit App | http://localhost:5173 | Main application |
| Neo4j Browser | http://localhost:7474 | Graph database UI |
| RabbitMQ Management | http://localhost:15672 | Message queue UI |
| Qdrant Dashboard | http://localhost:6333/dashboard | Vector search UI |

## 🛠 NPM Scripts

### Docker Management
```bash
npm run docker:up        # Start all services
npm run docker:down      # Stop all services  
npm run docker:restart   # Restart all services
npm run docker:logs      # View service logs
npm run docker:status    # Check container status
npm run docker:health    # Health check all services
npm run docker:clean     # Clean volumes and containers
```

### Development
```bash
npm run setup           # Initial setup
npm run health          # Check all services
npm run dev             # Start development server
```

## 🤖 AI Model Setup

After services are running, load your AI models:

```bash
# Load Gemma3 model
docker exec deeds-ollama-gpu ollama pull gemma3

# Test AI service
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3","prompt":"Hello","stream":false}'
```

## 🐛 Troubleshooting

### Docker Desktop Issues
1. Make sure Docker Desktop is running
2. Enable WSL integration: Settings > Resources > WSL Integration
3. Restart Docker Desktop if needed

### WSL Issues
1. Update WSL: `wsl --update`
2. Restart WSL: `wsl --shutdown` then open new terminal
3. Check Docker in WSL: `docker ps`

### Service Not Starting
```bash
# Check specific service logs
docker-compose -f docker-compose-fixed.yml logs [service-name]

# Example: Check Ollama logs
docker-compose -f docker-compose-fixed.yml logs ollama
```

### Port Conflicts
If ports are already in use, stop conflicting services:
```bash
# Find what's using a port
netstat -ano | findstr :5173

# Kill the process (replace PID)
taskkill /PID [PID] /F
```

## 🔄 Data Persistence

All your data is stored in Docker volumes:
- `postgres_data` - Database data
- `ollama_data` - AI models
- `qdrant_data` - Vector embeddings
- `neo4j_data` - Graph data
- `rabbitmq_data` - Message queue data
- `redis_data` - Cache data

## 📝 Development Workflow

1. Start services: `npm run docker:up`
2. Check health: `npm run health`
3. Make your changes in `sveltekit-frontend/`
4. The app will hot-reload automatically
5. View logs: `npm run docker:logs`
6. Stop services: `npm run docker:down`

## 🎯 What's Fixed vs Original

| Issue | Before | After |
|-------|--------|-------|
| Duplicate YAML keys | ❌ Broken | ✅ Fixed |
| Missing services | ❌ Incomplete | ✅ Full stack |
| Data persistence | ❌ Lost on restart | ✅ Persistent volumes |
| Service dependencies | ❌ Random startup | ✅ Ordered startup |
| Health checks | ❌ No monitoring | ✅ Full monitoring |
| WSL integration | ❌ Manual setup | ✅ Automated scripts |

Your stack now works reliably with Docker Desktop + WSL! 🎉
