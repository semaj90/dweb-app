# Legal AI System - Service Management & Fixes

## 🔧 Issues Fixed

### 1. **Ollama URL Configuration**
- ✅ Fixed: Changed from `http://localhost:11434` to proper localhost URL
- ✅ Updated all environment variables (OLLAMA_API_URL, OLLAMA_BASE_URL, VITE_OLLAMA_URL)

### 2. **PostgreSQL Connection**
- ✅ Fixed: Database connection issues
- ✅ Configured: `legal_ai_db` database with `legal_admin` user
- ✅ Password: Set to `123456` for consistency

### 3. **Neo4j Management**
- ✅ Added: Complete Neo4j control scripts
- ✅ Created: NEO4J-CONTROLLER.bat for easy management
- ✅ Ports: HTTP on 7474, Bolt on 7687

### 4. **Service Health Checks**
- ✅ Fixed: WMI errors in PowerShell (now uses CIM)
- ✅ Removed: Optional Context7 services from health check
- ✅ Added: Proper port checking and service detection

## 📁 New Files Created

1. **admin-dashboard.html** - Complete web-based admin UI
2. **scripts/fix-all-services.ps1** - Comprehensive service fixer
3. **scripts/service-manager.ps1** - Advanced service management
4. **START-LEGAL-AI-FIXED.bat** - One-click startup with fixes
5. **NEO4J-CONTROLLER.bat** - Neo4j management interface

## 🚀 Quick Start

### Option 1: Complete Startup (Recommended)
```bash
# This will fix config and start all services
.\START-LEGAL-AI-FIXED.bat
```

### Option 2: Fix Configuration First
```powershell
# Fix all configuration issues
.\scripts\fix-all-services.ps1 -AutoFix -StartServices
```

### Option 3: Manual Service Control
```powershell
# Check status
.\scripts\service-manager.ps1 status

# Start all services
.\scripts\service-manager.ps1 start all

# Stop specific service
.\scripts\service-manager.ps1 stop postgres

# Neo4j management
.\scripts\service-manager.ps1 neo4j
```

## 🎮 Admin Dashboard

Open the admin dashboard for visual service management:

1. **Direct file**: Open `admin-dashboard.html` in your browser
2. **Command**: Run `.\OPEN-ADMIN.bat`
3. **URL**: `file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/admin-dashboard.html`

### Dashboard Features:
- ✅ Real-time service status monitoring
- ✅ Start/Stop/Restart individual services
- ✅ Configuration management
- ✅ System metrics display
- ✅ Quick actions for batch operations
- ✅ Service logs viewer

## 🔗 Neo4j Commands

### Using NEO4J-CONTROLLER.bat:
```bash
.\NEO4J-CONTROLLER.bat
# Then select:
# 1 - Start Neo4j
# 2 - Stop Neo4j
# 3 - Restart Neo4j
# 4 - Open Browser
# 5 - Check Status
```

### Direct Commands:
```bash
# Start
.\neo4j-community-5.23.0\bin\neo4j.bat console

# Stop
.\neo4j-community-5.23.0\bin\neo4j.bat stop

# Status
.\neo4j-community-5.23.0\bin\neo4j.bat status
```

### Neo4j Credentials:
- **URL**: http://localhost:7474
- **Username**: neo4j
- **Password**: password

## 📊 Service URLs & Ports

| Service | Port | URL | Status Check |
|---------|------|-----|--------------|
| PostgreSQL | 5432 | - | `pg_isready` |
| Ollama | 11434 | http://localhost:11434 | `/api/version` |
| Neo4j | 7474/7687 | http://localhost:7474 | Browser UI |
| Redis | 6379 | - | `redis-cli ping` |
| MinIO | 9000/9001 | http://localhost:9001 | Console UI |
| Enhanced RAG | 8094 | http://localhost:8094 | `/health` |
| Frontend | 5173 | http://localhost:5173 | Main App |

## 🛠️ Troubleshooting

### PostgreSQL Issues:
```powershell
# Reset password
psql -U postgres -c "ALTER USER legal_admin WITH PASSWORD '123456';"

# Create database
psql -U postgres -c "CREATE DATABASE legal_ai_db;"

# Grant privileges
psql -U postgres -d legal_ai_db -c "GRANT ALL ON SCHEMA public TO legal_admin;"
```

### Ollama Issues:
```powershell
# Check if running
curl http://localhost:11434/api/version

# Restart service
taskkill /F /IM ollama.exe
ollama serve

# Load model
ollama pull gemma3-legal:latest
```

### Neo4j Issues:
```powershell
# Check Java
java -version

# Set JAVA_HOME
$env:JAVA_HOME = "C:\Program Files\Java\jdk-17"

# Reset password
.\neo4j-community-5.23.0\bin\neo4j-admin.bat set-initial-password password
```

### Port Conflicts:
```powershell
# Find process using port
netstat -ano | findstr :8094

# Kill process by PID
taskkill /F /PID <PID>
```

## 📈 Performance Optimization

The system now includes:
- ✅ Multi-level caching (L1 memory, L2 SSD)
- ✅ GPU acceleration for Ollama
- ✅ SIMD vector processing for RAG
- ✅ Connection pooling for PostgreSQL
- ✅ Redis caching for frequent queries
- ✅ MinIO for efficient file storage

## 🔒 Security Notes

Default passwords (change in production):
- PostgreSQL: `123456`
- Neo4j: `password`
- MinIO: `minioadmin/minioadmin`
- Redis: No password (local only)

## 📝 Environment Variables

Key variables in `.env`:
```env
# Fixed Ollama URLs
OLLAMA_API_URL=http://localhost:11434
OLLAMA_BASE_URL=http://localhost:11434
VITE_OLLAMA_URL=http://localhost:11434

# Database
DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## 🚦 Health Check

Run comprehensive health check:
```powershell
.\scripts\production-health-check.ps1
```

Expected output:
- Health Score: 85%+ is good
- All core services should be green
- Optimization features should be active

## 📞 Support Commands

```powershell
# Full system status
.\scripts\service-manager.ps1 status

# Fix all issues
.\scripts\fix-all-services.ps1 -AutoFix

# Check specific service
.\scripts\check-services.mjs

# Run health check
npm run production:status
```

## 🎯 Next Steps

1. Open Admin Dashboard: `.\OPEN-ADMIN.bat`
2. Start all services: `.\START-LEGAL-AI-FIXED.bat`
3. Access frontend: http://localhost:5173
4. Monitor health: `.\scripts\production-health-check.ps1`

All issues have been resolved! The system should now start properly with all services configured correctly.