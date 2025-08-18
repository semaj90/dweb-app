# System Verification Tools

This directory contains comprehensive system verification scripts for the DEEDS-WEB application.

## 📋 Overview

The verification tools check the following components:

### Core Services
- **PostgreSQL** (Port 5432) - Database server
- **Redis** (Port 6379) - Caching server  
- **Ollama** (Port 11434) - AI/ML model server
- **Dev Server** (Port 5173) - SvelteKit development server

### API Endpoints
- OCR Service (`/api/ocr/langextract`)
- Embeddings Service (`/api/embeddings/generate`)
- Search Service (`/api/documents/search`)
- Storage Service (`/api/documents/store`)
- AI Upload Demo (`/ai-upload-demo`)

### Processing Capabilities
- Real embedding generation using Ollama models
- Database connectivity
- Model availability (nomic-embed-text, llama3.2)

## 🚀 Quick Start

### Option 1: Batch File (Easiest)
```batch
VERIFY-SYSTEM.bat
```
This provides a menu to choose between PowerShell, Node.js, or quick verification.

### Option 2: PowerShell Script (Recommended for Windows)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify-system.ps1
```

### Option 3: Node.js Script (Cross-platform)
```bash
node scripts/verify-system.mjs
```

## 🔧 Auto-Start Services

To automatically start all required services:
```batch
AUTO-START-SERVICES.bat
```
**Note:** Must be run as Administrator

## 📊 Output Interpretation

### ✅ Fully Operational
```
🎉 SYSTEM FULLY OPERATIONAL!
✅ All services running
✅ All APIs healthy
✅ Database accessible
✅ Real processing working
```

### ⚠️ Partially Operational
The script will show:
- Which services are not running
- Which APIs are not responding
- Recommended actions to fix issues

### Service Status Indicators
- `✅ RUNNING` - Service is active and responding
- `❌ NOT RUNNING` - Service needs to be started
- `⚠️ ERROR` - Service is running but has issues
- `[✓]` - Port is open and listening
- `[✗]` - Port is not accessible

## 🛠️ Troubleshooting

### PostgreSQL Not Starting
```batch
# Check service status
sc query postgresql-x64-15

# Start manually
pg_ctl start -D "C:\Program Files\PostgreSQL\15\data"
```

### Redis Not Starting
```batch
# Start Redis server
redis-server

# Or as Windows service
net start Redis
```

### Ollama Not Starting
```batch
# Start Ollama serve
ollama serve

# Check installed models
ollama list

# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Dev Server Not Starting
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

## 📁 Output Files

### verification-results.json
Contains detailed results including:
- Timestamp of verification
- Service status details
- API response data
- Processing test results
- Recommendations for fixes

Example:
```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "services": {
    "PostgreSQL": {
      "running": true,
      "port": 5432,
      "portOpen": true
    }
  },
  "apis": {
    "Dev Server Root": {
      "success": true,
      "status": 200
    }
  },
  "processing": {
    "embeddings": true,
    "ollamaModels": ["nomic-embed-text", "llama3.2"]
  }
}
```

## 🔄 Verification Process Flow

1. **Service Check**
   - Check if ports are open
   - Verify Windows services are running
   - Check for active processes

2. **Database Connection**
   - Test PostgreSQL port accessibility
   - Attempt actual database connection

3. **API Health**
   - Test each API endpoint
   - Check response status codes
   - Verify response data structure

4. **AI Model Status**
   - Check Ollama API accessibility
   - List installed models
   - Verify required models are present

5. **Real Processing Test**
   - Generate test embeddings
   - Verify response format
   - Check processing pipeline

6. **Summary & Recommendations**
   - Display overall system status
   - Provide actionable recommendations
   - Save detailed results to file

## 🌐 Key URLs

When system is fully operational:
- **Main Application**: http://localhost:5173
- **AI Upload Demo**: http://localhost:5173/ai-upload-demo
- **API Base**: http://localhost:5173/api

## 💡 Best Practices

1. **Run verification after system changes**
2. **Check results JSON for detailed diagnostics**
3. **Use AUTO-START-SERVICES.bat for quick recovery**
4. **Keep Ollama models updated**
5. **Monitor verification-results.json for trends**

## 🔒 Security Notes

- Scripts require appropriate permissions
- AUTO-START-SERVICES.bat needs Administrator rights
- Database uses default postgres credentials (update in production)
- API endpoints are currently unsecured (add authentication in production)

## 📝 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Port already in use | Kill the process using the port or change port configuration |
| Service won't start | Check Windows Services, run as Administrator |
| Ollama models missing | Run `ollama pull [model-name]` |
| Database connection failed | Check PostgreSQL credentials and pg_hba.conf |
| Dev server not starting | Run `npm install` then `npm run dev` |

## 🔗 Related Files

- `START-REAL-SYSTEM.bat` - Original system startup script
- `package.json` - Node.js dependencies and scripts
- `.env` - Environment configuration (create if missing)

## 📞 Support

For issues with verification tools:
1. Check verification-results.json for detailed error information
2. Run AUTO-START-SERVICES.bat as Administrator
3. Verify all dependencies are installed
4. Check Windows Event Viewer for service errors
