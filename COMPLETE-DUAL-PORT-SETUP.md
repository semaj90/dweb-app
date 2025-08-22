# 🌐 Complete Dual-Port Setup - Legal AI Platform

## 🎯 **What This Setup Provides**

This configuration gives you a **complete dual-port setup** for your Legal AI Platform:
- **Port 5173**: Your main SvelteKit frontend
- **Port 5180**: A proxy server that forwards all requests to port 5173
- **Environment Variables**: Complete configuration for all services
- **Automatic Startup**: Scripts to start everything with one command

## 🚀 **Quick Start Options**

### **Option 1: NPM Scripts (Recommended)**
```bash
# From the root directory (deeds-web-app)
npm run start:env         # Start both frontend and proxy with environment
npm run dev:dual          # Start both frontend and proxy
npm run start:dual        # Start both frontend and proxy
npm run dev:proxy         # Start proxy only (if frontend is running)

# From the frontend directory
cd sveltekit-frontend
npm run start:env         # Start both with environment
npm run dev:dual          # Start both frontend and proxy
npm run start:dual        # Start both frontend and proxy
```

### **Option 2: Direct Script Execution**
```bash
# Start both services with environment
node start-dual-ports-with-env.cjs

# Start just the proxy (if frontend is already running)
node start-port-5180.cjs
```

### **Option 3: Windows Batch Files**
```bash
# From the root directory
start-dual-ports.bat      # Start both services
start-port-5180.bat       # Start just proxy

# From the root directory (deeds-web)
start-dual-ports.bat      # Start both services
start-port-5180.bat       # Start just proxy
```

## 📁 **Files Created**

### **Root Directory (deeds-web-app/)**
- `env-config.js` - Environment configuration loader
- `start-dual-ports-with-env.cjs` - Complete startup script with environment
- `start-port-5180.cjs` - Simple proxy server
- `simple-port-5180-proxy.cjs` - Alternative proxy implementation
- `start-all-services.cjs` - Node.js script to start all services

### **Root Directory (deeds-web/)**
- `start-dual-ports.bat` - Windows batch file for complete setup
- `start-port-5180.bat` - Windows batch file for proxy only
- `PORT-5180-SETUP.md` - Setup documentation

## 🔧 **How It Works**

1. **Environment Loading**: `env-config.js` sets all necessary environment variables
2. **Frontend Service**: SvelteKit runs on port 5173 (your existing setup)
3. **Proxy Service**: Listens on port 5180 and forwards all requests to port 5173
4. **Result**: Both URLs serve identical content

## 🌐 **Access Your App**

Once both services are running:
- **Main Frontend**: http://localhost:5173
- **Port 5180**: http://localhost:5180

Both URLs will show the exact same homepage and functionality.

## 📋 **Environment Variables Set**

### **Database Configuration**
- `DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db`
- `PGPASSWORD=123456`
- Database connection settings

### **Service Ports**
- `FRONTEND_PORT=5173`
- `PROXY_PORT=5180`
- `API_PORT=8080`
- `UPLOAD_SERVICE_PORT=8093`
- `RAG_SERVICE_PORT=8094`
- `QUIC_GATEWAY_PORT=8447`

### **AI & GPU Configuration**
- `OLLAMA_ENDPOINT=http://localhost:11434`
- `GPU_ENABLED=true`
- `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`

### **File Paths**
- `UPLOADS_DIR=./uploads`
- `DOCUMENTS_DIR=./documents`
- `EVIDENCE_DIR=./evidence`
- `LOGS_DIR=./logs`
- `GENERATED_REPORTS_DIR=./generated_reports`

## 🎯 **NPM Scripts Added**

### **Root Package.json**
```json
{
  "dev:dual": "concurrently --restart-tries 3 --restart-after 4000 -n FRONTEND,PROXY \"pnpm --filter yorha-legal-ai-frontend run dev\" \"node start-port-5180.cjs\"",
  "dev:proxy": "node start-port-5180.cjs",
  "start:proxy": "node start-port-5180.cjs",
  "start:dual": "concurrently --restart-tries 3 --restart-after 4000 -n FRONTEND,PROXY \"pnpm --filter yorha-legal-ai-frontend run dev\" \"node start-port-5180.cjs\"",
  "start:env": "node start-dual-ports-with-env.cjs"
}
```

### **Frontend Package.json**
```json
{
  "dev:dual": "concurrently --restart-tries 3 --restart-after 4000 -n FRONTEND,PROXY \"vite dev\" \"cd .. && node start-port-5180.cjs\"",
  "dev:proxy": "cd .. && node start-port-5180.cjs",
  "start:dual": "concurrently --restart-tries 3 --restart-after 4000 -n FRONTEND,PROXY \"vite dev\" \"cd .. && node start-port-5180.cjs\""
}
```

## 🚀 **Complete Startup Process**

### **Step 1: Environment Setup**
```bash
# The startup script automatically:
# 1. Loads environment variables from env-config.js
# 2. Creates necessary directories (uploads, documents, evidence, logs, etc.)
# 3. Sets up all configuration
```

### **Step 2: Service Startup**
```bash
# 1. Starts frontend on port 5173
# 2. Waits 15 seconds for frontend to initialize
# 3. Starts proxy server on port 5180
# 4. Both services run concurrently
```

### **Step 3: Verification**
```bash
# Check both ports are accessible:
curl http://localhost:5173/        # Frontend
curl http://localhost:5180/        # Proxy (should show same content)
```

## ⚠️ **Important Notes**

- **Frontend must be running first** before starting the proxy
- Environment variables are automatically loaded
- All necessary directories are created automatically
- Both ports serve identical content
- Use Ctrl+C to stop services

## 🐛 **Troubleshooting**

### **Proxy Shows "Connection Refused"**
- Make sure frontend is running on port 5173
- Check: `netstat -ano | findstr ":5173"`

### **Port 5180 Already in Use**
- Stop existing services: `netstat -ano | findstr ":5180"`
- Kill process if needed

### **Environment Not Loading**
- Check `env-config.js` exists
- Verify file permissions
- Check console for error messages

## 🎉 **Success Indicators**

When everything is working:
- ✅ Frontend accessible at http://localhost:5173
- ✅ Port 5180 accessible at http://localhost:5180
- ✅ Both URLs show identical content
- ✅ Environment variables loaded
- ✅ All directories created
- ✅ No proxy errors in console

## 📞 **Need Help?**

### **Check Service Status**
```bash
# Check if services are running
netstat -ano | findstr ":5173"    # Frontend
netstat -ano | findstr ":5180"    # Proxy

# Check environment
node -e "console.log(process.env.FRONTEND_PORT)"
```

### **Restart Everything**
```bash
# Stop all services (Ctrl+C)
# Then restart:
npm run dev:dual
```

---

**Last Updated**: August 21, 2025  
**Status**: Complete and Ready to Use  
**Features**: Dual Ports + Environment + Auto-Startup
