# 🚀 Legal AI Platform - One-Command Launch

## Quick Start

### Option 1: Direct zx (Recommended)
```bash
zx launch.mjs
```

### Option 2: npm script
```bash
npm run launch
```

### Option 3: Full script path
```bash
zx scripts/launch-legal-ai-platform.mjs
```

## What This Does

The automated launch script performs these steps:

### 🔧 **Step 1: External Services**
- ✅ Checks and starts MinIO (Object Storage)
- ✅ Checks and starts Redis (Caching)  
- ✅ Verifies PostgreSQL (Database)
- ✅ Checks Neo4j (Graph Database)
- ✅ Ensures Ollama is running (AI Models)

### 🔍 **Step 2: Health Check**
- ✅ Comprehensive service status verification
- ✅ Reports which services are running
- ✅ Continues with reduced functionality if some services are down

### 🚀 **Step 3: Platform Launch**
- ✅ Cleans up any conflicting processes
- ✅ Sets proper environment variables
- ✅ Launches `npm run dev:full` with full orchestration

## Features

- **🤖 Automated**: Zero manual intervention required
- **🔧 Intelligent**: Detects and starts missing services
- **🛡️ Safe**: Cleans up conflicts before starting
- **📊 Monitored**: Real-time status reporting
- **🎯 Complete**: Handles all 37 microservices + frontend

## Output

You'll see beautiful terminal output with:
- YoRHa-themed startup messages
- Service status indicators (✅ ❌)
- Progress tracking for each step
- Colored output for easy reading

## Manual Alternatives

If the automated script fails, you can run manually:

```bash
# 1. Start services
scripts/start-external-services.bat

# 2. Check health
node scripts/check-services.mjs

# 3. Launch platform
cd sveltekit-frontend && npm run dev:full
```

## Requirements

- **zx**: Should be installed (`npm install -g zx`)
- **External Services**: MinIO, Redis, PostgreSQL, Neo4j, Ollama
- **Dependencies**: All npm packages installed

## Troubleshooting

If launch fails:
1. Check that you're in the `deeds-web-app` directory
2. Ensure zx is globally installed: `npm install -g zx`  
3. Verify external services are installed
4. Check port availability (8093, 8094, etc.)

---

**Ready to Launch Your Legal AI Platform? Run:** `zx launch.mjs`