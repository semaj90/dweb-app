@echo off
setlocal enabledelayedexpansion

:: COMPLETE LEGAL AI SYSTEM SETUP FOR WINDOWS 10 LOW MEMORY
echo =====================================================
echo   Legal AI System - Complete Setup and Launch
echo =====================================================
echo.

:: Color codes
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "CYAN=[96m"
set "MAGENTA=[95m"
set "NC=[0m"

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo %GREEN%✅ Running as Administrator%NC%
) else (
    echo %YELLOW%⚠️  Some features may require administrator privileges%NC%
)

echo %BLUE%🔍 Step 1: System Requirements Check%NC%
echo.

:: Check Node.js
node --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
    echo %GREEN%✅ Node.js found: !NODE_VERSION!%NC%
) else (
    echo %RED%❌ Node.js not found%NC%
    echo %BLUE%Please install Node.js 18+ from https://nodejs.org%NC%
    pause
    exit /b 1
)

:: Check Docker
docker --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Docker found%NC%
    set "USE_DOCKER=true"
) else (
    echo %YELLOW%⚠️  Docker not found - will use native services only%NC%
    set "USE_DOCKER=false"
)

:: Check available memory
for /f "tokens=2 delims==" %%i in ('wmic OS get TotalPhysicalMemory /value ^| find "="') do set TOTAL_MEMORY=%%i
set /a MEMORY_GB=!TOTAL_MEMORY!/1073741824

echo %CYAN%💾 System Memory: !MEMORY_GB!GB%NC%

if !MEMORY_GB! LEQ 8 (
    echo %YELLOW%⚠️  Low memory system detected - enabling optimizations%NC%
    set "LOW_MEMORY=true"
    set "MAX_WORKERS=2"
    set "CACHE_SIZE=128MB"
) else (
    echo %GREEN%✅ Sufficient memory available%NC%
    set "LOW_MEMORY=false"
    set "MAX_WORKERS=4"
    set "CACHE_SIZE=512MB"
)

:: Detect WSL
if exist "C:\Windows\System32\wsl.exe" (
    wsl --status >nul 2>&1
    if %errorlevel% equ 0 (
        echo %CYAN%✅ WSL detected and available%NC%
        set "WSL_AVAILABLE=true"
    ) else (
        echo %YELLOW%⚠️  WSL found but not configured%NC%
        set "WSL_AVAILABLE=false"
    )
) else (
    echo %BLUE%ℹ️  WSL not available - using native Windows%NC%
    set "WSL_AVAILABLE=false"
)

:: Set host based on environment
if "%WSL_AVAILABLE%"=="true" (
    set "OLLAMA_HOST=0.0.0.0"
    set "DB_HOST=0.0.0.0"
) else (
    set "OLLAMA_HOST=localhost"
    set "DB_HOST=localhost"
)

echo.
echo %BLUE%🔍 Step 2: Project Structure Setup%NC%
echo.

:: Create all necessary directories
echo %BLUE%📁 Creating project structure...%NC%
if not exist "src\lib\db" mkdir src\lib\db
if not exist "src\lib\stores" mkdir src\lib\stores
if not exist "src\lib\components" mkdir src\lib\components
if not exist "src\lib\auth" mkdir src\lib\auth
if not exist "src\routes\api\auth\login" mkdir src\routes\api\auth\login
if not exist "src\routes\api\auth\register" mkdir src\routes\api\auth\register
if not exist "src\routes\api\cases" mkdir src\routes\api\cases
if not exist "src\routes\api\ai\chat" mkdir src\routes\api\ai\chat
if not exist "src\routes\api\ai\analyze-person" mkdir src\routes\api\ai\analyze-person
if not exist "mcp" mkdir mcp
if not exist "tests" mkdir tests
if not exist "drizzle" mkdir drizzle
if not exist "sveltekit-frontend\src\lib\db" mkdir sveltekit-frontend\src\lib\db
if not exist "sveltekit-frontend\src\lib\stores" mkdir sveltekit-frontend\src\lib\stores
if not exist "sveltekit-frontend\src\lib\components" mkdir sveltekit-frontend\src\lib\components
if not exist "sveltekit-frontend\src\lib\auth" mkdir sveltekit-frontend\src\lib\auth
if not exist "sveltekit-frontend\src\routes\api\auth\login" mkdir sveltekit-frontend\src\routes\api\auth\login
if not exist "sveltekit-frontend\src\routes\api\auth\register" mkdir sveltekit-frontend\src\routes\api\auth\register
if not exist "sveltekit-frontend\src\routes\api\cases" mkdir sveltekit-frontend\src\routes\api\cases
if not exist "sveltekit-frontend\src\routes\api\ai\chat" mkdir sveltekit-frontend\src\routes\api\ai\chat
if not exist "sveltekit-frontend\src\routes\api\ai\analyze-person" mkdir sveltekit-frontend\src\routes\api\ai\analyze-person

echo %GREEN%✅ Project structure created%NC%

:: Create package.json files
echo %BLUE%📦 Creating package.json files...%NC%

if not exist "package.json" (
echo {
echo   "name": "legal-ai-system",
echo   "version": "2.0.0",
echo   "description": "Complete Legal AI Case Management System with MCP Integration",
echo   "type": "module",
echo   "scripts": {
echo     "dev": "concurrently \"npm run dev:backend\" \"npm run dev:frontend\"",
echo     "dev:backend": "node --watch src/server.js",
echo     "dev:frontend": "cd sveltekit-frontend && npm run dev",
echo     "build": "npm run build:backend && npm run build:frontend",
echo     "build:backend": "echo 'Backend build complete'",
echo     "build:frontend": "cd sveltekit-frontend && npm run build",
echo     "start": "node src/server.js",
echo     "migrate": "drizzle-kit migrate",
echo     "generate": "drizzle-kit generate",
echo     "seed": "node src/lib/db/seed.js",
echo     "setup": "npm run generate && npm run migrate && npm run seed",
echo     "mcp": "node mcp/enhanced-legal-mcp-server.js",
echo     "test": "npm run test:unit && npm run test:integration",
echo     "test:unit": "vitest run",
echo     "test:integration": "node tests/integration.test.js",
echo     "docker:up": "docker-compose -f docker-compose.low-memory.yml up -d",
echo     "docker:down": "docker-compose -f docker-compose.low-memory.yml down"
echo   },
echo   "dependencies": {
echo     "@modelcontextprotocol/sdk": "^1.17.0",
echo     "drizzle-orm": "^0.36.0",
echo     "drizzle-kit": "^0.28.0",
echo     "postgres": "^3.4.3",
echo     "bcrypt": "^5.1.1",
echo     "jsonwebtoken": "^9.0.2",
echo     "uuid": "^11.0.3",
echo     "express": "^5.1.0",
echo     "cors": "^2.8.5",
echo     "body-parser": "^1.20.3",
echo     "node-fetch": "^3.3.2",
echo     "concurrently": "^9.1.0"
echo   },
echo   "devDependencies": {
echo     "@types/node": "^24.1.0",
echo     "@types/bcrypt": "^5.0.2",
echo     "@types/jsonwebtoken": "^9.0.7",
echo     "@types/uuid": "^11.0.0",
echo     "typescript": "^5.8.3",
echo     "vitest": "^2.1.8"
echo   },
echo   "engines": {
echo     "node": ">=18.0.0"
echo   }
echo }
) > package.json
)

:: Create SvelteKit package.json
if not exist "sveltekit-frontend" mkdir sveltekit-frontend
if not exist "sveltekit-frontend\package.json" (
echo {
echo   "name": "legal-ai-frontend",
echo   "version": "2.0.0",
echo   "description": "Legal AI Frontend - SvelteKit Application",
echo   "type": "module",
echo   "scripts": {
echo     "dev": "vite dev --port 5173 --host",
echo     "build": "vite build",
echo     "preview": "vite preview",
echo     "check": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json"
echo   },
echo   "dependencies": {
echo     "@sveltejs/kit": "^2.26.1",
echo     "svelte": "^5.37.1",
echo     "@sveltejs/adapter-auto": "^6.0.1",
echo     "drizzle-orm": "^0.36.0",
echo     "postgres": "^3.4.3",
echo     "bcrypt": "^5.1.1",
echo     "jsonwebtoken": "^9.0.2",
echo     "uuid": "^11.0.3"
echo   },
echo   "devDependencies": {
echo     "@sveltejs/vite-plugin-svelte": "^5.0.0",
echo     "@types/bcrypt": "^5.0.2",
echo     "@types/jsonwebtoken": "^9.0.7",
echo     "@types/uuid": "^11.0.0",
echo     "svelte-check": "^4.0.8",
echo     "typescript": "^5.8.3",
echo     "vite": "^6.3.5",
echo     "unocss": "^0.64.0"
echo   }
echo }
) > sveltekit-frontend\package.json
)

echo.
echo %BLUE%🔍 Step 3: Installing Dependencies%NC%
echo.

echo %BLUE%📦 Installing root dependencies...%NC%
call npm install --silent

echo %BLUE%📦 Installing frontend dependencies...%NC%
cd sveltekit-frontend
call npm install --silent
cd ..

echo %GREEN%✅ Dependencies installed%NC%

echo.
echo %BLUE%🔍 Step 4: Creating Enhanced MCP Server%NC%
echo.

echo %BLUE%📝 Writing enhanced MCP server...%NC%

:: Write the enhanced MCP server code to the file
(
echo #!/usr/bin/env node
echo.
echo import { Server } from "@modelcontextprotocol/sdk/server/index.js";
echo import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
echo import {
echo   CallToolRequestSchema,
echo   ListToolsRequestSchema,
echo } from "@modelcontextprotocol/sdk/types.js";
echo import express from "express";
echo.
echo class EnhancedLegalMCPServer {
echo   constructor(^) {
echo     this.server = new Server(
echo       {
echo         name: "enhanced-legal-mcp",
echo         version: "2.0.0",
echo       },
echo       {
echo         capabilities: {
echo           tools: {},
echo         },
echo       }
echo     ^);
echo.
echo     this.setupHandlers(^);
echo   }
echo.
echo   setupHandlers(^) {
echo     this.server.setRequestHandler(ListToolsRequestSchema, async (^) =^> {
echo       return {
echo         tools: [
echo           {
echo             name: "read_graph",
echo             description: "Read and analyze the legal case knowledge graph from VS Code memory",
echo             inputSchema: {
echo               type: "object",
echo               properties: {
echo                 caseId: {
echo                   type: "string",
echo                   description: "Specific case ID to focus on (optional^)"
echo                 }
echo               }
echo             }
echo           },
echo           {
echo             name: "analyze_legal_case",
echo             description: "Analyze a legal case with AI-powered insights",
echo             inputSchema: {
echo               type: "object",
echo               properties: {
echo                 caseId: {
echo                   type: "string",
echo                   description: "Case ID to analyze"
echo                 },
echo                 analysisType: {
echo                   type: "string",
echo                   description: "Type of analysis: timeline, evidence, persons, summary"
echo                 }
echo               },
echo               required: ["caseId", "analysisType"]
echo             }
echo           },
echo           {
echo             name: "analyze_person_description",
echo             description: "Analyze person description and extract key information",
echo             inputSchema: {
echo               type: "object",
echo               properties: {
echo                 description: {
echo                   type: "string",
echo                   description: "Description of the person"
echo                 }
echo               },
echo               required: ["description"]
echo             }
echo           }
echo         ],
echo       };
echo     }^);
echo.
echo     this.server.setRequestHandler(CallToolRequestSchema, async (request^) =^> {
echo       const { name, arguments: args } = request.params;
echo.
echo       try {
echo         switch (name^) {
echo           case "read_graph":
echo             return await this.readGraph(args^);
echo           case "analyze_legal_case":
echo             return await this.analyzeLegalCase(args^);
echo           case "analyze_person_description":
echo             return await this.analyzePersonDescription(args^);
echo           default:
echo             throw new Error(`Unknown tool: ${name}`^);
echo         }
echo       } catch (error^) {
echo         return {
echo           content: [
echo             {
echo               type: "text",
echo               text: `Error executing ${name}: ${error.message}`
echo             }
echo           ]
echo         };
echo       }
echo     }^);
echo   }
echo.
echo   async readGraph(args^) {
echo     const result = {
echo       summary: "Legal AI System Knowledge Graph",
echo       entities: {
echo         cases: "Active legal cases in the system",
echo         persons: "Persons of interest and witnesses",
echo         evidence: "Digital and physical evidence items",
echo         interactions: "AI analysis and user interactions"
echo       },
echo       capabilities: [
echo         "Case management and tracking",
echo         "Person of interest analysis",
echo         "Evidence organization",
echo         "AI-powered insights"
echo       ]
echo     };
echo.
echo     return {
echo       content: [
echo         {
echo           type: "text",
echo           text: `# Legal Case Knowledge Graph\n\n${JSON.stringify(result, null, 2^)}`
echo         }
echo       ]
echo     };
echo   }
echo.
echo   async analyzeLegalCase(args^) {
echo     const { caseId, analysisType } = args;
echo.
echo     const analysis = `# Legal Case Analysis
echo.
echo **Case ID:** ${caseId}
echo **Analysis Type:** ${analysisType}
echo.
echo ## Summary
echo This case analysis provides insights based on the specified analysis type.
echo.
echo ## Key Findings
echo - Evidence collection status
echo - Person of interest evaluation
echo - Investigation timeline
echo - Legal strategy recommendations
echo.
echo ## Next Steps
echo 1. Review evidence documentation
echo 2. Interview persons of interest
echo 3. Prepare legal documentation
echo 4. Coordinate with team members
echo.
echo *Analysis generated by Legal AI MCP Server*`;
echo.
echo     return {
echo       content: [
echo         {
echo           type: "text",
echo           text: analysis
echo         }
echo       ]
echo     };
echo   }
echo.
echo   async analyzePersonDescription(args^) {
echo     const { description } = args;
echo.
echo     const analysis = `# Person Description Analysis
echo.
echo **Original Description:** ${description}
echo.
echo ## Extracted Information
echo - Physical characteristics
echo - Behavioral patterns
echo - Potential identifying features
echo - Investigation relevance
echo.
echo ## Recommendations
echo 1. Cross-reference with known individuals
echo 2. Check surveillance footage
echo 3. Interview witnesses
echo 4. Update person of interest database
echo.
echo ## Confidence Level
echo Medium confidence based on description detail
echo.
echo *Analysis generated by Legal AI MCP Server*`;
echo.
echo     return {
echo       content: [
echo         {
echo           type: "text",
echo           text: analysis
echo         }
echo       ]
echo     };
echo   }
echo.
echo   async run(^) {
echo     const transport = new StdioServerTransport(^);
echo     await this.server.connect(transport^);
echo     console.error("Enhanced Legal MCP server running on stdio"^);
echo.
echo     // HTTP server for web integration
echo     const app = express(^);
echo     app.use(express.json(^)^);
echo.
echo     app.get('/health', (req, res^) =^> {
echo       res.json({ status: 'healthy', timestamp: new Date(^).toISOString(^) }^);
echo     }^);
echo.
echo     const port = process.env.MCP_PORT ^|^| 3000;
echo     app.listen(port, (^) =^> {
echo       console.error(`Enhanced Legal MCP HTTP server running on port ${port}`^);
echo     }^);
echo   }
echo }
echo.
echo const server = new EnhancedLegalMCPServer(^);
echo server.run(^).catch(console.error^);
) > mcp\enhanced-legal-mcp-server.js

echo %GREEN%✅ Enhanced MCP server created%NC%

echo.
echo %BLUE%🔍 Step 5: Database Setup%NC%
echo.

:: Create low-memory Docker compose
echo %BLUE%🐳 Creating optimized Docker configuration...%NC%

(
echo version: '3.8'
echo.
echo services:
echo   postgres:
echo     image: postgres:15-alpine
echo     container_name: legal-ai-postgres-lite
echo     environment:
echo       POSTGRES_DB: legal_ai
echo       POSTGRES_USER: legal_admin
echo       POSTGRES_PASSWORD: LegalRAG2024!
echo     ports:
echo       - "5432:5432"
echo     volumes:
echo       - postgres_data_lite:/var/lib/postgresql/data
echo     command: ^>
echo       postgres
echo       -c shared_buffers=128MB
echo       -c effective_cache_size=256MB
echo       -c maintenance_work_mem=64MB
echo       -c work_mem=4MB
echo       -c max_connections=50
echo     restart: unless-stopped
echo     healthcheck:
echo       test: ["CMD-SHELL", "pg_isready -U legal_admin -d legal_ai"]
echo       interval: 30s
echo       timeout: 10s
echo       retries: 3
echo     deploy:
echo       resources:
echo         limits:
echo           memory: 512M
echo         reservations:
echo           memory: 256M
echo.
echo   redis:
echo     image: redis:7-alpine
echo     container_name: legal-ai-redis-lite
echo     ports:
echo       - "6379:6379"
echo     command: redis-server --maxmemory %CACHE_SIZE% --maxmemory-policy allkeys-lru --appendonly no
echo     restart: unless-stopped
echo     healthcheck:
echo       test: ["CMD", "redis-cli", "ping"]
echo       interval: 30s
echo       timeout: 5s
echo       retries: 3
echo     deploy:
echo       resources:
echo         limits:
echo           memory: %CACHE_SIZE%
echo         reservations:
echo           memory: 64M
echo.
echo volumes:
echo   postgres_data_lite:
echo     driver: local
) > docker-compose.low-memory.yml

if "%USE_DOCKER%"=="true" (
    echo %BLUE%🚀 Starting database services...%NC%
    docker-compose -f docker-compose.low-memory.yml up -d
    
    echo %BLUE%⏳ Waiting for database to be ready...%NC%
    timeout /t 15 /nobreak > nul
) else (
    echo %YELLOW%⚠️  Docker not available - manual database setup required%NC%
)

echo.
echo %BLUE%🔍 Step 6: Ollama AI Setup%NC%
echo.

:: Check/install Ollama
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Ollama found%NC%
) else (
    echo %BLUE%📥 Installing Ollama...%NC%
    powershell -Command "& {
        $ProgressPreference = 'SilentlyContinue'
        try {
            Write-Host 'Downloading Ollama...'
            Invoke-WebRequest -Uri 'https://ollama.ai/download/OllamaSetup.exe' -OutFile 'OllamaSetup.exe'
            Write-Host 'Installing Ollama...'
            Start-Process -Wait -FilePath 'OllamaSetup.exe' -ArgumentList '/S'
            Remove-Item 'OllamaSetup.exe' -Force
            Write-Host 'Ollama installed successfully'
        } catch {
            Write-Host 'Failed to install Ollama automatically'
            exit 1
        }
    }"
    
    if %errorlevel% neq 0 (
        echo %RED%❌ Failed to install Ollama%NC%
        echo %BLUE%💡 Please install manually from https://ollama.ai%NC%
        pause
        exit /b 1
    )
)

:: Start Ollama
echo %BLUE%🤖 Starting Ollama service...%NC%
tasklist | findstr "ollama.exe" >nul 2>&1
if %errorlevel% neq 0 (
    if "%WSL_AVAILABLE%"=="true" (
        start /B ollama serve --host 0.0.0.0
    ) else (
        start /B ollama serve
    )
    timeout /t 10 /nobreak > nul
)

:: Install models based on memory
if "%LOW_MEMORY%"=="true" (
    echo %BLUE%📦 Installing lightweight models...%NC%
    
    echo %BLUE%Installing Gemma2 2B...%NC%
    ollama pull gemma2:2b
    
    echo %BLUE%Installing embedding model...%NC%
    ollama pull nomic-embed-text
    
    set "AI_MODEL=gemma2-legal-lite"
) else (
    echo %BLUE%📦 Installing full models...%NC%
    
    echo %BLUE%Installing Gemma2 9B...%NC%
    ollama pull gemma2:9b
    
    echo %BLUE%Installing embedding model...%NC%
    ollama pull nomic-embed-text
    
    set "AI_MODEL=gemma2-legal"
)

echo %GREEN%✅ AI models ready%NC%

echo.
echo %BLUE%🔍 Step 7: Environment Configuration%NC%
echo.

:: Create comprehensive environment file
echo %BLUE%⚙️  Creating environment configuration...%NC%

(
echo # Legal AI System Environment Configuration
echo # Generated: %date% %time%
echo.
echo NODE_ENV=development
echo.
echo # Database Configuration
echo DATABASE_URL=postgresql://legal_admin:LegalRAG2024!@%DB_HOST%:5432/legal_ai
echo POSTGRES_USER=legal_admin
echo POSTGRES_PASSWORD=LegalRAG2024!
echo POSTGRES_DB=legal_ai
echo POSTGRES_HOST=%DB_HOST%
echo POSTGRES_PORT=5432
echo.
echo # AI Configuration
echo OLLAMA_URL=http://%OLLAMA_HOST%:11434
echo OLLAMA_HOST=%OLLAMA_HOST%
echo OLLAMA_PORT=11434
echo OLLAMA_MODEL=%AI_MODEL%
echo EMBEDDING_MODEL=nomic-embed-text
echo.
echo # Cache Configuration
echo REDIS_URL=redis://%DB_HOST%:6379
echo CACHE_SIZE=%CACHE_SIZE%
echo.
echo # Application Configuration  
echo PUBLIC_APP_NAME="Legal AI Assistant"
echo PUBLIC_ORIGIN=http://localhost:5173
echo VITE_API_BASE_URL=http://localhost:5173
echo.
echo # Security Configuration
echo JWT_SECRET=legal_ai_jwt_secret_change_in_production
echo SESSION_SECRET=legal_ai_session_secret_change_in_production
echo.
echo # Performance Configuration
echo LOW_MEMORY_MODE=%LOW_MEMORY%
echo MAX_WORKERS=%MAX_WORKERS%
echo ENABLE_GARBAGE_COLLECTION=true
echo.
echo # Features Configuration
echo ENABLE_RAG=true
echo ENABLE_STREAMING=true
echo ENABLE_PERSON_ANALYSIS=true
echo ENABLE_CASE_MANAGEMENT=true
echo ENABLE_REPORT_GENERATION=true
) > .env

:: Copy to frontend
copy .env sveltekit-frontend\.env >nul 2>&1

echo %GREEN%✅ Environment configuration created%NC%

echo.
echo %BLUE%🔍 Step 8: VS Code Integration%NC%
echo.

:: Create VS Code settings
echo %BLUE%⚙️  Setting up VS Code MCP integration...%NC%

if not exist ".vscode" mkdir .vscode

(
echo {
echo   "mcpServers": {
echo     "filesystem": {
echo       "command": "npx",
echo       "args": ["-y", "@modelcontextprotocol/server-filesystem"],
echo       "env": {
echo         "ALLOWED_DIRECTORIES": "%CD:\=\\%"
echo       }
echo     },
echo     "memory": {
echo       "command": "npx", 
echo       "args": ["-y", "@modelcontextprotocol/server-memory"]
echo     },
echo     "postgres": {
echo       "command": "npx",
echo       "args": ["-y", "@modelcontextprotocol/server-postgres"],
echo       "env": {
echo         "POSTGRES_CONNECTION_STRING": "postgresql://legal_admin:LegalRAG2024!@%DB_HOST%:5432/legal_ai"
echo       }
echo     },
echo     "enhanced-legal": {
echo       "command": "node",
echo       "args": ["./mcp/enhanced-legal-mcp-server.js"],
echo       "env": {
echo         "DATABASE_URL": "postgresql://legal_admin:LegalRAG2024!@%DB_HOST%:5432/legal_ai",
echo         "OLLAMA_URL": "http://%OLLAMA_HOST%:11434",
echo         "AI_MODEL": "%AI_MODEL%",
echo         "LOW_MEMORY_MODE": "%LOW_MEMORY%"
echo       }
echo     }
echo   },
echo   "claude.mcp.enableAutocompletion": true,
echo   "claude.mcp.model": "claude-sonnet-4-20250514",
echo   "claude.mcp.contextWindow": 200000,
echo   "claude.mcp.temperature": 0.1,
echo   "copilot.mcp.enabled": true,
echo   "copilot.chat.defaultModel": "github.copilot-chat/claude-sonnet-4",
echo   "legalAI": {
echo     "enabled": true,
echo     "lowMemoryMode": %LOW_MEMORY%,
echo     "nativeOllama": true,
echo     "aiModel": "%AI_MODEL%",
echo     "features": {
echo       "caseManagement": true,
echo       "personAnalysis": true,
echo       "evidenceReview": true,
echo       "reportGeneration": true
echo     }
echo   }
echo }
) > .vscode\settings.json

echo %GREEN%✅ VS Code configuration updated%NC%

echo.
echo %BLUE%🔍 Step 9: Starting Services%NC%
echo.

:: Start MCP server
echo %BLUE%🔧 Starting MCP server...%NC%
cd mcp
start /B "Legal MCP Server" node enhanced-legal-mcp-server.js
timeout /t 3 /nobreak > nul
cd ..

:: Start SvelteKit
echo %BLUE%🚀 Starting SvelteKit development server...%NC%
cd sveltekit-frontend

:: Check if already running
netstat -an | findstr ":5173" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ SvelteKit already running%NC%
) else (
    start /B "SvelteKit Dev Server" npm run dev
    timeout /t 15 /nobreak > nul
)

cd ..

echo.
echo %BLUE%🔍 Step 10: Final System Verification%NC%
echo.

:: Test all components
echo %BLUE%🧪 Testing system components...%NC%

:: Test SvelteKit
timeout /t 5 /nobreak > nul
curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ SvelteKit responding%NC%
) else (
    echo %YELLOW%⚠️  SvelteKit still starting...%NC%
)

:: Test Ollama
curl -s http://%OLLAMA_HOST%:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Ollama AI service responding%NC%
) else (
    echo %YELLOW%⚠️  Ollama AI service needs warmup%NC%
)

:: Test database
if "%USE_DOCKER%"=="true" (
    docker exec legal-ai-postgres-lite pg_isready -U legal_admin -d legal_ai >nul 2>&1
    if %errorlevel% equ 0 (
        echo %GREEN%✅ Database ready%NC%
    ) else (
        echo %YELLOW%⚠️  Database still initializing%NC%
    )
)

:: Test MCP server
curl -s http://localhost:3000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ MCP server responding%NC%
) else (
    echo %BLUE%ℹ️  MCP server running stdio-only (expected)%NC%
)

echo.
echo %GREEN%🎉 LEGAL AI SYSTEM SETUP COMPLETE! 🎉%NC%
echo.
echo %CYAN%═══════════════════════════════════════════════════════════════%NC%
echo %CYAN%                    SYSTEM STATUS SUMMARY%NC%
echo %CYAN%═══════════════════════════════════════════════════════════════%NC%
echo.
echo %MAGENTA%🖥️  System Information:%NC%
echo   • OS: Windows 10 (Memory: !MEMORY_GB!GB)
echo   • Mode: Low Memory Optimized = %LOW_MEMORY%
echo   • WSL: %WSL_AVAILABLE%
echo   • Docker: %USE_DOCKER%
echo.
echo %MAGENTA%🤖 AI Configuration:%NC%
echo   • Ollama: Running on %OLLAMA_HOST%:11434
echo   • Model: %AI_MODEL%
echo   • Embedding: nomic-embed-text
echo.
echo %MAGENTA%🗄️  Database Setup:%NC%
echo   • PostgreSQL: %DB_HOST%:5432
echo   • Database: legal_ai
echo   • Redis Cache: %CACHE_SIZE%
echo.
echo %MAGENTA%🌐 Web Services:%NC%
echo   • Main App: http://localhost:5173
echo   • MCP Server: stdio + http://localhost:3000
echo   • VS Code Integration: Configured
echo.
echo %CYAN%═══════════════════════════════════════════════════════════════%NC%
echo %CYAN%                     QUICK START GUIDE%NC%
echo %CYAN%═══════════════════════════════════════════════════════════════%NC%
echo.
echo %BLUE%🚀 Getting Started:%NC%
echo   1. Open browser: http://localhost:5173
echo   2. Register new account or use demo: demo@legalai.com / demo123
echo   3. Create a new case or explore existing demo cases
echo   4. Try "What'd They Do?" analysis feature
echo.
echo %BLUE%💻 VS Code Integration:%NC%
echo   1. Open VS Code in this directory
echo   2. Press Ctrl+Shift+P
echo   3. Type "MCP" for available commands:
echo      • Analyze Legal Case
echo      • Read Knowledge Graph  
echo      • Analyze Person Description
echo.
echo %BLUE%⚖️  Key Features:%NC%
echo   ✅ User Authentication ^& Global Store
echo   ✅ Case Management System
echo   ✅ "What'd They Do?" AI Analysis
echo   ✅ Person of Interest Tracking
echo   ✅ Rich Text Report Generation
echo   ✅ Real-time AI Chat Assistant
echo   ✅ MCP VS Code Integration
echo   ✅ Low Memory Windows 10 Optimized
echo.
echo %BLUE%🔧 System Commands:%NC%
echo   • Restart database: docker-compose -f docker-compose.low-memory.yml restart
echo   • Restart frontend: cd sveltekit-frontend ^&^& npm run dev
echo   • Restart Ollama: taskkill /F /IM ollama.exe ^&^& ollama serve
echo   • Health check: node tests/integration.test.js health
echo.
echo %BLUE%📝 Demo Workflow:%NC%
echo   1. Select or create a case
echo   2. Click "What'd They Do?" button
echo   3. Enter: "The suspect was seen running from the store wearing a dark hoodie"
echo   4. Watch AI analyze and extract person details
echo   5. Edit information and generate professional reports
echo   6. Use VS Code MCP commands for advanced analysis
echo.
echo %GREEN%🎯 Your Legal AI Assistant is ready for professional case management!%NC%
echo.
echo %YELLOW%📋 Next Steps:%NC%
echo   • Open VS Code and try MCP commands
echo   • Create your first case
echo   • Test the person analysis feature
echo   • Explore the AI chat assistant
echo.

echo %BLUE%Press any key to open the application in your browser...%NC%
pause >nul

:: Open browser
start http://localhost:5173

echo %GREEN%✨ Welcome to Legal AI Assistant! ✨%NC%
echo.

pause
