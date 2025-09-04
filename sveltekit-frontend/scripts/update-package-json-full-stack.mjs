// Script to add Full Stack AI Synthesis commands to package.json
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function updatePackageJson() {
    const packageJsonPath = path.join(__dirname, '../package.json');

    try {
        // Read existing package.json
        const content = await fs.readFile(packageJsonPath, 'utf8');
        const packageJson = JSON.parse(content);

        // Add Full Stack AI Synthesis specific scripts
        const fullStackScripts = {
            // Full Stack Core Commands
            "ai:full-stack": "powershell -ExecutionPolicy Bypass -File scripts/orchestration/start-ai-synthesis-full-stack.ps1",
            "ai:full-stack:windows": "START-AI-SYNTHESIS-FULL-STACK.bat",
            "ai:full-stack:fast": "powershell -ExecutionPolicy Bypass -File scripts/orchestration/start-ai-synthesis-full-stack.ps1 -FastStart",

            // Database Management
            "db:neo4j:start": "C:\\neo4j-community-5.23.0\\bin\\neo4j.bat console",
            "db:neo4j:stop": "C:\\neo4j-community-5.23.0\\bin\\neo4j.bat stop",
            "db:pgvector:init": "psql -U postgres -d legal_ai -c \"CREATE EXTENSION IF NOT EXISTS vector;\"",
            "db:pgvector:indexes": "psql -U postgres -d legal_ai -f scripts/create-pgvector-indexes.sql",

            // Go Microservices
            "go:rag": "cd go-microservice/cmd/enhanced-rag-v2-local && go run main.go",
            "go:gpu": "cd go-microservice && go run ./cmd/gpu-orchestrator",
            "go:llama": "cd go-microservice && go run go-llama-chat-service.go",
            "go:all": "concurrently \"npm run go:rag\" \"npm run go:gpu\" \"npm run go:llama\"",

            // Ollama Model Management
            "ollama:create-legal": "ollama create gemma3:legal-latest -f Modelfile-legal",
            "ollama:pull-nomic": "ollama pull nomic-embed-text",
            "ollama:pull-gemma": "ollama pull gemma2:2b",
            "ollama:setup": "npm run ollama:pull-gemma && npm run ollama:pull-nomic && npm run ollama:create-legal",

            // MCP Servers
            "mcp:context7": "node mcp-servers/context7-server.js",
            "mcp:synthesis": "node mcp-servers/ai-synthesis-mcp.js",
            "mcp:all": "concurrently \"npm run mcp:context7\" \"npm run mcp:synthesis\"",

            // Full Stack Development
            "dev:full-stack": "concurrently -n \"Frontend,Neo4j,PG,Redis,Ollama,Go,MCP\" -c \"cyan,green,blue,red,magenta,yellow,white\" \"npm run dev\" \"npm run db:neo4j:start\" \"npm run synthesis:cache:start\" \"npm run synthesis:ollama:start\" \"npm run go:all\" \"npm run mcp:all\"",
            "dev:ai:complete": "npm run ai:full-stack:windows && npm run dev",

            // Testing Full Stack
            "test:full-stack": "node scripts/test-full-stack-integration.mjs",
            "test:neo4j": "node scripts/test-neo4j-connection.mjs",
            "test:pgvector": "node scripts/test-pgvector-search.mjs",
            "test:orchestrator": "vitest run src/lib/server/ai/enhanced-ai-synthesis-orchestrator.test.ts",

            // Health Checks
            "health:full-stack": "curl http://localhost:5173/api/ai-synthesizer/health",
            "health:services": "node scripts/check-all-services.mjs",

            // Monitoring
            "monitor:full-stack": "powershell -ExecutionPolicy Bypass -File scripts/orchestration/monitor-ai-synthesis.ps1",
            "monitor:gpu": "nvidia-smi -l 1",
            "monitor:services": "node scripts/monitor-services.mjs",

            // AutoSolve Enhanced
            "autosolve:full-stack": "node scripts/autosolve-full-stack.mjs",
            "autosolve:neo4j": "node scripts/autosolve-neo4j-queries.mjs",
            "autosolve:legal": "node scripts/autosolve-legal-analysis.mjs",

            // Production
            "build:full-stack": "npm run check:all && npm run test:full-stack && vite build",
            "start:production": "NODE_ENV=production node build",
            "deploy:windows-service": "powershell -ExecutionPolicy Bypass -File scripts/install-windows-services.ps1"
        };

        // Merge scripts
        packageJson.scripts = {
            ...packageJson.scripts,
            ...fullStackScripts
        };

        // Sort scripts alphabetically for better organization
        const sortedScripts = {};
        Object.keys(packageJson.scripts).sort().forEach(key => {
            sortedScripts[key] = packageJson.scripts[key];
        });
        packageJson.scripts = sortedScripts;

        // Write updated package.json
        await fs.writeFile(
            packageJsonPath,
            JSON.stringify(packageJson, null, 2) + '\n',
            'utf8'
        );

        console.log('✅ package.json updated with Full Stack AI Synthesis commands');
        console.log('\n📝 Key commands added:');
        console.log('  npm run ai:full-stack         - Start complete system with PowerShell');
        console.log('  npm run ai:full-stack:windows - Start with Windows batch file');
        console.log('  npm run dev:full-stack        - Development with all services');
        console.log('  npm run test:full-stack       - Run integration tests');
        console.log('  npm run monitor:full-stack    - Launch monitoring dashboard');
        console.log('  npm run health:full-stack     - Check system health');
        console.log('\n🎯 Service-specific commands:');
        console.log('  npm run db:neo4j:start        - Start Neo4j graph database');
        console.log('  npm run go:all                - Start all Go microservices');
        console.log('  npm run ollama:setup          - Setup Ollama models');
        console.log('  npm run mcp:all               - Start all MCP servers');

    } catch (error) {
        console.error('❌ Error updating package.json:', error);
        process.exit(1);
    }
}

// Run the update
updatePackageJson();
