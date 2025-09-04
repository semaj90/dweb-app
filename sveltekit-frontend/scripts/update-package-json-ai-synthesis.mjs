// Script to add AI Synthesis commands to package.json
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
        
        // Add AI Synthesis specific scripts
        const aiSynthesisScripts = {
            // AI Synthesis Core
            "ai:start": "powershell -ExecutionPolicy Bypass -File scripts/orchestration/start-ai-synthesis.ps1",
            "ai:start:windows": "START-AI-SYNTHESIS-WINDOWS.bat",
            "ai:monitor": "powershell -ExecutionPolicy Bypass -File scripts/orchestration/monitor-ai-synthesis.ps1",
            "ai:test": "curl http://localhost:5173/api/ai-synthesizer/test",
            "ai:health": "curl http://localhost:5173/api/ai-synthesizer/health",
            
            // AutoSolve for AI Synthesis
            "autosolve:ai-synthesis": "node scripts/autosolve-ai-synthesis.mjs",
            "autosolve:check": "npm run check:ultra-fast && npm run autosolve:ai-synthesis",
            "autosolve:full": "npm run autosolve:all && npm run autosolve:ai-synthesis",
            
            // Service Management
            "synthesis:cache:start": "redis-server --port 6379 --maxmemory 512mb",
            "synthesis:cache:flush": "redis-cli -p 6379 FLUSHALL",
            "synthesis:ollama:start": "ollama serve",
            "synthesis:ollama:pull": "ollama pull llama2 && ollama pull mistral",
            "synthesis:ollama:legal": "node scripts/create-legal-models.mjs",
            
            // Development with AI Synthesis
            "dev:ai": "concurrently -n \"Dev,Redis,Ollama,Monitor\" -c \"cyan,red,magenta,yellow\" \"npm run dev\" \"npm run synthesis:cache:start\" \"npm run synthesis:ollama:start\" \"npm run ai:monitor\"",
            "dev:ai:full": "npm run ai:start:windows && npm run dev",
            
            // Testing
            "test:ai-synthesis": "vitest run src/lib/server/ai/*.test.ts",
            "test:ai-integration": "node scripts/test-ai-integration.mjs",
            
            // MCP Integration
            "mcp:ai-synthesis": "node mcp-servers/ai-synthesis-mcp.js",
            "mcp:health:all": "node scripts/check-mcp-health.mjs"
        };
        
        // Merge scripts
        packageJson.scripts = {
            ...packageJson.scripts,
            ...aiSynthesisScripts
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
        
        console.log('✅ package.json updated with AI Synthesis commands');
        console.log('\n📝 New commands added:');
        Object.keys(aiSynthesisScripts).forEach(cmd => {
            console.log(`  npm run ${cmd}`);
        });
        
    } catch (error) {
        console.error('❌ Error updating package.json:', error);
        process.exit(1);
    }
}

// Run the update
updatePackageJson();
