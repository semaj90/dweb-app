// aggregate-server.cjs - REST API for Autosolve System
const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const { autosolve } = require('./autosolve-runner.cjs');

const PORT = process.env.AGGREGATE_PORT || 8123;
const CONFIG = {
    historyFile: 'logs/autosolve-history.jsonl',
    recommendationsFile: 'logs/recommendations-aggregate.json',
    errorCacheFile: 'logs/error-cache.json',
    maxHistoryLines: 100
};

// Middleware for CORS
function setCorsHeaders(res) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

// Read last N lines from history
function getRecentHistory(n = 10) {
    if (!fs.existsSync(CONFIG.historyFile)) {
        return [];
    }
    
    const content = fs.readFileSync(CONFIG.historyFile, 'utf8');
    const lines = content.trim().split('\n').filter(Boolean);
    
    return lines.slice(-n).map(line => {
        try {
            return JSON.parse(line);
        } catch {
            return null;
        }
    }).filter(Boolean);
}

// Get current TypeScript errors
function getCurrentErrors() {
    try {
        execSync('npx tsc --noEmit', { stdio: 'pipe' });
        return { count: 0, errors: [] };
    } catch (err) {
        const output = err.stdout?.toString() || '';
        const errors = [];
        const lines = output.split('\n');
        const errorPattern = /^(.+?)\((\d+),(\d+)\):\s+error\s+(\w+):\s+(.+)$/;
        
        for (const line of lines) {
            const match = line.match(errorPattern);
            if (match) {
                errors.push({
                    file: match[1],
                    line: parseInt(match[2]),
                    column: parseInt(match[3]),
                    code: match[4],
                    message: match[5]
                });
            }
        }
        
        // Cache errors
        fs.writeFileSync(CONFIG.errorCacheFile, JSON.stringify(errors, null, 2));
        
        return { count: errors.length, errors };
    }
}

// Get recommendations summary
function getRecommendations() {
    if (!fs.existsSync(CONFIG.recommendationsFile)) {
        return { recommendations: [], generated: null };
    }
    
    try {
        const data = JSON.parse(fs.readFileSync(CONFIG.recommendationsFile, 'utf8'));
        return data;
    } catch {
        return { recommendations: [], generated: null };
    }
}

// Get system status
function getSystemStatus() {
    const status = {
        services: {},
        errors: getCurrentErrors(),
        history: getRecentHistory(5),
        recommendations: getRecommendations()
    };
    
    // Check service status
    const services = [
        { name: 'PostgreSQL', port: 5432 },
        { name: 'Redis', port: 6379 },
        { name: 'RabbitMQ', port: 5672 },
        { name: 'Enhanced RAG', port: 8097 },
        { name: 'Ollama', port: 11434 }
    ];
    
    services.forEach(service => {
        // Simple port check (would need actual implementation)
        status.services[service.name] = {
            port: service.port,
            status: 'unknown' // Would need actual port checking
        };
    });
    
    return status;
}

// Trigger autosolve asynchronously
function triggerAutosolve(callback) {
    const child = spawn('node', [path.join(__dirname, 'autosolve-runner.cjs')], {
        detached: false,
        stdio: 'pipe'
    });
    
    let output = '';
    let error = '';
    
    child.stdout.on('data', (data) => {
        output += data.toString();
    });
    
    child.stderr.on('data', (data) => {
        error += data.toString();
    });
    
    child.on('close', (code) => {
        callback({
            success: code === 0,
            exitCode: code,
            output,
            error
        });
    });
    
    return child.pid;
}

// Request handler
async function handleRequest(req, res) {
    setCorsHeaders(res);
    
    const url = new URL(req.url, `http://localhost:${PORT}`);
    const pathname = url.pathname;
    
    // Handle OPTIONS for CORS
    if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
    }
    
    // Routes
    switch (pathname) {
        case '/health':
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ 
                status: 'healthy', 
                timestamp: new Date().toISOString(),
                port: PORT
            }));
            break;
            
        case '/aggregate':
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                timestamp: new Date().toISOString(),
                status: getSystemStatus(),
                summary: {
                    totalErrors: getCurrentErrors().count,
                    recentFixes: getRecentHistory(10).filter(h => h.action === 'complete').length,
                    recommendations: getRecommendations().recommendations.length
                }
            }));
            break;
            
        case '/errors':
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(getCurrentErrors()));
            break;
            
        case '/history':
            const limit = parseInt(url.searchParams.get('limit') || '20');
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(getRecentHistory(limit)));
            break;
            
        case '/recommendations':
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(getRecommendations()));
            break;
            
        case '/autosolve/trigger':
            if (req.method !== 'POST') {
                res.writeHead(405, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Method not allowed' }));
                break;
            }
            
            console.log('Triggering autosolve...');
            const pid = triggerAutosolve((result) => {
                console.log(`Autosolve completed with exit code: ${result.exitCode}`);
                
                // Store result
                const resultFile = `logs/autosolve-result-${Date.now()}.json`;
                fs.writeFileSync(resultFile, JSON.stringify(result, null, 2));
            });
            
            res.writeHead(202, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                status: 'triggered',
                pid,
                message: 'Autosolve process started in background'
            }));
            break;
            
        case '/autosolve/status':
            // Check if autosolve is running
            try {
                const processes = execSync('tasklist /FI "IMAGENAME eq node.exe"', { encoding: 'utf8' });
                const isRunning = processes.includes('autosolve-runner.cjs');
                
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    running: isRunning,
                    lastRun: getRecentHistory(1)[0] || null
                }));
            } catch {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ running: false, lastRun: null }));
            }
            break;
            
        case '/ollama/summary':
            if (req.method !== 'POST') {
                res.writeHead(405, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Method not allowed' }));
                break;
            }
            
            // Generate summary using Ollama
            let body = '';
            req.on('data', chunk => { body += chunk; });
            req.on('end', async () => {
                try {
                    const data = JSON.parse(body);
                    const errors = data.errors || getCurrentErrors().errors;
                    
                    // Call Ollama for summary
                    const fetch = require('node-fetch');
                    const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model: 'gemma3:latest',
                            prompt: `Summarize these TypeScript errors and suggest fixes:\n${JSON.stringify(errors.slice(0, 10), null, 2)}`,
                            stream: false
                        })
                    });
                    
                    if (ollamaResponse.ok) {
                        const result = await ollamaResponse.json();
                        res.writeHead(200, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ summary: result.response }));
                    } else {
                        throw new Error('Ollama request failed');
                    }
                } catch (err) {
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: err.message }));
                }
            });
            break;
            
        default:
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ 
                error: 'Not found',
                availableEndpoints: [
                    '/health',
                    '/aggregate',
                    '/errors',
                    '/history',
                    '/recommendations',
                    '/autosolve/trigger',
                    '/autosolve/status',
                    '/ollama/summary'
                ]
            }));
    }
}

// Create and start server
const server = http.createServer(handleRequest);

server.listen(PORT, () => {
    console.log(`\n🚀 Aggregate Server running on http://localhost:${PORT}`);
    console.log('\nAvailable endpoints:');
    console.log(`  GET  /health            - Server health check`);
    console.log(`  GET  /aggregate         - Complete system summary`);
    console.log(`  GET  /errors            - Current TypeScript errors`);
    console.log(`  GET  /history           - Autosolve history`);
    console.log(`  GET  /recommendations   - AI recommendations`);
    console.log(`  POST /autosolve/trigger - Trigger autosolve process`);
    console.log(`  GET  /autosolve/status  - Check if autosolve is running`);
    console.log(`  POST /ollama/summary    - Generate error summary with Ollama`);
    console.log('\n');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down aggregate server...');
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});

module.exports = { server };
