import express from 'express';
import { WebSocketServer } from 'ws';
import fetch from 'node-fetch';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);
const app = express();
const port = 3000;

// Real-time monitoring dashboard
class SystemMonitor {
    constructor() {
        this.services = {
            microservice: { url: 'http://localhost:8081/health', status: 'unknown' },
            neo4j: { url: 'http://localhost:7474', status: 'unknown' },
            ollama: { url: 'http://localhost:11434/api/tags', status: 'unknown' },
            redis: { cmd: 'redis-cli ping', status: 'unknown' },
            postgres: { cmd: 'pg_isready -h localhost', status: 'unknown' }
        };
        
        this.metrics = {
            gpu: { utilization: 0, memory: 0, temperature: 0 },
            indexedFiles: 0,
            totalErrors: 0,
            cacheHits: 0,
            activeSessions: 0
        };
        
        this.startMonitoring();
    }

    async startMonitoring() {
        setInterval(() => this.checkServices(), 5000);
        setInterval(() => this.collectMetrics(), 10000);
    }

    async checkServices() {
        for (const [name, service] of Object.entries(this.services)) {
            if (service.url) {
                try {
                    const response = await fetch(service.url, { timeout: 2000 });
                    service.status = response.ok ? 'running' : 'error';
                } catch {
                    service.status = 'offline';
                }
            } else if (service.cmd) {
                try {
                    await execAsync(service.cmd);
                    service.status = 'running';
                } catch {
                    service.status = 'offline';
                }
            }
        }
    }

    async collectMetrics() {
        // GPU metrics
        try {
            const { stdout } = await execAsync('nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits');
            const [gpu, memory, temp] = stdout.trim().split(', ').map(Number);
            this.metrics.gpu = { utilization: gpu, memory, temperature: temp };
        } catch {}

        // Microservice metrics
        try {
            const response = await fetch('http://localhost:8081/metrics');
            const data = await response.json();
            this.metrics.indexedFiles = data.indexed_files || 0;
            this.metrics.totalErrors = data.total_errors || 0;
            this.metrics.cacheHits = data.cache_metrics?.hits || 0;
            this.metrics.activeSessions = data.stream_metrics?.active_connections || 0;
        } catch {}
    }

    getDashboardHTML() {
        return `<!DOCTYPE html>
<html>
<head>
    <title>AI System Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0a0e27;
            color: #e4e6eb;
            padding: 20px;
        }
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            margin-bottom: 30px;
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: #1c1e2e;
            border: 1px solid #2d2f3e;
            border-radius: 8px;
            padding: 20px;
        }
        .card h2 {
            font-size: 14px;
            color: #8b92a9;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .service {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #2d2f3e;
        }
        .service:last-child { border-bottom: none; }
        .status {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .status.running { background: #00a550; }
        .status.offline { background: #e74c3c; }
        .status.unknown { background: #6c757d; }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
        }
        .gpu-bar {
            height: 20px;
            background: #2d2f3e;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .gpu-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00a550);
            transition: width 0.3s;
        }
        .actions {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        button {
            padding: 10px 20px;
            background: #00d4ff;
            color: #0a0e27;
            border: none;
            border-radius: 4px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:hover {
            background: #00a8d4;
            transform: translateY(-1px);
        }
        .logs {
            background: #0d1117;
            border: 1px solid #2d2f3e;
            border-radius: 4px;
            padding: 15px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            height: 300px;
            overflow-y: auto;
        }
        .log-entry {
            padding: 2px 0;
            color: #8b92a9;
        }
        .log-entry.error { color: #e74c3c; }
        .log-entry.success { color: #00a550; }
        .log-entry.info { color: #00d4ff; }
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>🚀 AI System Monitor</h1>
        
        <div class="grid">
            <div class="card">
                <h2>Services</h2>
                <div id="services"></div>
            </div>
            
            <div class="card">
                <h2>GPU Status</h2>
                <div class="metric">
                    <span>Utilization</span>
                    <span class="metric-value" id="gpu-util">0%</span>
                </div>
                <div class="gpu-bar">
                    <div class="gpu-fill" id="gpu-bar" style="width: 0%"></div>
                </div>
                <div class="metric">
                    <span>Memory</span>
                    <span id="gpu-mem">0%</span>
                </div>
                <div class="metric">
                    <span>Temperature</span>
                    <span id="gpu-temp">0°C</span>
                </div>
            </div>
            
            <div class="card">
                <h2>System Metrics</h2>
                <div class="metric">
                    <span>Indexed Files</span>
                    <span class="metric-value" id="indexed-files">0</span>
                </div>
                <div class="metric">
                    <span>Total Errors</span>
                    <span class="metric-value" id="total-errors">0</span>
                </div>
                <div class="metric">
                    <span>Cache Hits</span>
                    <span class="metric-value" id="cache-hits">0</span>
                </div>
                <div class="metric">
                    <span>Active Sessions</span>
                    <span class="metric-value" id="active-sessions">0</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Actions</h2>
            <div class="actions">
                <button onclick="indexProject()">Index Project</button>
                <button onclick="analyzeErrors()">Analyze Errors</button>
                <button onclick="runFixes()">Run Auto-Fix</button>
                <button onclick="clearCache()">Clear Cache</button>
                <button onclick="restartServices()">Restart Services</button>
            </div>
        </div>
        
        <div class="card">
            <h2>Live Logs</h2>
            <div class="logs" id="logs"></div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket('ws://localhost:3000');
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        function updateDashboard(data) {
            // Update services
            const servicesHtml = Object.entries(data.services).map(([name, service]) => 
                \`<div class="service">
                    <span>\${name}</span>
                    <span class="status \${service.status}">\${service.status}</span>
                </div>\`
            ).join('');
            document.getElementById('services').innerHTML = servicesHtml;
            
            // Update GPU metrics
            document.getElementById('gpu-util').textContent = data.metrics.gpu.utilization + '%';
            document.getElementById('gpu-bar').style.width = data.metrics.gpu.utilization + '%';
            document.getElementById('gpu-mem').textContent = data.metrics.gpu.memory + '%';
            document.getElementById('gpu-temp').textContent = data.metrics.gpu.temperature + '°C';
            
            // Update system metrics
            document.getElementById('indexed-files').textContent = data.metrics.indexedFiles;
            document.getElementById('total-errors').textContent = data.metrics.totalErrors;
            document.getElementById('cache-hits').textContent = data.metrics.cacheHits;
            document.getElementById('active-sessions').textContent = data.metrics.activeSessions;
        }
        
        function addLog(message, type = 'info') {
            const logs = document.getElementById('logs');
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + type;
            entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + message;
            logs.appendChild(entry);
            logs.scrollTop = logs.scrollHeight;
        }
        
        async function indexProject() {
            addLog('Starting project indexing...', 'info');
            const response = await fetch('http://localhost:8081/index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rootPath: './sveltekit-frontend' })
            });
            if (response.ok) {
                addLog('Indexing started successfully', 'success');
            } else {
                addLog('Indexing failed', 'error');
            }
        }
        
        async function analyzeErrors() {
            addLog('Analyzing TypeScript errors...', 'info');
            const response = await fetch('/api/analyze-errors', { method: 'POST' });
            const data = await response.json();
            addLog(\`Found \${data.totalErrors} errors in \${data.filesAnalyzed} files\`, 'info');
        }
        
        async function runFixes() {
            addLog('Running auto-fix pipeline...', 'info');
            const response = await fetch('/api/run-fixes', { method: 'POST' });
            if (response.ok) {
                addLog('Auto-fix pipeline started', 'success');
            }
        }
        
        async function clearCache() {
            addLog('Clearing cache...', 'info');
            const response = await fetch('http://localhost:8081/cache/clear', { method: 'DELETE' });
            if (response.ok) {
                addLog('Cache cleared', 'success');
            }
        }
        
        async function restartServices() {
            addLog('Restarting services...', 'info');
            const response = await fetch('/api/restart-services', { method: 'POST' });
            if (response.ok) {
                addLog('Services restart initiated', 'success');
            }
        }
        
        // Auto-refresh
        setInterval(() => {
            ws.send(JSON.stringify({ type: 'refresh' }));
        }, 5000);
    </script>
</body>
</html>`;
    }
}

const monitor = new SystemMonitor();

// WebSocket server for real-time updates
const wss = new WebSocketServer({ port: 3000 });

wss.on('connection', (ws) => {
    console.log('Dashboard connected');
    
    // Send initial data
    ws.send(JSON.stringify({
        services: monitor.services,
        metrics: monitor.metrics
    }));
    
    // Send updates
    const interval = setInterval(() => {
        ws.send(JSON.stringify({
            services: monitor.services,
            metrics: monitor.metrics
        }));
    }, 2000);
    
    ws.on('close', () => {
        clearInterval(interval);
    });
});

// Express routes
app.get('/', (req, res) => {
    res.send(monitor.getDashboardHTML());
});

app.post('/api/analyze-errors', async (req, res) => {
    const { stdout } = await execAsync('npm run check 2>&1 || true', {
        cwd: './sveltekit-frontend'
    });
    
    const errors = stdout.match(/error TS\d+/g) || [];
    res.json({
        totalErrors: errors.length,
        filesAnalyzed: new Set(stdout.match(/\S+\.(ts|tsx|svelte)/g) || []).size
    });
});

app.post('/api/run-fixes', async (req, res) => {
    execAsync('node fix-typescript-errors.mjs', (error, stdout, stderr) => {
        console.log(stdout);
        if (error) console.error(stderr);
    });
    res.json({ status: 'started' });
});

app.post('/api/restart-services', async (req, res) => {
    // Restart Go microservice
    execAsync('taskkill /F /IM ai-microservice.exe & timeout 2 & start /B go-microservice\\ai-microservice.exe');
    res.json({ status: 'restarting' });
});

app.listen(port, () => {
    console.log(`🎯 Monitor dashboard: http://localhost:${port}`);
    console.log('📊 WebSocket: ws://localhost:3000');
});
