// Memory Monitor API Service
// Provides real-time memory data to the admin dashboard

const http = require('http');
const os = require('os');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

const PORT = 3456;
const LOG_DIR = path.join(__dirname, '..', 'logs', 'memory');
const CRASH_LOG_DIR = path.join(LOG_DIR, 'crash-prevention');

// Ensure log directories exist
if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
}
if (!fs.existsSync(CRASH_LOG_DIR)) {
    fs.mkdirSync(CRASH_LOG_DIR, { recursive: true });
}

// Memory history storage
const memoryHistory = [];
const MAX_HISTORY = 100;

// Get current memory status
function getMemoryStatus() {
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = totalMem - freeMem;
    const usagePercent = (usedMem / totalMem) * 100;
    
    return {
        totalGB: (totalMem / (1024 * 1024 * 1024)).toFixed(2),
        usedGB: (usedMem / (1024 * 1024 * 1024)).toFixed(2),
        freeGB: (freeMem / (1024 * 1024 * 1024)).toFixed(2),
        usagePercent: usagePercent.toFixed(2),
        timestamp: new Date().toISOString()
    };
}

// Get process information
function getProcessInfo(callback) {
    if (process.platform === 'win32') {
        // Windows: Use PowerShell to get process info
        const cmd = `powershell -Command "Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 5 | ForEach-Object { @{Name=$_.ProcessName; Memory=[math]::Round($_.WorkingSet64/1MB,2)} } | ConvertTo-Json"`;
        
        exec(cmd, (error, stdout, stderr) => {
            if (error) {
                callback([]);
                return;
            }
            try {
                const processes = JSON.parse(stdout);
                callback(Array.isArray(processes) ? processes : [processes]);
            } catch (e) {
                callback([]);
            }
        });
    } else {
        // Unix/Linux: Use ps command
        exec('ps aux --sort=-%mem | head -6 | tail -5', (error, stdout, stderr) => {
            if (error) {
                callback([]);
                return;
            }
            
            const lines = stdout.trim().split('\n');
            const processes = lines.map(line => {
                const parts = line.split(/\s+/);
                return {
                    Name: parts[10] || 'Unknown',
                    Memory: parseFloat(parts[3]) || 0
                };
            });
            callback(processes);
        });
    }
}

// Save crash prevention log
function saveCrashLog(memoryStatus, processes) {
    const crashLog = {
        timestamp: new Date().toISOString(),
        memory: memoryStatus,
        topProcesses: processes,
        platform: {
            type: os.type(),
            release: os.release(),
            arch: os.arch(),
            cpus: os.cpus().length,
            uptime: os.uptime()
        }
    };
    
    const filename = `crash-log-${Date.now()}.json`;
    const filepath = path.join(CRASH_LOG_DIR, filename);
    
    fs.writeFileSync(filepath, JSON.stringify(crashLog, null, 2));
    
    return filename;
}

// Optimize memory (Windows specific)
function optimizeMemory(callback) {
    const results = [];
    
    if (process.platform === 'win32') {
        // Clear Node.js garbage
        if (global.gc) {
            global.gc();
            results.push('Node.js garbage collection completed');
        }
        
        // Clear Windows working sets
        exec('powershell -Command "Get-Process | ForEach-Object { $_.Refresh() }"', (error) => {
            if (!error) {
                results.push('Windows working sets refreshed');
            }
            
            // Clear temp files
            const tempDir = process.env.TEMP || process.env.TMP;
            if (tempDir) {
                exec(`powershell -Command "Remove-Item '${tempDir}\\*' -Force -Recurse -ErrorAction SilentlyContinue"`, (error) => {
                    if (!error) {
                        results.push('Temporary files cleared');
                    }
                    callback(results);
                });
            } else {
                callback(results);
            }
        });
    } else {
        // Unix/Linux memory optimization
        exec('sync && echo 3 > /proc/sys/vm/drop_caches', (error) => {
            if (!error) {
                results.push('System caches dropped');
            }
            callback(results);
        });
    }
}

// HTTP Server
const server = http.createServer((req, res) => {
    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
    }
    
    const url = new URL(req.url, `http://${req.headers.host}`);
    
    switch (url.pathname) {
        case '/status':
            // Get current memory status
            const memStatus = getMemoryStatus();
            getProcessInfo((processes) => {
                const response = {
                    memory: memStatus,
                    processes: processes,
                    processCount: processes.length
                };
                
                // Add to history
                memoryHistory.push(memStatus);
                if (memoryHistory.length > MAX_HISTORY) {
                    memoryHistory.shift();
                }
                
                // Check for high memory and save crash log
                if (parseFloat(memStatus.usagePercent) > 85) {
                    const logFile = saveCrashLog(memStatus, processes);
                    response.crashLogSaved = logFile;
                }
                
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(response));
            });
            break;
            
        case '/optimize':
            // Run memory optimization
            optimizeMemory((results) => {
                const beforeStatus = getMemoryStatus();
                
                setTimeout(() => {
                    const afterStatus = getMemoryStatus();
                    const response = {
                        before: beforeStatus,
                        after: afterStatus,
                        freedGB: (parseFloat(beforeStatus.usedGB) - parseFloat(afterStatus.usedGB)).toFixed(2),
                        actions: results
                    };
                    
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify(response));
                }, 1000);
            });
            break;
            
        case '/history':
            // Get memory history
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(memoryHistory));
            break;
            
        case '/logs':
            // Get crash logs
            const crashLogs = fs.readdirSync(CRASH_LOG_DIR)
                .filter(file => file.endsWith('.json'))
                .map(file => {
                    const content = fs.readFileSync(path.join(CRASH_LOG_DIR, file), 'utf8');
                    return JSON.parse(content);
                })
                .slice(-10); // Last 10 logs
            
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(crashLogs));
            break;
            
        case '/health':
            // Health check endpoint
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'running', uptime: process.uptime() }));
            break;
            
        default:
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Not found' }));
    }
});

// Start server
server.listen(PORT, () => {
    console.log(`🧠 Memory Monitor API running on http://localhost:${PORT}`);
    console.log('Available endpoints:');
    console.log('  GET /status   - Current memory status');
    console.log('  GET /optimize - Run memory optimization');
    console.log('  GET /history  - Memory usage history');
    console.log('  GET /logs     - Crash prevention logs');
    console.log('  GET /health   - Service health check');
});

// Monitor memory every 30 seconds
setInterval(() => {
    const status = getMemoryStatus();
    memoryHistory.push(status);
    if (memoryHistory.length > MAX_HISTORY) {
        memoryHistory.shift();
    }
    
    // Log warning if memory is high
    if (parseFloat(status.usagePercent) > 80) {
        console.log(`⚠️ High memory usage: ${status.usagePercent}%`);
    }
}, 30000);

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Memory Monitor API shutting down...');
    server.close(() => {
        process.exit(0);
    });
});