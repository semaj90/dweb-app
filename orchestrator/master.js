// orchestrator/master.js
// Master process that spawns worker cluster for GPU orchestration
// Usage: node master.js [--dev] [--autosolve]

import cluster from 'cluster';
import os from 'os';
import path from 'path';
import http from 'http';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isDev = process.argv.includes('--dev');
const isAutoSolve = process.argv.includes('--autosolve');

async function main() {
if (cluster.isMaster || cluster.isPrimary) {
    console.log(`🚀 GPU Orchestrator Master starting on ${os.hostname()}`);
    console.log(`📊 CPU cores available: ${os.cpus().length}`);
    console.log(`🔧 Mode: ${isDev ? 'Development' : 'Production'}`);
    console.log(`🤖 Auto-solve: ${isAutoSolve ? 'Enabled' : 'Disabled'}`);
    
    // Calculate optimal worker count (leave 1 core for master + system)
    const maxWorkers = Math.max(1, os.cpus().length - 1);
    const workerCount = isDev ? Math.min(2, maxWorkers) : maxWorkers;
    
    console.log(`👥 Starting ${workerCount} worker processes...`);
    
    // Set environment variables for workers
    process.env.WORKER_MODE = isDev ? 'dev' : 'production';
    process.env.AUTO_SOLVE_ENABLED = isAutoSolve ? 'true' : 'false';
    process.env.CUDA_WORKER_PATH = path.join(__dirname, '..', 'cuda-worker', 'cuda-worker.exe');
    
    // Track worker metrics
    const workerMetrics = new Map();
    let totalJobsProcessed = 0;
    
    // Start workers
    for (let i = 0; i < workerCount; i++) {
        const worker = cluster.fork();
        workerMetrics.set(worker.id, {
            pid: worker.process.pid,
            started: Date.now(),
            jobsProcessed: 0,
            lastActivity: Date.now()
        });
        
        console.log(`✅ Worker ${worker.id} started (PID: ${worker.process.pid})`);
    }
    
    // Handle worker messages (metrics, health updates)
    cluster.on('message', (worker, message) => {
        if (message.type === 'job_completed') {
            const metrics = workerMetrics.get(worker.id);
            if (metrics) {
                metrics.jobsProcessed++;
                metrics.lastActivity = Date.now();
                totalJobsProcessed++;
                
                if (isDev) {
                    console.log(`📈 Worker ${worker.id} completed job ${message.jobId} (Total: ${totalJobsProcessed})`);
                }
            }
        } else if (message.type === 'worker_health') {
            const metrics = workerMetrics.get(worker.id);
            if (metrics) {
                metrics.lastActivity = Date.now();
                metrics.health = message.data;
            }
        }
    });
    
    // Handle worker exits and restart
    cluster.on('exit', (worker, code, signal) => {
        const metrics = workerMetrics.get(worker.id);
        const uptime = metrics ? Math.round((Date.now() - metrics.started) / 1000) : 0;
        
        console.log(`❌ Worker ${worker.id} died (PID: ${worker.process.pid}, uptime: ${uptime}s, code: ${code}, signal: ${signal})`);
        workerMetrics.delete(worker.id);
        
        // Restart worker unless it's a graceful shutdown
        if (code !== 0 && signal !== 'SIGTERM' && signal !== 'SIGINT') {
            console.log(`🔄 Restarting worker ${worker.id}...`);
            const newWorker = cluster.fork();
            workerMetrics.set(newWorker.id, {
                pid: newWorker.process.pid,
                started: Date.now(),
                jobsProcessed: 0,
                lastActivity: Date.now()
            });
            console.log(`✅ Worker ${newWorker.id} restarted (PID: ${newWorker.process.pid})`);
        }
    });
    
    // Periodic health check and metrics
    setInterval(() => {
        const now = Date.now();
        let healthyWorkers = 0;
        let stalledWorkers = 0;
        
        for (const [workerId, metrics] of workerMetrics) {
            const timeSinceActivity = now - metrics.lastActivity;
            
            if (timeSinceActivity > 300000) { // 5 minutes
                stalledWorkers++;
                console.log(`⚠️ Worker ${workerId} appears stalled (last activity: ${Math.round(timeSinceActivity/1000)}s ago)`);
            } else {
                healthyWorkers++;
            }
        }
        
        if (isDev || totalJobsProcessed > 0) {
            console.log(`📊 Health check: ${healthyWorkers} healthy, ${stalledWorkers} stalled, ${totalJobsProcessed} total jobs processed`);
        }
    }, 60000); // Every minute
    
    // Graceful shutdown handling
    function gracefulShutdown(signal) {
        console.log(`\n🛑 Received ${signal}, initiating graceful shutdown...`);
        
        // Stop accepting new work
        for (const id in cluster.workers) {
            cluster.workers[id].send({ type: 'shutdown' });
        }
        
        // Give workers time to finish current jobs
        setTimeout(() => {
            console.log('💀 Force killing remaining workers...');
            for (const id in cluster.workers) {
                cluster.workers[id].kill('SIGKILL');
            }
            process.exit(0);
        }, 10000); // 10 second grace period
        
        // Wait for workers to exit naturally
        cluster.disconnect(() => {
            console.log('✅ All workers disconnected gracefully');
            process.exit(0);
        });
    }
    
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));
    
    // Basic HTTP status endpoint
    if (isDev) {
        const server = http.createServer((req, res) => {
            if (req.url === '/health') {
                const workers = Array.from(workerMetrics.entries()).map(([id, metrics]) => ({
                    id,
                    pid: metrics.pid,
                    uptime: Math.round((Date.now() - metrics.started) / 1000),
                    jobsProcessed: metrics.jobsProcessed,
                    lastActivity: new Date(metrics.lastActivity).toISOString()
                }));
                
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    status: 'healthy',
                    totalWorkers: workerMetrics.size,
                    totalJobsProcessed,
                    workers,
                    uptime: process.uptime(),
                    memory: process.memoryUsage()
                }, null, 2));
            } else {
                res.writeHead(404);
                res.end('Not Found');
            }
        });
        
        server.listen(8099, () => {
            console.log(`🌐 Health endpoint available at http://localhost:8099/health`);
        });
    }
    
    console.log(`🎯 GPU Orchestrator Master ready with ${workerCount} workers`);
    
} else {
    // Worker process
    await import('./worker_process.js');
}
}

main().catch(console.error);