#!/usr/bin/env node

/**
 * Legal AI Node.js Cluster Manager
 * 
 * Manages specialized worker processes for legal document processing,
 * AI analysis, vector operations, and database queries.
 */

const cluster = require('cluster');
const os = require('os');
const http = require('http');

const CLUSTER_CONFIG = {
  legal: { count: 3, memory: '512MB', port: 3010 },
  ai: { count: 2, memory: '1GB', port: 3020 },
  vector: { count: 2, memory: '256MB', port: 3030 },
  database: { count: 3, memory: '256MB', port: 3040 }
};

const MANAGER_PORT = 3000;

class LegalAIClusterManager {
  constructor() {
    this.workers = new Map();
    this.workerTypes = new Map();
  }

  async initialize() {
    console.log('[CLUSTER] Starting Legal AI Cluster Manager...');
    
    if (cluster.isMaster) {
      await this.startMasterProcess();
    } else {
      await this.startWorkerProcess();
    }
  }

  async startMasterProcess() {
    console.log(`[MASTER] Master process started (PID: ${process.pid})`);
    console.log(`[MASTER] CPU cores available: ${os.cpus().length}`);
    
    // Start management server
    this.startManagementServer();
    
    // Spawn worker processes
    this.spawnWorkers();
    
    // Set up cluster event handlers
    this.setupClusterEvents();
    
    console.log('[MASTER] Cluster manager fully initialized');
  }

  spawnWorkers() {
    Object.entries(CLUSTER_CONFIG).forEach(([type, config]) => {
      for (let i = 0; i < config.count; i++) {
        this.spawnWorker(type, config);
      }
    });
  }

  spawnWorker(type, config) {
    const worker = cluster.fork({
      WORKER_TYPE: type,
      WORKER_MEMORY: config.memory,
      WORKER_PORT: config.port + this.getWorkerIndex(type)
    });

    const workerId = `${type}-${worker.id}`;
    this.workers.set(workerId, {
      worker,
      type,
      pid: worker.process.pid,
      status: 'starting',
      startTime: Date.now()
    });

    this.workerTypes.set(worker.id, type);
    
    console.log(`[MASTER] Spawned ${type} worker: ${workerId} (PID: ${worker.process.pid})`);
    
    return worker;
  }

  getWorkerIndex(type) {
    const existingWorkers = Array.from(this.workers.values())
      .filter(w => w.type === type);
    return existingWorkers.length;
  }

  setupClusterEvents() {
    cluster.on('exit', (worker, code, signal) => {
      const workerId = this.findWorkerById(worker.id);
      const workerType = this.workerTypes.get(worker.id);
      
      console.log(`[MASTER] Worker ${workerId} died (code: ${code}, signal: ${signal})`);
      
      // Remove from tracking
      this.workers.delete(workerId);
      this.workerTypes.delete(worker.id);
      
      // Respawn worker after delay
      if (code !== 0 && !worker.exitedAfterDisconnect) {
        console.log(`[MASTER] Respawning ${workerType} worker in 2 seconds...`);
        setTimeout(() => {
          const config = CLUSTER_CONFIG[workerType];
          this.spawnWorker(workerType, config);
        }, 2000);
      }
    });

    cluster.on('online', (worker) => {
      const workerId = this.findWorkerById(worker.id);
      console.log(`[MASTER] Worker ${workerId} is online`);
      
      if (this.workers.has(workerId)) {
        this.workers.get(workerId).status = 'online';
      }
    });
  }

  findWorkerById(id) {
    for (const [workerId, info] of this.workers) {
      if (info.worker.id === id) {
        return workerId;
      }
    }
    return `worker-${id}`;
  }

  startManagementServer() {
    const server = http.createServer((req, res) => {
      if (req.url === '/status' && req.method === 'GET') {
        const status = {
          master: {
            pid: process.pid,
            uptime: process.uptime(),
            memory: process.memoryUsage()
          },
          workers: Array.from(this.workers.entries()).map(([id, info]) => ({
            id,
            type: info.type,
            pid: info.pid,
            status: info.status,
            uptime: (Date.now() - info.startTime) / 1000
          }))
        };
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(status, null, 2));
      } else if (req.url === '/health' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'healthy', timestamp: Date.now() }));
      } else {
        res.writeHead(404);
        res.end('Not Found');
      }
    });

    server.listen(MANAGER_PORT, () => {
      console.log(`[MASTER] Management server listening on port ${MANAGER_PORT}`);
    });

    return server;
  }

  async startWorkerProcess() {
    const workerType = process.env.WORKER_TYPE;
    const workerPort = process.env.WORKER_PORT;
    
    console.log(`[WORKER] Starting ${workerType} worker (PID: ${process.pid})`);
    
    // Create basic worker server
    const server = http.createServer((req, res) => {
      if (req.url === '/health' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          type: workerType,
          pid: process.pid,
          uptime: process.uptime(),
          memory: process.memoryUsage(),
          status: 'healthy'
        }));
      } else {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          message: `${workerType} worker placeholder`,
          worker: workerType,
          pid: process.pid
        }));
      }
    });
    
    server.listen(workerPort, () => {
      console.log(`[WORKER] ${workerType} worker listening on port ${workerPort}`);
    });

    // Graceful shutdown
    process.on('SIGTERM', () => {
      console.log(`[WORKER] ${workerType} worker shutting down gracefully...`);
      server.close(() => {
        process.exit(0);
      });
    });
  }
}

// Start the cluster manager
if (require.main === module) {
  const manager = new LegalAIClusterManager();
  manager.initialize().catch(console.error);
}

module.exports = LegalAIClusterManager;