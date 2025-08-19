#!/usr/bin/env node
/**
 * Legal AI Node.js Cluster Manager (CommonJS .cjs)
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

const MANAGER_PORT = parseInt(process.env.MANAGER_PORT || '3000',10);
const LOCK_FILE = '.cluster-manager.pid';

class LegalAIClusterManager {
  constructor() {
    this.workers = new Map();
    this.workerTypes = new Map();
  }
  async initialize() {
    console.log('[CLUSTER] Starting Legal AI Cluster Manager...');
    if (cluster.isPrimary) {
      if(!this.ensureSingleton()) return;
      await this.startMasterProcess();
    } else {
      await this.startWorkerProcess();
    }
  }
  async startMasterProcess() {
    console.log(`[MASTER] Master process started (PID: ${process.pid})`);
    console.log(`[MASTER] CPU cores available: ${os.cpus().length}`);
    this.startManagementServer();
    this.spawnWorkers();
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
    const existingWorkers = Array.from(this.workers.values()).filter(w => w.type === type);
    return existingWorkers.length;
  }
  setupClusterEvents() {
    cluster.on('exit', (worker, code, signal) => {
      const workerId = this.findWorkerById(worker.id);
      const workerType = this.workerTypes.get(worker.id);
      console.log(`[MASTER] Worker ${workerId} died (code: ${code}, signal: ${signal})`);
      this.workers.delete(workerId);
      this.workerTypes.delete(worker.id);
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
      if (this.workers.has(workerId)) this.workers.get(workerId).status = 'online';
    });
  }
  findWorkerById(id) {
    for (const [workerId, info] of this.workers) {
      if (info.worker.id === id) return workerId;
    }
    return `worker-${id}`;
  }
  startManagementServer() {
  const server = http.createServer((req, res) => {
      if (req.url === '/status' && req.method === 'GET') {
        const status = {
          master: { pid: process.pid, uptime: process.uptime(), memory: process.memoryUsage() },
          workers: Array.from(this.workers.entries()).map(([id, info]) => ({
            id, type: info.type, pid: info.pid, status: info.status, uptime: (Date.now() - info.startTime) / 1000
          }))
        };
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(status, null, 2));
      } else if (req.url === '/health' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'healthy', timestamp: Date.now() }));
      } else { res.writeHead(404); res.end('Not Found'); }
    });
    server.on('error', (err) => {
      if(err.code === 'EADDRINUSE'){
        if(process.env.ALLOW_MULTIPLE_CLUSTERS === 'true'){
          console.warn(`[MASTER] Port ${MANAGER_PORT} in use; continuing without management server (ALLOW_MULTIPLE_CLUSTERS=true).`);
        } else {
          console.error(`[MASTER] Port ${MANAGER_PORT} already in use. Another cluster manager may be running. Exiting.`);
          process.exit(2);
        }
      } else {
        console.error('[MASTER] Management server error:', err);
      }
    });
    server.listen(MANAGER_PORT, () => console.log(`[MASTER] Management server listening on port ${MANAGER_PORT}`));
    return server;
  }
  async startWorkerProcess() {
    const workerType = process.env.WORKER_TYPE;
    const workerPort = process.env.WORKER_PORT;
    console.log(`[WORKER] Starting ${workerType} worker (PID: ${process.pid})`);
    const server = http.createServer((req, res) => {
      if (req.url === '/health' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ type: workerType, pid: process.pid, uptime: process.uptime(), memory: process.memoryUsage(), status: 'healthy' }));
      } else {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ message: `${workerType} worker placeholder`, worker: workerType, pid: process.pid }));
      }
    });
    server.listen(workerPort, () => console.log(`[WORKER] ${workerType} worker listening on port ${workerPort}`));
    process.on('SIGTERM', () => { console.log(`[WORKER] ${workerType} worker shutting down gracefully...`); server.close(() => process.exit(0)); });
  }
  ensureSingleton(){
    const fs = require('fs');
    try {
      if(fs.existsSync(LOCK_FILE)){
        const existingPid = parseInt(fs.readFileSync(LOCK_FILE,'utf8').trim(),10);
        if(existingPid && !isNaN(existingPid)){
          try { process.kill(existingPid,0); } catch(e){
            console.warn('[MASTER] Stale lock file detected; removing.');
            fs.unlinkSync(LOCK_FILE);
          }
        }
      }
      if(fs.existsSync(LOCK_FILE)){
        if(process.env.FORCE_CLUSTER==='1'){
          console.warn('[MASTER] FORCE_CLUSTER=1 override active; replacing existing lock.');
          fs.unlinkSync(LOCK_FILE);
        } else {
          console.error('[MASTER] Another cluster manager appears active (lock file present). Aborting. Set FORCE_CLUSTER=1 to override.');
          return false;
        }
      }
      fs.writeFileSync(LOCK_FILE,String(process.pid));
      const cleanup = ()=>{ try { fs.unlinkSync(LOCK_FILE); } catch(_){} };
      process.on('exit', cleanup);
      process.on('SIGINT', ()=>{ cleanup(); process.exit(0); });
      process.on('SIGTERM', ()=>{ cleanup(); process.exit(0); });
      return true;
    } catch(err){
      console.warn('[MASTER] Singleton lock handling error:', err.message);
      return true; // non-fatal
    }
  }
}

if (require.main === module) {
  const manager = new LegalAIClusterManager();
  manager.initialize().catch(err => { console.error('[MASTER] Cluster manager failed to initialize:', err); process.exit(1); });
}

module.exports = LegalAIClusterManager;
