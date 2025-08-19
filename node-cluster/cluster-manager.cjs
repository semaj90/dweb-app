#!/usr/bin/env node
/**
 * Legal AI Node.js Cluster Manager (CommonJS .cjs)
 */
const cluster = require('cluster');
const os = require('os');
const http = require('http');

// Parse CLI arguments (before building config) to allow overriding env vars easily in PowerShell / scripts
function parseCliArgs() {
    const argMap = new Map();
    for (let i = 2; i < process.argv.length; i++) {
        const a = process.argv[i];
        if (a.startsWith('--')) {
            const [key, rawVal] = a.replace(/^--/, '').split('=');
            const val = rawVal === undefined ? 'true' : rawVal;
            argMap.set(key.toLowerCase(), val);
        }
    }
    const setIf = (cliKey, envKey) => {
        if (argMap.has(cliKey)) process.env[envKey] = String(argMap.get(cliKey));
    };
    setIf('manager-port', 'MANAGER_PORT');
    setIf('legal-count', 'LEGAL_COUNT');
    setIf('ai-count', 'AI_COUNT');
    setIf('vector-count', 'VECTOR_COUNT');
    setIf('database-count', 'DATABASE_COUNT');
    setIf('legal-base-port', 'LEGAL_BASE_PORT');
    setIf('ai-base-port', 'AI_BASE_PORT');
    setIf('vector-base-port', 'VECTOR_BASE_PORT');
    setIf('database-base-port', 'DATABASE_BASE_PORT');
    setIf('shutdown-grace-ms', 'SHUTDOWN_GRACE_MS');
}

// Build runtime configuration with environment overrides (env may be mutated by parseCliArgs)
function buildClusterConfig() {
    const int = (v, d) => {
        const n = parseInt(v, 10);
        return Number.isFinite(n) && n >= 0 ? n : d;
    };
    const base = {
        legal: {
            count: int(process.env.LEGAL_COUNT, 3),
            memory: process.env.LEGAL_MEMORY || '512MB',
            port: int(process.env.LEGAL_BASE_PORT, 3010)
        },
        ai: {
            count: int(process.env.AI_COUNT, 2),
            memory: process.env.AI_MEMORY || '1GB',
            port: int(process.env.AI_BASE_PORT, 3020)
        },
        vector: {
            count: int(process.env.VECTOR_COUNT, 2),
            memory: process.env.VECTOR_MEMORY || '256MB',
            port: int(process.env.VECTOR_BASE_PORT, 3030)
        },
        database: {
            count: int(process.env.DATABASE_COUNT, 3),
            memory: process.env.DATABASE_MEMORY || '256MB',
            port: int(process.env.DATABASE_BASE_PORT, 3040)
        }
    };
    return base;
}

// Defer building until after potential CLI parsing inside constructor
let CLUSTER_CONFIG = null;

// Manager port will be resolved after CLI arg parsing (see constructor)
let MANAGER_PORT = parseInt(process.env.MANAGER_PORT || '3000', 10);
const LOCK_FILE = '.cluster-manager.pid';

class LegalAIClusterManager {
  constructor() {
      // Allow CLI args like: node cluster-manager.cjs --manager-port=3050 --legal-count=1 --ai-count=1
      parseCliArgs();
      // Re-resolve manager port after potential CLI arg overrides
      MANAGER_PORT = parseInt(process.env.MANAGER_PORT || '3000', 10);
      if (process.env.LOG_LEVEL === 'debug') {
          console.log('[DEBUG] argv:', process.argv.slice(2));
          console.log('[DEBUG] resolved MANAGER_PORT:', MANAGER_PORT);
          console.log('[DEBUG] relevant env snapshot:', {
              MANAGER_PORT: process.env.MANAGER_PORT,
              LEGAL_COUNT: process.env.LEGAL_COUNT,
              AI_COUNT: process.env.AI_COUNT,
              VECTOR_COUNT: process.env.VECTOR_COUNT,
              DATABASE_COUNT: process.env.DATABASE_COUNT
          });
      }
      if (!CLUSTER_CONFIG) {
          CLUSTER_CONFIG = buildClusterConfig();
      }
    this.workers = new Map();
    this.workerTypes = new Map();
      this.shuttingDown = false;
      this.deferredSpawns = []; // { type, config, ordinalIndex, attempts }
      this.deferTimer = null;
      this.metrics = {
        startTime: Date.now(),
        spawned: {}, // type -> count
        deferredTotal: 0,
        deferredActive: 0,
        portSearches: { successes: 0, failures: 0 },
        lastAllocation: null,
        events: [] // rolling log
      };
      this.metricsFile = '.vscode/cluster-metrics.json';
      this.metricsWriteInterval = parseInt(process.env.METRICS_WRITE_INTERVAL_MS || '3000', 10);
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
      await this.spawnWorkers();
    this.setupClusterEvents();
  this.startMetricsWriter();
    console.log('[MASTER] Cluster manager fully initialized');
  }
    async spawnWorkers() {
        for (const [type, config] of Object.entries(CLUSTER_CONFIG)) {
      for (let i = 0; i < config.count; i++) {
          await this.spawnWorker(type, config, i);
      }
        }
    }
    async spawnWorker(type, config, ordinalIndex) {
        const basePort = config.port + ordinalIndex; // deterministic anchor
        const port = await this.allocateClosestPort(basePort);
        if (port == null) {
            console.warn(`[MASTER] No free port found near ${basePort} (range ${process.env.PORT_SEARCH_RANGE || 50}); deferring spawn of ${type}#${ordinalIndex}`);
            this.queueDeferredSpawn(type, config, ordinalIndex);
            return null;
        }
    const worker = cluster.fork({
      WORKER_TYPE: type,
      WORKER_MEMORY: config.memory,
        WORKER_PORT: port
    });
    const workerId = `${type}-${worker.id}`;
    this.workers.set(workerId, {
      worker,
      type,
      pid: worker.process.pid,
      status: 'starting',
        startTime: Date.now(),
        port
    });
    this.workerTypes.set(worker.id, type);
      console.log(`[MASTER] Spawned ${type} worker: ${workerId} (PID: ${worker.process.pid}) on port ${port}`);
  this.recordSpawn(type, port);
    return worker;
  }
    // Closest-port allocation: search outward from basePort (delta: 0,+1,-1,+2,-2,...) up to range.
    async allocateClosestPort(basePort) {
        const range = parseInt(process.env.PORT_SEARCH_RANGE || '50', 10);
        const tested = new Set();
        for (let delta = 0; delta <= range; delta++) {
            const candidates = delta === 0 ? [basePort] : [basePort + delta, basePort - delta];
            for (const p of candidates) {
                if (p < 1024 || p > 65535) continue; // skip privileged/out-of-range
                if (tested.has(p)) continue;
                tested.add(p);
                const free = await this.checkPortFree(p);
                if (free) return p;
            }
        }
        return null; // none free within range
    }
    checkPortFree(port) {
        const net = require('net');
        return new Promise((resolve) => {
            const srv = net.createServer()
                .once('error', err => { if (err.code === 'EADDRINUSE') resolve(false); else resolve(false); })
                .once('listening', () => srv.close(() => resolve(true)))
                .listen(port, '0.0.0.0');
        });
    }
    queueDeferredSpawn(type, config, ordinalIndex) {
        this.deferredSpawns.push({ type, config, ordinalIndex, attempts: 0 });
  this.metrics.deferredTotal++;
  this.metrics.deferredActive = this.deferredSpawns.length;
  this.pushEvent(`defer:${type}#${ordinalIndex}`);
        this.ensureDeferLoop();
    }
    ensureDeferLoop() {
        if (this.deferTimer) return;
        const interval = parseInt(process.env.PORT_DEFER_INTERVAL_MS || '1000', 10);
        this.deferTimer = setInterval(async () => {
            if (this.shuttingDown) return;
            if (this.deferredSpawns.length === 0) return;
            const item = this.deferredSpawns.shift();
            if (!item) return;
            item.attempts++;
            try {
                const worker = await this.spawnWorker(item.type, item.config, item.ordinalIndex);
                if (!worker) {
                    // still no port; requeue with backoff if under limit
                    const max = parseInt(process.env.PORT_DEFER_MAX_ATTEMPTS || '30', 10);
                    if (item.attempts < max) {
                        const backoffMs = Math.min(interval * (1 + Math.floor(item.attempts / 5)), 5000);
                        setTimeout(() => { this.deferredSpawns.push(item); }, backoffMs);
                    } else {
                        console.error(`[MASTER] Abandoning spawn of ${item.type}#${item.ordinalIndex} after ${item.attempts} attempts.`);
                        this.pushEvent(`abandon:${item.type}#${item.ordinalIndex}`);
                    }
                }
            } catch (e) {
                console.error('[MASTER] Deferred spawn error:', e.message);
            }
            this.metrics.deferredActive = this.deferredSpawns.length;
        }, interval).unref();
    }
  recordSpawn(type, port){
    this.metrics.spawned[type] = (this.metrics.spawned[type]||0)+1;
    this.metrics.portSearches.successes++;
    this.metrics.lastAllocation = { type, port, time: Date.now() };
    this.pushEvent(`spawn:${type}@${port}`);
  }
  pushEvent(e){
    this.metrics.events.push({ e, t: Date.now() });
    if (this.metrics.events.length > 200) this.metrics.events.shift();
  }
  startMetricsWriter(){
    const fs = require('fs');
    const write = () => {
      if (this.shuttingDown) return;
      try {
        if (!fs.existsSync('.vscode')) fs.mkdirSync('.vscode', { recursive: true });
        const payload = { timestamp: Date.now(), uptimeSec: Math.round((Date.now()-this.metrics.startTime)/1000), ...this.metrics, workers: Array.from(this.workers.values()).map(w=>({type:w.type,pid:w.pid,port:w.port,status:w.status,uptimeSec: Math.round((Date.now()-w.startTime)/1000)})), deferredQueue: this.deferredSpawns.map(d=>({type:d.type,ordinalIndex:d.ordinalIndex,attempts:d.attempts})) };
        fs.writeFileSync(this.metricsFile, JSON.stringify(payload,null,2));
      } catch(err){ /* silent */ }
      setTimeout(write, this.metricsWriteInterval).unref();
    };
    write();
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
        if (!this.shuttingDown && code !== 0 && !worker.exitedAfterDisconnect) {
        console.log(`[MASTER] Respawning ${workerType} worker in 2 seconds...`);
        setTimeout(() => {
          const config = CLUSTER_CONFIG[workerType];
            // ordinal index recomputed by current count
            this.spawnWorker(workerType, config, this.getWorkerIndex(workerType)).catch(err => {
                console.error('[MASTER] Failed to respawn worker:', err.message);
            });
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
          const send = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(obj, null, 2)); };
          if (req.method === 'GET' && req.url === '/status') {
        const status = {
            master: { pid: process.pid, uptime: process.uptime(), memory: process.memoryUsage(), shuttingDown: this.shuttingDown },
            config: CLUSTER_CONFIG,
          workers: Array.from(this.workers.entries()).map(([id, info]) => ({
              id, type: info.type, pid: info.pid, status: info.status, port: info.port, uptime: (Date.now() - info.startTime) / 1000
          })),
            deferred: this.deferredSpawns.map(d => ({ type: d.type, ordinalIndex: d.ordinalIndex, attempts: d.attempts }))
        };
              return send(200, status);
          }
          if (req.method === 'GET' && req.url === '/health') {
              return send(200, { status: 'healthy', timestamp: Date.now(), workers: this.workers.size });
          }
          if (req.method === 'GET' && req.url === '/metrics') {
            const payload = { timestamp: Date.now(), uptimeSec: Math.round((Date.now()-this.metrics.startTime)/1000), ...this.metrics, workers: Array.from(this.workers.values()).map(w=>({type:w.type,pid:w.pid,port:w.port,status:w.status,uptimeSec: Math.round((Date.now()-w.startTime)/1000)})), deferredQueue: this.deferredSpawns.map(d=>({type:d.type,ordinalIndex:d.ordinalIndex,attempts:d.attempts})) };
            return send(200, payload);
          }
          if (req.method === 'POST' && req.url === '/shutdown') {
              if (this.shuttingDown) return send(202, { message: 'Shutdown already in progress' });
              this.shuttingDown = true;
              console.log('[MASTER] Shutdown requested via /shutdown');
              for (const { worker } of this.workers.values()) {
                  try { worker.process.kill('SIGTERM'); } catch (_) { }
              }
              setTimeout(() => { console.log('[MASTER] Exiting after graceful period'); process.exit(0); }, parseInt(process.env.SHUTDOWN_GRACE_MS || '4000', 10));
              return send(202, { message: 'Shutdown initiated', graceMs: parseInt(process.env.SHUTDOWN_GRACE_MS || '4000', 10) });
          }
          res.writeHead(404); res.end('Not Found');
    });
    server.on('error', (err) => {
        if (err.code === 'EADDRINUSE') {
            if (process.env.ALLOW_MULTIPLE_CLUSTERS === 'true') {
          console.warn(`[MASTER] Port ${MANAGER_PORT} in use; continuing without management server (ALLOW_MULTIPLE_CLUSTERS=true).`);
            } else if (process.env.MANAGER_PORT_AUTO === '1') {
                // auto-increment until free (max 20 attempts)
                let attempts = 0;
                const tryNext = () => {
                    if (attempts++ > 20) {
                        console.error(`[MASTER] Failed to find free management port after 20 attempts starting at ${MANAGER_PORT}`);
                        process.exit(2);
                    }
                    MANAGER_PORT++;
                    server.listen(MANAGER_PORT, () => console.log(`[MASTER] Management server listening (auto) on port ${MANAGER_PORT}`))
                        .on('error', (e) => {
                            if (e.code === 'EADDRINUSE') return tryNext();
                            console.error('[MASTER] Management server auto-port error:', e);
                            process.exit(2);
                        });
                };
                console.warn(`[MASTER] Port ${MANAGER_PORT} busy; auto-scan enabled (MANAGER_PORT_AUTO=1)`);
                tryNext();
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
