#!/usr/bin/env node

/**
 * YoRHa Legal AI - Full Stack Development Environment
 * 
 * Complete orchestration with:
 * - Node.js Clustering with GPU acceleration
 * - llama.cpp integration with Ollama
 * - WebAssembly modules with SIMD optimization
 * - Multi-layer caching (Loki.js + Redis + Qdrant + Fuse.js)
 * - XState state machine management
 * - RabbitMQ message queuing
 * - Real-time service workers
 * 
 * @author YoRHa Legal AI Team
 * @version 3.0.0
 */

import 'zx/globals';
import chalk from 'chalk';
import ora from 'ora';
import cluster from 'cluster';
import os from 'os';
import { WebSocket } from 'ws';
import fetch from 'node-fetch';
import { createMachine, interpret } from 'xstate';
import amqp from 'amqplib';
import Loki from 'lokijs';
import Fuse from 'fuse.js';
import { Worker } from 'worker_threads';

// Advanced Configuration
const CONFIG = {
  clustering: {
    enabled: true,
    workers: os.cpus().length,
    strategy: 'gpu_aware', // Balance GPU workloads across workers
    autoRestart: true,
    gracefulShutdown: 30000
  },
  
  services: {
    postgresql: {
      name: 'PostgreSQL + pgvector',
      port: 5432,
      healthUrl: null,
      command: '"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe" -h localhost -p 5432',
      priority: 1,
      cluster_mode: false
    },
    
    redis: {
      name: 'Redis Cache Cluster',
      port: 6379,
      healthUrl: null,
      command: '.\\redis-windows\\redis-server.exe --cluster-enabled yes --cluster-config-file nodes.conf',
      cwd: process.cwd(),
      priority: 2,
      cluster_mode: true
    },
    
    rabbitmq: {
      name: 'RabbitMQ Message Broker',
      port: 5672,
      healthUrl: 'http://localhost:15672/api/overview',
      command: 'rabbitmq-server',
      priority: 3,
      cluster_mode: false
    },
    
    ollama: {
      name: 'Ollama + llama.cpp',
      port: 11434,
      healthUrl: 'http://localhost:11434/api/version',
      command: 'ollama serve',
      priority: 4,
      env: {
        'LLAMA_CUBLAS': '1',
        'CUDA_VISIBLE_DEVICES': '0',
        'OLLAMA_GPU_OVERHEAD': '2GiB'
      }
    },
    
    qdrant: {
      name: 'Qdrant Vector DB',
      port: 6333,
      healthUrl: 'http://localhost:6333/health',
      command: '.\\qdrant-windows\\qdrant.exe --config-path ./qdrant-cluster.toml',
      cwd: process.cwd(),
      priority: 5,
      optional: true
    },
    
    goGPU: {
      name: 'GPU Legal AI Service',
      port: 8082,
      healthUrl: 'http://localhost:8082/health',
      command: 'go run gpu-accelerated-legal-service.go',
      cwd: path.join(process.cwd(), 'go-microservice'),
      priority: 6,
      env: {
        'REDIS_ADDR': 'localhost:6379',
        'RABBITMQ_URL': 'amqp://guest:guest@localhost:5672/',
        'OLLAMA_URL': 'http://localhost:11434',
        'CUDA_VISIBLE_DEVICES': '0'
      }
    },
    
    goSIMD: {
      name: 'Go SIMD Canvas Service',
      port: 8081,
      healthUrl: 'http://localhost:8081/health',
      command: 'go run go-ollama-simd.go',
      cwd: path.join(process.cwd(), 'go-microservice'),
      priority: 7,
      env: {
        'OLLAMA_URL': 'http://localhost:11434'
      }
    },
    
    rustTagger: {
      name: 'Rust Auto-Tagger (WASM)',
      port: 8083,
      healthUrl: 'http://localhost:8083/health',
      command: 'cargo run --release --target=wasm32-wasi',
      cwd: path.join(process.cwd(), 'rust-tagger'),
      priority: 8,
      optional: true
    },
    
    sveltekit: {
      name: 'SvelteKit Frontend Cluster',
      port: 5173,
      healthUrl: 'http://localhost:5173/',
      command: 'npm run dev',
      cwd: path.join(process.cwd(), 'sveltekit-frontend'),
      priority: 9,
      cluster_mode: true
    }
  },
  
  caching: {
    layers: [
      { name: 'Loki.js', type: 'memory', ttl: 300 },
      { name: 'Redis', type: 'distributed', ttl: 3600 },
      { name: 'Qdrant', type: 'vector', ttl: 86400 },
      { name: 'Fuse.js', type: 'search', ttl: 1800 }
    ],
    strategy: 'write_through',
    compression: true
  },
  
  gpu: {
    enabled: true,
    devices: ['0'], // CUDA device IDs
    memory_fraction: 0.8,
    allow_growth: true,
    optimization: {
      mixed_precision: true,
      tensor_fusion: true,
      kernel_optimization: true
    }
  },
  
  monitoring: {
    enabled: true,
    metrics_port: 9090,
    healthcheck_interval: 5000,
    performance_tracking: true
  }
};

// XState Machine for Service Orchestration
const serviceMachine = createMachine({
  id: 'serviceOrchestrator',
  initial: 'initializing',
  context: {
    services: new Map(),
    workers: new Map(),
    cache: null,
    messageQueue: null,
    errorCount: 0
  },
  
  states: {
    initializing: {
      entry: 'initializeSystem',
      on: {
        INIT_COMPLETE: 'starting_services',
        INIT_FAILED: 'error'
      }
    },
    
    starting_services: {
      entry: 'startAllServices',
      on: {
        SERVICES_STARTED: 'running',
        SERVICE_FAILED: 'recovering'
      }
    },
    
    running: {
      entry: 'startMonitoring',
      on: {
        SERVICE_DOWN: 'recovering',
        SCALE_UP: 'scaling',
        SHUTDOWN: 'stopping'
      }
    },
    
    scaling: {
      entry: 'scaleServices',
      on: {
        SCALE_COMPLETE: 'running',
        SCALE_FAILED: 'error'
      }
    },
    
    recovering: {
      entry: 'recoverService',
      on: {
        RECOVERY_COMPLETE: 'running',
        RECOVERY_FAILED: 'error'
      }
    },
    
    stopping: {
      entry: 'stopAllServices',
      on: {
        STOP_COMPLETE: 'stopped'
      }
    },
    
    error: {
      entry: 'handleError',
      on: {
        RETRY: 'initializing',
        SHUTDOWN: 'stopping'
      }
    },
    
    stopped: {
      type: 'final'
    }
  }
});

// Global State Management
const state = {
  services: new Map(),
  workers: new Map(),
  cache: new Map(),
  messageQueue: null,
  stateMachine: null,
  lokiDB: null,
  fuseIndex: null,
  startTime: Date.now(),
  metrics: {
    requests: 0,
    errors: 0,
    gpu_utilization: 0,
    cache_hits: 0,
    cache_misses: 0
  }
};

// Advanced Logging with Performance Metrics
const log = {
  info: (msg, meta = {}) => {
    const timestamp = new Date().toISOString();
    const worker = cluster.worker?.id || 'master';
    console.log(chalk.blue('ℹ'), `[${timestamp}][W${worker}]`, msg, meta.perf ? chalk.gray(`(${meta.perf}ms)`) : '');
  },
  
  success: (msg, meta = {}) => {
    const timestamp = new Date().toISOString();
    const worker = cluster.worker?.id || 'master';
    console.log(chalk.green('✓'), `[${timestamp}][W${worker}]`, msg, meta.perf ? chalk.gray(`(${meta.perf}ms)`) : '');
  },
  
  error: (msg, meta = {}) => {
    const timestamp = new Date().toISOString();
    const worker = cluster.worker?.id || 'master';
    console.log(chalk.red('✗'), `[${timestamp}][W${worker}]`, msg);
    state.metrics.errors++;
    if (meta.stack) console.log(chalk.gray(meta.stack));
  },
  
  warn: (msg, meta = {}) => {
    const timestamp = new Date().toISOString();
    const worker = cluster.worker?.id || 'master';
    console.log(chalk.yellow('⚠'), `[${timestamp}][W${worker}]`, msg);
  },
  
  debug: (msg, meta = {}) => {
    if (process.env.DEBUG) {
      const timestamp = new Date().toISOString();
      const worker = cluster.worker?.id || 'master';
      console.log(chalk.gray('🔍'), `[${timestamp}][W${worker}]`, msg);
    }
  },
  
  gpu: (msg, utilization = 0) => {
    const timestamp = new Date().toISOString();
    const worker = cluster.worker?.id || 'master';
    console.log(chalk.magenta('🚀'), `[${timestamp}][W${worker}]`, msg, chalk.cyan(`GPU: ${utilization}%`));
    state.metrics.gpu_utilization = utilization;
  }
};

// Multi-Layer Caching System
class MultiLayerCache {
  constructor() {
    this.layers = new Map();
    this.stats = {
      hits: { loki: 0, redis: 0, qdrant: 0, fuse: 0 },
      misses: { loki: 0, redis: 0, qdrant: 0, fuse: 0 }
    };
    
    this.initializeLayers();
  }
  
  initializeLayers() {
    // Loki.js in-memory database
    this.lokiDB = new Loki('legal-ai-cache.db', {
      adapter: new loki.LokiMemoryAdapter(),
      autoload: true,
      autosave: true,
      autosaveInterval: 1000
    });
    
    this.embeddings = this.lokiDB.addCollection('embeddings', {
      ttl: 300000, // 5 minutes
      ttlInterval: 60000 // Check every minute
    });
    
    this.documents = this.lokiDB.addCollection('documents', {
      ttl: 1800000, // 30 minutes
      ttlInterval: 300000 // Check every 5 minutes
    });
    
    // Fuse.js search index
    this.searchIndex = new Map();
    this.layers.set('loki', this.lokiDB);
    this.layers.set('fuse', this.searchIndex);
  }
  
  async get(key, layer = 'auto') {
    const startTime = Date.now();
    
    if (layer === 'auto') {
      // Try layers in order of speed
      const layers = ['loki', 'redis', 'qdrant', 'fuse'];
      
      for (const layerName of layers) {
        try {
          const result = await this.getFromLayer(key, layerName);
          if (result !== null) {
            this.stats.hits[layerName]++;
            state.metrics.cache_hits++;
            log.debug(`Cache hit in ${layerName} for key: ${key}`, { perf: Date.now() - startTime });
            return result;
          }
        } catch (error) {
          log.debug(`Cache miss in ${layerName} for key: ${key}`);
        }
      }
      
      state.metrics.cache_misses++;
      return null;
    } else {
      return await this.getFromLayer(key, layer);
    }
  }
  
  async set(key, value, ttl = 3600, layer = 'auto') {
    if (layer === 'auto') {
      // Write to all layers
      const promises = [];
      
      // Loki.js
      promises.push(this.setInLayer(key, value, ttl, 'loki'));
      
      // Redis (if available)
      if (state.services.has('redis')) {
        promises.push(this.setInLayer(key, value, ttl, 'redis'));
      }
      
      // Qdrant (for vector data)
      if (typeof value === 'object' && value.embedding && state.services.has('qdrant')) {
        promises.push(this.setInLayer(key, value, ttl, 'qdrant'));
      }
      
      // Fuse.js (for searchable text)
      if (typeof value === 'object' && value.text) {
        promises.push(this.setInLayer(key, value, ttl, 'fuse'));
      }
      
      await Promise.allSettled(promises);
    } else {
      await this.setInLayer(key, value, ttl, layer);
    }
  }
  
  async getFromLayer(key, layer) {
    switch (layer) {
      case 'loki':
        const doc = this.embeddings.findOne({ key: key }) || this.documents.findOne({ key: key });
        return doc ? doc.value : null;
        
      case 'redis':
        // Redis integration would go here
        return null;
        
      case 'qdrant':
        // Qdrant vector search would go here
        return null;
        
      case 'fuse':
        return this.searchIndex.get(key) || null;
        
      default:
        return null;
    }
  }
  
  async setInLayer(key, value, ttl, layer) {
    switch (layer) {
      case 'loki':
        if (typeof value === 'object' && value.embedding) {
          this.embeddings.insert({ key, value, timestamp: Date.now() });
        } else {
          this.documents.insert({ key, value, timestamp: Date.now() });
        }
        break;
        
      case 'fuse':
        this.searchIndex.set(key, value);
        if (typeof value === 'object' && value.text) {
          this.updateFuseIndex(key, value);
        }
        break;
    }
  }
  
  updateFuseIndex(key, data) {
    if (!this.fuseSearcher) {
      this.fuseSearcher = new Fuse([], {
        keys: ['text', 'title', 'content'],
        threshold: 0.3,
        includeScore: true
      });
    }
    
    this.fuseSearcher.add({ id: key, ...data });
  }
  
  search(query, options = {}) {
    if (this.fuseSearcher) {
      return this.fuseSearcher.search(query, options);
    }
    return [];
  }
  
  getStats() {
    return {
      ...this.stats,
      total_hits: Object.values(this.stats.hits).reduce((sum, hits) => sum + hits, 0),
      total_misses: Object.values(this.stats.misses).reduce((sum, misses) => sum + misses, 0),
      hit_ratio: state.metrics.cache_hits / (state.metrics.cache_hits + state.metrics.cache_misses) || 0
    };
  }
}

// GPU-Aware Worker Management
class GPUWorkerManager {
  constructor() {
    this.workers = new Map();
    this.gpuQueue = [];
    this.cpuQueue = [];
    this.currentGPUWorker = 0;
  }
  
  createWorkers() {
    const numWorkers = CONFIG.clustering.workers;
    const gpuWorkers = Math.ceil(numWorkers * 0.3); // 30% GPU workers
    
    for (let i = 0; i < numWorkers; i++) {
      const isGPUWorker = i < gpuWorkers;
      const worker = cluster.fork({
        WORKER_TYPE: isGPUWorker ? 'gpu' : 'cpu',
        WORKER_ID: i,
        CUDA_VISIBLE_DEVICES: isGPUWorker ? CONFIG.gpu.devices[i % CONFIG.gpu.devices.length] : '',
        GPU_MEMORY_FRACTION: isGPUWorker ? CONFIG.gpu.memory_fraction : 0
      });
      
      this.workers.set(worker.id, {
        worker,
        type: isGPUWorker ? 'gpu' : 'cpu',
        busy: false,
        tasks: 0,
        gpuUtilization: 0
      });
      
      worker.on('message', (msg) => this.handleWorkerMessage(worker.id, msg));
      worker.on('exit', () => this.handleWorkerExit(worker.id));
      
      log.info(`Worker ${worker.id} created (${isGPUWorker ? 'GPU' : 'CPU'} mode)`);
    }
  }
  
  assignTask(task, preferGPU = false) {
    const availableWorkers = Array.from(this.workers.values())
      .filter(w => !w.busy)
      .sort((a, b) => a.tasks - b.tasks);
    
    let selectedWorker;
    
    if (preferGPU) {
      selectedWorker = availableWorkers.find(w => w.type === 'gpu') || availableWorkers[0];
    } else {
      selectedWorker = availableWorkers[0];
    }
    
    if (!selectedWorker) {
      if (preferGPU) {
        this.gpuQueue.push(task);
      } else {
        this.cpuQueue.push(task);
      }
      return false;
    }
    
    selectedWorker.busy = true;
    selectedWorker.tasks++;
    selectedWorker.worker.send({ type: 'TASK', data: task });
    
    log.debug(`Task assigned to worker ${selectedWorker.worker.id} (${selectedWorker.type})`);
    return true;
  }
  
  handleWorkerMessage(workerId, message) {
    const workerInfo = this.workers.get(workerId);
    if (!workerInfo) return;
    
    switch (message.type) {
      case 'TASK_COMPLETE':
        workerInfo.busy = false;
        if (message.data.gpuUtilization) {
          workerInfo.gpuUtilization = message.data.gpuUtilization;
        }
        
        // Process queued tasks
        const nextTask = workerInfo.type === 'gpu' ? this.gpuQueue.shift() : this.cpuQueue.shift();
        if (nextTask) {
          this.assignTask(nextTask, workerInfo.type === 'gpu');
        }
        break;
        
      case 'HEALTH_UPDATE':
        workerInfo.health = message.data;
        break;
    }
  }
  
  handleWorkerExit(workerId) {
    if (CONFIG.clustering.autoRestart) {
      log.warn(`Worker ${workerId} died, restarting...`);
      const workerInfo = this.workers.get(workerId);
      if (workerInfo) {
        const newWorker = cluster.fork({
          WORKER_TYPE: workerInfo.type,
          WORKER_ID: workerId,
          CUDA_VISIBLE_DEVICES: workerInfo.type === 'gpu' ? CONFIG.gpu.devices[0] : '',
          GPU_MEMORY_FRACTION: workerInfo.type === 'gpu' ? CONFIG.gpu.memory_fraction : 0
        });
        
        this.workers.set(newWorker.id, {
          ...workerInfo,
          worker: newWorker,
          busy: false
        });
      }
    }
    
    this.workers.delete(workerId);
  }
  
  getStats() {
    const workers = Array.from(this.workers.values());
    return {
      total: workers.length,
      gpu: workers.filter(w => w.type === 'gpu').length,
      cpu: workers.filter(w => w.type === 'cpu').length,
      busy: workers.filter(w => w.busy).length,
      queued_gpu: this.gpuQueue.length,
      queued_cpu: this.cpuQueue.length,
      avg_gpu_utilization: workers
        .filter(w => w.type === 'gpu')
        .reduce((sum, w) => sum + w.gpuUtilization, 0) / workers.filter(w => w.type === 'gpu').length || 0
    };
  }
}

// Message Queue Integration
class MessageQueueManager {
  constructor() {
    this.connection = null;
    this.channel = null;
    this.queues = new Map();
    this.exchanges = new Map();
  }
  
  async connect() {
    try {
      this.connection = await amqp.connect(CONFIG.services.rabbitmq ? 'amqp://localhost:5672' : null);
      this.channel = await this.connection.createChannel();
      
      await this.setupQueues();
      log.success('RabbitMQ connected and configured');
    } catch (error) {
      log.error('Failed to connect to RabbitMQ', { stack: error.stack });
      throw error;
    }
  }
  
  async setupQueues() {
    // Define exchanges
    const exchanges = [
      { name: 'legal.direct', type: 'direct' },
      { name: 'legal.topic', type: 'topic' },
      { name: 'legal.fanout', type: 'fanout' },
      { name: 'gpu.processing', type: 'direct' }
    ];
    
    for (const exchange of exchanges) {
      await this.channel.assertExchange(exchange.name, exchange.type, { durable: true });
      this.exchanges.set(exchange.name, exchange);
    }
    
    // Define queues
    const queues = [
      'gpu.tasks',
      'cpu.tasks', 
      'embedding.generation',
      'document.analysis',
      'cache.invalidation',
      'search.indexing',
      'legal.entity.extraction',
      'canvas.analysis'
    ];
    
    for (const queueName of queues) {
      const queue = await this.channel.assertQueue(queueName, {
        durable: true,
        arguments: { 'x-max-priority': 10 }
      });
      this.queues.set(queueName, queue);
    }
  }
  
  async publishTask(queueName, task, priority = 0) {
    const message = Buffer.from(JSON.stringify({
      ...task,
      id: task.id || `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      priority
    }));
    
    return this.channel.sendToQueue(queueName, message, {
      persistent: true,
      priority: priority
    });
  }
  
  async consumeTask(queueName, handler, options = {}) {
    await this.channel.prefetch(options.prefetch || 1);
    
    return this.channel.consume(queueName, async (msg) => {
      if (msg) {
        try {
          const task = JSON.parse(msg.content.toString());
          const startTime = Date.now();
          
          const result = await handler(task);
          
          const processingTime = Date.now() - startTime;
          log.debug(`Task ${task.id} processed in ${processingTime}ms`);
          
          this.channel.ack(msg);
        } catch (error) {
          log.error(`Task processing failed: ${error.message}`);
          this.channel.nack(msg, false, true); // Requeue
        }
      }
    }, options);
  }
}

// Service Health Monitoring
class HealthMonitor {
  constructor() {
    this.checks = new Map();
    this.interval = null;
    this.metrics = {
      uptime: 0,
      checks_passed: 0,
      checks_failed: 0,
      response_times: []
    };
  }
  
  addCheck(name, checkFn, interval = 5000) {
    this.checks.set(name, {
      fn: checkFn,
      interval,
      lastCheck: 0,
      status: 'unknown',
      responseTime: 0
    });
  }
  
  start() {
    this.interval = setInterval(() => {
      this.runAllChecks();
    }, CONFIG.monitoring.healthcheck_interval);
    
    log.info('Health monitoring started');
  }
  
  async runAllChecks() {
    const checks = Array.from(this.checks.entries());
    const results = await Promise.allSettled(
      checks.map(([name, check]) => this.runCheck(name, check))
    );
    
    let passed = 0, failed = 0;
    results.forEach((result, index) => {
      const [name] = checks[index];
      if (result.status === 'fulfilled' && result.value) {
        passed++;
      } else {
        failed++;
        log.warn(`Health check failed: ${name}`);
      }
    });
    
    this.metrics.checks_passed = passed;
    this.metrics.checks_failed = failed;
    this.metrics.uptime = Date.now() - state.startTime;
  }
  
  async runCheck(name, check) {
    const startTime = Date.now();
    
    try {
      const result = await check.fn();
      check.status = result ? 'healthy' : 'unhealthy';
      check.responseTime = Date.now() - startTime;
      check.lastCheck = Date.now();
      
      if (check.responseTime > 1000) {
        log.warn(`Slow health check: ${name} (${check.responseTime}ms)`);
      }
      
      return result;
    } catch (error) {
      check.status = 'error';
      check.responseTime = Date.now() - startTime;
      check.lastCheck = Date.now();
      log.debug(`Health check error for ${name}: ${error.message}`);
      return false;
    }
  }
  
  getStatus() {
    const checks = Object.fromEntries(
      Array.from(this.checks.entries()).map(([name, check]) => [
        name, 
        {
          status: check.status,
          responseTime: check.responseTime,
          lastCheck: new Date(check.lastCheck).toISOString()
        }
      ])
    );
    
    return {
      overall: this.metrics.checks_failed === 0 ? 'healthy' : 'degraded',
      uptime: this.metrics.uptime,
      checks,
      metrics: this.metrics
    };
  }
}

// XState Service Actions
const serviceActions = {
  initializeSystem: (context) => {
    log.info('Initializing YoRHa Legal AI System...');
    
    // Initialize multi-layer cache
    state.cache = new MultiLayerCache();
    
    // Initialize worker manager
    if (CONFIG.clustering.enabled && cluster.isMaster) {
      state.workerManager = new GPUWorkerManager();
    }
    
    // Initialize message queue
    state.messageQueue = new MessageQueueManager();
    
    // Initialize health monitor
    state.healthMonitor = new HealthMonitor();
    
    log.success('System initialization complete');
    return { type: 'INIT_COMPLETE' };
  },
  
  startAllServices: async (context) => {
    log.info('Starting all services...');
    
    const sortedServices = Object.entries(CONFIG.services)
      .filter(([_, config]) => !config.optional || process.argv.includes('--include-optional'))
      .sort(([, a], [, b]) => a.priority - b.priority);
    
    for (const [serviceName, config] of sortedServices) {
      await startService(serviceName, config);
      await sleep(2000); // Stagger starts
    }
    
    // Start workers if clustering enabled
    if (CONFIG.clustering.enabled && cluster.isMaster) {
      state.workerManager.createWorkers();
    }
    
    // Connect to message queue
    if (CONFIG.services.rabbitmq) {
      try {
        await state.messageQueue.connect();
      } catch (error) {
        log.warn('RabbitMQ not available, continuing without message queue');
      }
    }
    
    // Start health monitoring
    setupHealthChecks();
    state.healthMonitor.start();
    
    log.success('All services started successfully');
    return { type: 'SERVICES_STARTED' };
  },
  
  startMonitoring: (context) => {
    log.info('Starting system monitoring...');
    
    // Start performance metrics collection
    setInterval(() => {
      collectMetrics();
    }, 10000); // Every 10 seconds
    
    // Start GPU monitoring
    if (CONFIG.gpu.enabled) {
      setInterval(() => {
        monitorGPU();
      }, 5000); // Every 5 seconds
    }
    
    log.success('System monitoring active');
  },
  
  scaleServices: async (context) => {
    log.info('Scaling services...');
    
    if (state.workerManager) {
      const stats = state.workerManager.getStats();
      
      if (stats.queued_gpu > 5 || stats.queued_cpu > 10) {
        log.info('High queue detected, scaling up workers...');
        // Scale up logic would go here
      }
    }
    
    return { type: 'SCALE_COMPLETE' };
  },
  
  recoverService: async (context, event) => {
    log.warn(`Recovering service: ${event.service}`);
    
    const config = CONFIG.services[event.service];
    if (config) {
      await startService(event.service, config);
      log.success(`Service ${event.service} recovered`);
    }
    
    return { type: 'RECOVERY_COMPLETE' };
  },
  
  stopAllServices: async (context) => {
    log.info('Stopping all services...');
    
    // Stop health monitoring
    if (state.healthMonitor?.interval) {
      clearInterval(state.healthMonitor.interval);
    }
    
    // Graceful shutdown of workers
    if (state.workerManager) {
      for (const [id, worker] of state.workerManager.workers) {
        worker.worker.send({ type: 'SHUTDOWN' });
      }
    }
    
    // Close message queue connection
    if (state.messageQueue?.connection) {
      await state.messageQueue.connection.close();
    }
    
    // Stop all service processes
    for (const [name, serviceInfo] of state.services) {
      if (serviceInfo.process && !serviceInfo.process.killed) {
        log.info(`Stopping ${name}...`);
        serviceInfo.process.kill('SIGTERM');
        
        // Force kill after grace period
        setTimeout(() => {
          if (!serviceInfo.process.killed) {
            serviceInfo.process.kill('SIGKILL');
          }
        }, CONFIG.clustering.gracefulShutdown);
      }
    }
    
    log.success('All services stopped');
    return { type: 'STOP_COMPLETE' };
  },
  
  handleError: (context, event) => {
    log.error(`System error: ${event.error}`);
    
    context.errorCount++;
    
    if (context.errorCount > 3) {
      log.error('Too many errors, initiating shutdown');
      return { type: 'SHUTDOWN' };
    }
    
    log.info('Attempting system recovery...');
    return { type: 'RETRY' };
  }
};

// Service Management Functions
async function startService(serviceName, config) {
  const spinner = ora(`Starting ${config.name}...`).start();
  
  try {
    // Check if already running
    if (await isServiceHealthy(serviceName, config)) {
      spinner.succeed(`${config.name} already running`);
      return;
    }

    // Prepare environment
    const env = { ...process.env, ...config.env };
    
    const options = {
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      env
    };

    if (config.cwd) {
      options.cwd = config.cwd;
    }

    let proc;
    if (config.command.includes('.exe') || config.command.includes('ollama') || config.command.includes('rabbitmq')) {
      // For executables and special services
      const parts = config.command.split(' ');
      proc = spawn(parts[0], parts.slice(1), options);
    } else {
      // For other commands, use shell
      proc = spawn('cmd', ['/c', config.command], { ...options, shell: true });
    }

    if (proc.pid) {
      state.services.set(serviceName, { 
        process: proc, 
        config,
        startTime: Date.now(),
        restarts: 0
      });
      
      // Set up process monitoring
      proc.on('exit', (code) => {
        if (code !== 0 && CONFIG.clustering.autoRestart) {
          log.warn(`Service ${serviceName} exited with code ${code}, restarting...`);
          setTimeout(() => startService(serviceName, config), 5000);
        }
      });
      
      // Wait for service to be healthy
      let attempts = 0;
      const maxAttempts = 30;
      
      while (attempts < maxAttempts) {
        await sleep(1000);
        if (await isServiceHealthy(serviceName, config)) {
          const startupTime = Date.now() - state.services.get(serviceName).startTime;
          spinner.succeed(`${config.name} started successfully on port ${config.port} (${startupTime}ms)`);
          
          // Log GPU usage if applicable
          if (config.env && config.env.CUDA_VISIBLE_DEVICES) {
            log.gpu(`${config.name} using GPU ${config.env.CUDA_VISIBLE_DEVICES}`);
          }
          
          return;
        }
        attempts++;
      }
      
      spinner.warn(`${config.name} started but health check failed`);
    } else {
      throw new Error('Failed to start process');
    }
  } catch (error) {
    spinner.fail(`Failed to start ${config.name}: ${error.message}`);
    throw error;
  }
}

// Enhanced Health Checks
const healthCheckers = {
  async postgresql() {
    try {
      const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe" -h localhost -p 5432`;
      return result.exitCode === 0;
    } catch {
      return false;
    }
  },

  async redis() {
    try {
      const result = await $`echo "ping" | .\\redis-windows\\redis-cli.exe -h localhost -p 6379`;
      return result.stdout.includes('PONG');
    } catch {
      return false;
    }
  },

  async rabbitmq() {
    try {
      const response = await fetch('http://localhost:15672/api/overview', {
        timeout: 5000,
        headers: { 'Authorization': 'Basic ' + Buffer.from('guest:guest').toString('base64') }
      });
      return response.ok;
    } catch {
      return false;
    }
  },

  async gpu(deviceId = 0) {
    try {
      // Check CUDA availability
      const result = await $`nvidia-smi -i ${deviceId} --query-gpu=utilization.gpu --format=csv,noheader,nounits`;
      const utilization = parseInt(result.stdout.trim());
      return !isNaN(utilization) && utilization >= 0;
    } catch {
      return false;
    }
  },

  async http(url) {
    try {
      const response = await fetch(url, { 
        timeout: 5000,
        signal: AbortSignal.timeout(5000)
      });
      return response.ok;
    } catch {
      return false;
    }
  }
};

async function isServiceHealthy(serviceName, config) {
  try {
    switch (serviceName) {
      case 'postgresql':
        return await healthCheckers.postgresql();
      case 'redis':
        return await healthCheckers.redis();
      case 'rabbitmq':
        return await healthCheckers.rabbitmq();
      default:
        if (config.healthUrl) {
          return await healthCheckers.http(config.healthUrl);
        }
        return false;
    }
  } catch {
    return false;
  }
}

function setupHealthChecks() {
  const monitor = state.healthMonitor;
  
  // Add health checks for all services
  for (const [name, config] of Object.entries(CONFIG.services)) {
    if (state.services.has(name)) {
      monitor.addCheck(name, () => isServiceHealthy(name, config));
    }
  }
  
  // Add GPU health check
  if (CONFIG.gpu.enabled) {
    monitor.addCheck('gpu', () => healthCheckers.gpu(0));
  }
  
  // Add cache health check
  monitor.addCheck('cache', () => {
    return state.cache && state.cache.layers.size > 0;
  });
  
  // Add message queue health check
  if (CONFIG.services.rabbitmq) {
    monitor.addCheck('rabbitmq', () => healthCheckers.rabbitmq());
  }
}

async function collectMetrics() {
  try {
    // Collect GPU metrics
    if (CONFIG.gpu.enabled) {
      const gpuStats = await getGPUStats();
      state.metrics.gpu_utilization = gpuStats.utilization;
    }
    
    // Collect cache metrics
    if (state.cache) {
      const cacheStats = state.cache.getStats();
      state.metrics.cache_hits = cacheStats.total_hits;
      state.metrics.cache_misses = cacheStats.total_misses;
    }
    
    // Collect worker metrics
    if (state.workerManager) {
      const workerStats = state.workerManager.getStats();
      log.debug(`Workers: ${workerStats.total} total, ${workerStats.busy} busy, GPU util: ${workerStats.avg_gpu_utilization.toFixed(1)}%`);
    }
    
    // Log performance summary
    const uptime = Date.now() - state.startTime;
    const hours = Math.floor(uptime / 3600000);
    const minutes = Math.floor((uptime % 3600000) / 60000);
    
    log.info(`System metrics - Uptime: ${hours}h${minutes}m, Requests: ${state.metrics.requests}, Errors: ${state.metrics.errors}, GPU: ${state.metrics.gpu_utilization}%`);
    
  } catch (error) {
    log.debug(`Metrics collection failed: ${error.message}`);
  }
}

async function getGPUStats() {
  try {
    const result = await $`nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits`;
    const [utilization, memUsed, memTotal, temperature] = result.stdout.trim().split(', ').map(Number);
    
    return {
      utilization,
      memory: {
        used: memUsed,
        total: memTotal,
        usage: (memUsed / memTotal) * 100
      },
      temperature
    };
  } catch {
    return { utilization: 0, memory: { used: 0, total: 0, usage: 0 }, temperature: 0 };
  }
}

async function monitorGPU() {
  try {
    const stats = await getGPUStats();
    
    if (stats.utilization > 90) {
      log.warn(`High GPU utilization: ${stats.utilization}%`);
    }
    
    if (stats.memory.usage > 90) {
      log.warn(`High GPU memory usage: ${stats.memory.usage.toFixed(1)}%`);
    }
    
    if (stats.temperature > 80) {
      log.warn(`High GPU temperature: ${stats.temperature}°C`);
    }
    
    log.gpu(`GPU Stats: ${stats.utilization}% util, ${stats.memory.usage.toFixed(1)}% mem, ${stats.temperature}°C`);
    
  } catch (error) {
    log.debug(`GPU monitoring failed: ${error.message}`);
  }
}

// Worker Process Logic
function runWorker() {
  const workerType = process.env.WORKER_TYPE;
  const workerId = process.env.WORKER_ID;
  const cudaDevice = process.env.CUDA_VISIBLE_DEVICES;
  
  log.info(`Worker ${workerId} started (${workerType} mode)${cudaDevice ? `, GPU: ${cudaDevice}` : ''}`);
  
  // Worker message handling
  process.on('message', async (msg) => {
    switch (msg.type) {
      case 'TASK':
        await handleWorkerTask(msg.data, workerType);
        break;
      case 'SHUTDOWN':
        log.info(`Worker ${workerId} shutting down gracefully...`);
        process.exit(0);
        break;
    }
  });
  
  // Send periodic health updates
  setInterval(() => {
    process.send({
      type: 'HEALTH_UPDATE',
      data: {
        workerId,
        type: workerType,
        memory: process.memoryUsage(),
        uptime: process.uptime()
      }
    });
  }, 30000);
}

async function handleWorkerTask(task, workerType) {
  const startTime = Date.now();
  
  try {
    let result;
    
    if (workerType === 'gpu' && task.requiresGPU) {
      // GPU-specific task processing
      result = await processGPUTask(task);
    } else {
      // CPU task processing
      result = await processCPUTask(task);
    }
    
    const processingTime = Date.now() - startTime;
    
    // Get GPU utilization if available
    let gpuUtilization = 0;
    if (workerType === 'gpu') {
      try {
        const gpuStats = await getGPUStats();
        gpuUtilization = gpuStats.utilization;
      } catch {
        // GPU stats not available
      }
    }
    
    process.send({
      type: 'TASK_COMPLETE',
      data: {
        taskId: task.id,
        result,
        processingTime,
        gpuUtilization
      }
    });
    
  } catch (error) {
    process.send({
      type: 'TASK_ERROR',
      data: {
        taskId: task.id,
        error: error.message
      }
    });
  }
}

async function processGPUTask(task) {
  // GPU task processing logic
  log.gpu(`Processing GPU task: ${task.type}`);
  
  switch (task.type) {
    case 'embedding_generation':
      return await generateEmbeddingGPU(task.data);
    case 'tensor_operation':
      return await processTensorOperationGPU(task.data);
    case 'legal_analysis':
      return await analyzeLegalDocumentGPU(task.data);
    default:
      throw new Error(`Unknown GPU task type: ${task.type}`);
  }
}

async function processCPUTask(task) {
  // CPU task processing logic
  log.debug(`Processing CPU task: ${task.type}`);
  
  switch (task.type) {
    case 'text_processing':
      return await processTextCPU(task.data);
    case 'search_indexing':
      return await indexDocumentCPU(task.data);
    case 'cache_operation':
      return await processCacheOperationCPU(task.data);
    default:
      throw new Error(`Unknown CPU task type: ${task.type}`);
  }
}

// Placeholder task processing functions
async function generateEmbeddingGPU(data) {
  // Simulate GPU embedding generation
  await sleep(100);
  return { embedding: new Array(384).fill(0).map(() => Math.random()), dimensions: 384 };
}

async function processTensorOperationGPU(data) {
  // Simulate GPU tensor operation
  await sleep(50);
  return { result: 'tensor_processed', computation_time: 50 };
}

async function analyzeLegalDocumentGPU(data) {
  // Simulate GPU-accelerated legal document analysis
  await sleep(200);
  return { 
    entities: ['plaintiff', 'defendant', 'contract'],
    sentiment: 0.65,
    complexity_score: 0.8
  };
}

async function processTextCPU(data) {
  // Simulate CPU text processing
  await sleep(150);
  return { processed: true, word_count: data.text?.split(' ').length || 0 };
}

async function indexDocumentCPU(data) {
  // Simulate document indexing
  await sleep(100);
  return { indexed: true, document_id: data.id };
}

async function processCacheOperationCPU(data) {
  // Simulate cache operation
  await sleep(10);
  return { cached: true, key: data.key };
}

// Main Orchestration
async function main() {
  console.log(chalk.cyan.bold('🤖 YoRHa Legal AI - Full Stack Development Environment v3.0\n'));
  
  // Check if we're in worker mode
  if (!cluster.isMaster && process.env.WORKER_TYPE) {
    return runWorker();
  }
  
  // Master process initialization
  log.info('Initializing system architecture...');
  
  // Check prerequisites
  await checkPrerequisites();
  
  // Initialize XState machine
  const serviceInterpreter = interpret(serviceMachine.withConfig({
    actions: serviceActions
  }));
  
  state.stateMachine = serviceInterpreter;
  
  // Start the state machine
  serviceInterpreter.start();
  
  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    log.info('\n🛑 Initiating graceful shutdown...');
    serviceInterpreter.send({ type: 'SHUTDOWN' });
    
    setTimeout(() => {
      log.error('Force shutdown after timeout');
      process.exit(1);
    }, CONFIG.clustering.gracefulShutdown);
  });
  
  // Handle uncaught exceptions
  process.on('uncaughtException', (error) => {
    log.error('Uncaught exception:', { stack: error.stack });
    serviceInterpreter.send({ type: 'SHUTDOWN' });
  });
  
  // Handle unhandled promise rejections
  process.on('unhandledRejection', (reason) => {
    log.error('Unhandled rejection:', { stack: reason });
  });
  
  // Final status display
  setTimeout(async () => {
    await displaySystemStatus();
  }, 5000);
}

async function checkPrerequisites() {
  const spinner = ora('Checking prerequisites...').start();
  
  const checks = [
    { name: 'Node.js', check: () => $.which('node') },
    { name: 'PostgreSQL', check: () => fs.existsSync('C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe') },
    { name: 'Go toolchain', check: () => $.which('go') },
    { name: 'Redis', check: () => fs.existsSync('./redis-windows/redis-server.exe') },
    { name: 'Ollama', check: () => $.which('ollama') },
    { name: 'CUDA Runtime', check: () => $.which('nvidia-smi') },
    { name: 'RabbitMQ', check: () => $.which('rabbitmq-server') }
  ];
  
  const failed = [];
  
  for (const check of checks) {
    try {
      await check.check();
    } catch {
      failed.push(check.name);
    }
  }
  
  if (failed.length > 0) {
    spinner.fail(`Missing prerequisites: ${failed.join(', ')}`);
    console.log(chalk.yellow('\n📋 Setup Guide:'));
    console.log('  1. Install PostgreSQL 17 with pgvector extension');
    console.log('  2. Install Redis for Windows');
    console.log('  3. Install Ollama from https://ollama.com');
    console.log('  4. Install CUDA Toolkit and cuDNN');
    console.log('  5. Install RabbitMQ Server');
    console.log('  6. Build Go microservices: go build');
    process.exit(1);
  }
  
  spinner.succeed('Prerequisites check passed');
}

async function displaySystemStatus() {
  console.log(chalk.cyan('\n📊 System Status Dashboard:'));
  
  // Service status
  if (state.healthMonitor) {
    const healthStatus = state.healthMonitor.getStatus();
    console.log(`  Overall Health: ${healthStatus.overall === 'healthy' ? chalk.green('✓ HEALTHY') : chalk.red('✗ DEGRADED')}`);
    console.log(`  Uptime: ${Math.floor(healthStatus.uptime / 60000)}m${Math.floor((healthStatus.uptime % 60000) / 1000)}s`);
  }
  
  // Worker status
  if (state.workerManager) {
    const workerStats = state.workerManager.getStats();
    console.log(`  Workers: ${workerStats.total} total (${workerStats.gpu} GPU, ${workerStats.cpu} CPU)`);
    console.log(`  Queue: ${workerStats.queued_gpu} GPU tasks, ${workerStats.queued_cpu} CPU tasks`);
  }
  
  // Cache status
  if (state.cache) {
    const cacheStats = state.cache.getStats();
    console.log(`  Cache Hit Ratio: ${(cacheStats.hit_ratio * 100).toFixed(1)}%`);
  }
  
  // GPU status
  try {
    const gpuStats = await getGPUStats();
    console.log(`  GPU Utilization: ${gpuStats.utilization}%`);
    console.log(`  GPU Memory: ${gpuStats.memory.usage.toFixed(1)}% (${gpuStats.memory.used}MB/${gpuStats.memory.total}MB)`);
    console.log(`  GPU Temperature: ${gpuStats.temperature}°C`);
  } catch {
    console.log(`  GPU Status: ${chalk.yellow('Not Available')}`);
  }
  
  // Service URLs
  console.log(chalk.cyan('\n🌐 Service URLs:'));
  console.log(`  Frontend:          ${chalk.blue('http://localhost:5173')}`);
  console.log(`  GPU Legal AI:      ${chalk.blue('http://localhost:8082')}`);
  console.log(`  SIMD Canvas:       ${chalk.blue('http://localhost:8081')}`);
  console.log(`  Ollama:            ${chalk.blue('http://localhost:11434')}`);
  console.log(`  Qdrant:            ${chalk.blue('http://localhost:6333')}`);
  console.log(`  RabbitMQ Management: ${chalk.blue('http://localhost:15672')}`);
  
  // Development commands
  console.log(chalk.cyan('\n💡 Management Commands:'));
  console.log(`  Status:      ${chalk.gray('npm run status')}`);
  console.log(`  Health:      ${chalk.gray('npm run health')}`);
  console.log(`  Logs:        ${chalk.gray('npm run logs')}`);
  console.log(`  Stop:        ${chalk.gray('npm run stop')}`);
  console.log(`  GPU Monitor: ${chalk.gray('npm run gpu:monitor')}`);
  
  log.success('🚀 Full Stack Legal AI Development Environment Ready!');
  
  // Keep monitoring
  if (CONFIG.monitoring.enabled && !process.argv.includes('--no-monitor')) {
    await monitorSystemContinuously();
  }
}

async function monitorSystemContinuously() {
  log.info('🔍 Starting continuous system monitoring (Ctrl+C to stop)...');
  
  const monitorInterval = setInterval(async () => {
    await collectMetrics();
    
    // Check for anomalies
    if (state.metrics.gpu_utilization > 95) {
      log.warn('GPU utilization critical: consider scaling');
    }
    
    if (state.workerManager) {
      const stats = state.workerManager.getStats();
      if (stats.queued_gpu > 10 || stats.queued_cpu > 20) {
        log.warn('High task queue detected: consider adding workers');
      }
    }
  }, 30000); // Check every 30 seconds
  
  // Performance summary every 5 minutes
  const summaryInterval = setInterval(() => {
    const uptime = Date.now() - state.startTime;
    const hours = Math.floor(uptime / 3600000);
    const minutes = Math.floor((uptime % 3600000) / 60000);
    
    log.success(`📊 Performance Summary - ${hours}h${minutes}m uptime, ${state.metrics.requests} requests processed, ${state.metrics.errors} errors, ${(state.metrics.cache_hits / (state.metrics.cache_hits + state.metrics.cache_misses) * 100 || 0).toFixed(1)}% cache hit rate`);
  }, 300000); // Every 5 minutes
  
  // Graceful shutdown
  process.on('SIGINT', async () => {
    clearInterval(monitorInterval);
    clearInterval(summaryInterval);
    log.info('\n🛑 Monitoring stopped');
    
    if (process.argv.includes('--stop-on-exit')) {
      const stopScript = path.join(path.dirname(fileURLToPath(import.meta.url)), 'stop.mjs');
      if (fs.existsSync(stopScript)) {
        await $`zx ${stopScript}`;
      }
    }
    
    process.exit(0);
  });
  
  // Keep the process alive
  await new Promise(() => {});
}

// Utility functions
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Handle CLI arguments
if (process.argv.includes('--help')) {
  console.log(`
YoRHa Legal AI Full Stack Development Environment

Usage: npm run dev:full [options]

Options:
  --include-optional    Include optional services (Qdrant, Rust Tagger)
  --no-monitor         Don't monitor services after startup
  --stop-on-exit       Stop all services when script exits
  --disable-clustering Disable Node.js clustering
  --gpu-only           Only start GPU-accelerated services
  --cpu-only           Only start CPU services
  --debug              Enable debug logging
  --help               Show this help message

Examples:
  npm run dev:full                    # Start all core services with clustering
  npm run dev:full --include-optional # Start all services including optional ones
  npm run dev:full --gpu-only         # Start only GPU services
  npm run dev:full --debug            # Start with debug logging
`);
  process.exit(0);
}

// Apply CLI options
if (process.argv.includes('--disable-clustering')) {
  CONFIG.clustering.enabled = false;
}

if (process.argv.includes('--debug')) {
  process.env.DEBUG = 'true';
}

if (process.argv.includes('--gpu-only')) {
  const gpuServices = ['ollama', 'goGPU', 'goSIMD'];
  for (const [name, config] of Object.entries(CONFIG.services)) {
    if (!gpuServices.includes(name)) {
      config.optional = true;
    }
  }
}

// Run the orchestrator
main().catch(error => {
  log.error(`Orchestrator failed: ${error.message}`, { stack: error.stack });
  process.exit(1);
});