// Production Service Manager - Windows Native Optimization
// Handles event loops, caching, interrupts, and heuristic pattern matching

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const EventEmitter = require('events');

class ProductionServiceManager extends EventEmitter {
  constructor() {
    super();
    this.services = new Map();
    this.healthChecks = new Map();
    this.eventQueue = new Map();
    this.cache = new Map();
    this.patterns = new Map();

    // Performance optimization settings
    this.config = {
      environment: 'production',
      optimization: {
        eventLoop: {
          highPriority: ['vector-search', 'chat-response', 'error-fix'],
          batchSize: 100,
          processingInterval: 16 // ~60fps
        },
        caching: {
          l1Size: 1024 * 1024, // 1MB memory cache
          l2Size: 100 * 1024 * 1024, // 100MB SSD cache
          ttl: 300000 // 5 minutes
        },
        patterns: {
          precompileRegex: true,
          heuristicScoring: true,
          jsonbOptimization: true
        }
      }
    };

    this.initializeOptimizations();
    this.initializeServices();
  }

  initializeOptimizations() {
    console.log('⚡ Initializing production optimizations...');

    // Event loop optimization
    this.setupEventLoop();

    // Cache system
    this.setupIntelligentCaching();

    // Pattern recognition
    this.setupPatternMatching();

    // Error interrupts
    this.setupErrorHandling();

    console.log('✅ Production optimizations active');
  }

  setupEventLoop() {
    // High-frequency event processing
    setImmediate(() => this.processEventQueue());

    // Timer-based maintenance
    setInterval(() => this.performMaintenance(), 1000);

    // Monitor event loop lag
    setInterval(() => {
      const start = process.hrtime.bigint();
      setImmediate(() => {
        const lag = Number(process.hrtime.bigint() - start) / 1000000;
        if (lag > 10) {
          console.warn(`⚠️ Event loop lag detected: ${lag.toFixed(2)}ms`);
          this.optimizeEventLoop();
        }
      });
    }, 5000);
  }

  setupIntelligentCaching() {
    // L1 Cache: Memory (immediate access)
    this.l1Cache = new Map();

    // L2 Cache: Disk-based for larger data
    this.l2CachePath = path.join(__dirname, '../cache/l2');
    if (!fs.existsSync(this.l2CachePath)) {
      fs.mkdirSync(this.l2CachePath, { recursive: true });
    }

    // Cache cleanup timer
    setInterval(() => this.cleanupCache(), 60000);
  }

  setupPatternMatching() {
    // Precompile legal document patterns for performance
    this.patterns.set('case_citation', {
      regex: /\b\d+\s+[A-Z][a-z]+\.?\s+\d+\b/g,
      confidence: 0.9,
      priority: 'high'
    });

    this.patterns.set('statute', {
      regex: /\b\d+\s+U\.S\.C\.?\s+§?\s*\d+/g,
      confidence: 0.95,
      priority: 'high'
    });

    this.patterns.set('monetary', {
      regex: /\$[\d,]+(?:\.\d{2})?/g,
      confidence: 0.85,
      priority: 'medium'
    });

    this.patterns.set('date', {
      regex: /\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b/g,
      confidence: 0.8,
      priority: 'low'
    });

    console.log(`✅ ${this.patterns.size} legal patterns precompiled`);
  }

  setupErrorHandling() {
    // Interrupt-style error handling
    process.on('uncaughtException', (error) => {
      console.error('🚨 Uncaught Exception:', error);
      this.handleCriticalError(error);
    });

    process.on('unhandledRejection', (reason, promise) => {
      console.error('🚨 Unhandled Rejection at:', promise, 'reason:', reason);
      this.handleCriticalError(reason);
    });

    // Memory monitoring
    setInterval(() => {
      const usage = process.memoryUsage();
      if (usage.heapUsed > 500 * 1024 * 1024) { // >500MB
        console.warn('⚠️ High memory usage detected, triggering cleanup...');
        if (global.gc) global.gc();
        this.cleanupCache();
      }
    }, 10000);
  }

  async initializeServices() {
    console.log('🚀 Initializing Legal AI Production Services...');

    try {
      // Start services in dependency order with optimization
      await this.startOptimizedService('postgresql', {
        port: 5432,
        healthPath: '/health',
        priority: 'critical',
        restartDelay: 5000
      });

      await this.startOptimizedService('ollama', {
        port: 11434,
        healthPath: '/api/version',
        priority: 'critical',
        restartDelay: 10000,
        env: { OLLAMA_GPU_LAYERS: '999' }
      });

      await this.startOptimizedService('context7', {
        port: 4000,
        healthPath: '/health',
        priority: 'high',
        restartDelay: 5000,
        cmd: 'node',
        args: ['mcp-servers/context7-server.js']
      });

      await this.startOptimizedService('context7-multicore', {
        port: 4100,
        healthPath: '/health',
        priority: 'high',
        restartDelay: 5000,
        cmd: 'node',
        args: ['mcp-servers/context7-multicore.js'],
        env: { MCP_PORT: '4100', MCP_MULTICORE: 'true' }
      });

      await this.startOptimizedService('enhanced-rag', {
        port: 8094,
        healthPath: '/health',
        priority: 'high',
        restartDelay: 3000,
        cmd: 'go',
        args: ['run', 'go-microservice/cmd/enhanced-rag-v2-local/main.go'],
        env: { RAG_HTTP_PORT: '8094', EMBED_MODEL: 'nomic-embed-text' }
      });

      // Start monitoring and optimization
      this.startHealthMonitoring();
      this.startPerformanceOptimization();
      this.startAutoSolveIntegration();

      console.log('✅ All Legal AI services started successfully');
      this.emit('production-ready');

    } catch (error) {
      console.error('❌ Failed to start services:', error);
      this.handleCriticalError(error);
    }
  }

  async startOptimizedService(name, config) {
    return new Promise((resolve, reject) => {
      console.log(`🔄 Starting ${name}...`);

      if (config.cmd) {
        const process = spawn(config.cmd, config.args || [], {
          cwd: __dirname + '/..',
          stdio: ['pipe', 'pipe', 'pipe'],
          env: { ...process.env, ...config.env }
        });

        // Optimized logging
        process.stdout.on('data', (data) => {
          this.logOptimized(`[${name}]`, data.toString());
        });

        process.stderr.on('data', (data) => {
          this.logOptimized(`[${name}] ERROR`, data.toString());
        });

        // Store process reference
        this.services.set(name, {
          process,
          config,
          status: 'starting',
          startTime: Date.now()
        });
      }

      // Health check with retry logic
      const checkHealth = async (attempts = 0) => {
        try {
          if (attempts > 10) {
            throw new Error(`Health check failed after 10 attempts`);
          }

          const healthy = await this.checkServiceHealthOptimized(name, config.port);
          if (healthy) {
            this.services.get(name).status = 'running';
            console.log(`✅ ${name} healthy and operational`);
            resolve();
          } else {
            setTimeout(() => checkHealth(attempts + 1), 2000);
          }
        } catch (error) {
          reject(new Error(`${name} failed to start: ${error.message}`));
        }
      };

      setTimeout(() => checkHealth(), config.restartDelay || 5000);
    });
  }

  async checkServiceHealthOptimized(name, port) {
    // Use cache for recent health checks
    const cacheKey = `health_${name}_${port}`;
    const cached = this.getCachedResult(cacheKey);
    if (cached) return cached;

    try {
      const response = await fetch(`http://localhost:${port}/health`, {
        timeout: 3000,
        headers: { 'User-Agent': 'LegalAI-HealthCheck/1.0' }
      });

      const healthy = response.ok;
      this.setCachedResult(cacheKey, healthy, 5000); // Cache for 5 seconds
      return healthy;

    } catch (error) {
      // Try alternative health check methods
      if (name === 'postgresql') {
        return await this.checkPostgreSQLHealth(port);
      }
      return false;
    }
  }

  startHealthMonitoring() {
    console.log('📊 Starting intelligent health monitoring...');

    setInterval(async () => {
      for (const [name, service] of this.services) {
        if (service.status === 'running') {
          const healthy = await this.checkServiceHealthOptimized(name, service.config.port);

          if (!healthy) {
            console.log(`⚠️ Service ${name} unhealthy, attempting restart...`);
            await this.restartServiceOptimized(name);
          }
        }
      }
    }, 30000); // Check every 30 seconds
  }

  startPerformanceOptimization() {
    console.log('⚡ Starting performance optimization...');

    // CPU optimization
    setInterval(() => {
      const usage = process.cpuUsage();
      if (usage.user > 1000000) { // High CPU usage
        this.optimizeCPUUsage();
      }
    }, 5000);

    // Auto-scaling based on load
    setInterval(() => {
      this.autoScale();
    }, 60000);
  }

  startAutoSolveIntegration() {
    console.log('🔧 Starting AutoSolve integration...');

    // Monitor for AutoSolve requests
    this.on('autosolve-request', async (request) => {
      await this.processAutoSolveRequest(request);
    });

    // Periodic error analysis
    setInterval(() => {
      this.runErrorAnalysis();
    }, 300000); // Every 5 minutes
  }

  // Optimized caching methods
  getCachedResult(key) {
    const cached = this.l1Cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.config.optimization.caching.ttl) {
      return cached.value;
    }
    return null;
  }

  setCachedResult(key, value, ttl = null) {
    this.l1Cache.set(key, {
      value,
      timestamp: Date.now(),
      ttl: ttl || this.config.optimization.caching.ttl
    });

    // Manage cache size
    if (this.l1Cache.size > this.config.optimization.caching.l1Size / 1024) {
      this.evictOldestCache();
    }
  }

  evictOldestCache() {
    const oldest = [...this.l1Cache.entries()]
      .sort(([,a], [,b]) => a.timestamp - b.timestamp)[0];
    if (oldest) {
      this.l1Cache.delete(oldest[0]);
    }
  }

  processEventQueue() {
    // Process high-priority events first
    const priorities = ['high', 'medium', 'low'];

    for (const priority of priorities) {
      const events = this.eventQueue.get(priority) || [];
      if (events.length > 0) {
        const batch = events.splice(0, this.config.optimization.eventLoop.batchSize);
        batch.forEach(event => this.processEvent(event));
      }
    }

    // Schedule next processing cycle
    setImmediate(() => this.processEventQueue());
  }

  async processAutoSolveRequest(request) {
    console.log('🔧 Processing AutoSolve request:', request.type);

    try {
      switch (request.type) {
        case 'typescript-errors':
          return await this.fixTypeScriptErrors(request.data);
        case 'performance-optimization':
          return await this.optimizePerformance(request.data);
        case 'service-health':
          return await this.diagnoseServiceHealth(request.data);
        default:
          console.warn('Unknown AutoSolve request type:', request.type);
      }
    } catch (error) {
      console.error('AutoSolve processing error:', error);
    }
  }

  logOptimized(prefix, message) {
    // Batch logging for performance
    const timestamp = new Date().toISOString();
    const logEntry = `${timestamp} ${prefix} ${message}`;

    // Console output (immediate)
    console.log(logEntry);

    // File logging (batched)
    this.logBuffer = this.logBuffer || [];
    this.logBuffer.push(logEntry);

    if (this.logBuffer.length > 100) {
      this.flushLogBuffer();
    }
  }

  flushLogBuffer() {
    if (this.logBuffer && this.logBuffer.length > 0) {
      const logFile = path.join(__dirname, '../logs/production.log');
      fs.appendFileSync(logFile, this.logBuffer.join('\n') + '\n');
      this.logBuffer = [];
    }
  }

  // Graceful shutdown
  async shutdown() {
    console.log('🔄 Gracefully shutting down Legal AI System...');

    // Flush any pending logs
    this.flushLogBuffer();

    // Stop all services
    for (const [name, service] of this.services) {
      if (service.process) {
        console.log(`Stopping ${name}...`);
        service.process.kill('SIGTERM');
      }
    }

    console.log('✅ Legal AI System shutdown complete');
  }
}

// Start the production service manager
const manager = new ProductionServiceManager();

// Handle Windows service signals
process.on('SIGTERM', () => manager.shutdown());
process.on('SIGINT', () => manager.shutdown());

// Export for VS Code extension integration
module.exports = ProductionServiceManager;
