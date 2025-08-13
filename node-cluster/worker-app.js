/**
 * Worker Application Entry Point
 * Integrates SvelteKit with cluster communication and service coordination
 */

const { createServer } = require('http');
const { parse } = require('url');
const next = require('next'); // Will be replaced with SvelteKit adapter
const grpc = require('@grpc/grpc-js');
const nats = require('nats');

class WorkerApplication {
  constructor() {
    this.workerId = process.env.WORKER_ID;
    this.pid = process.pid;
    this.startTime = Date.now();
    
    // Service connections
    this.kratosClient = null;
    this.natsConnection = null;
    this.server = null;
    
    // Performance tracking
    this.requestCount = 0;
    this.lastHealthReport = Date.now();
    
    this.init();
  }
  
  async init() {
    console.log(`[WORKER-${this.pid}] Initializing worker application: ${this.workerId}`);
    
    // Setup service connections
    await this.connectToServices();
    
    // Setup SvelteKit application
    await this.setupSvelteKitApp();
    
    // Setup request handling
    this.setupRequestHandling();
    
    // Setup cluster communication
    this.setupClusterCommunication();
    
    // Start the server
    await this.startServer();
  }
  
  async connectToServices() {
    try {
      // Connect to NATS for message coordination
      this.natsConnection = await nats.connect({
        servers: [process.env.NATS_SERVER_URL || 'nats://localhost:4222'],
        reconnectTimeWait: 2000,
        maxReconnectAttempts: 10
      });
      
      console.log(`[WORKER-${this.pid}] Connected to NATS server`);
      
      // Setup NATS subscriptions for worker coordination
      await this.setupNATSSubscriptions();
      
    } catch (error) {
      console.error(`[WORKER-${this.pid}] Failed to connect to services:`, error);
    }
  }
  
  async setupNATSSubscriptions() {
    const js = this.natsConnection.jetstream();
    
    // Subscribe to legal AI coordination messages
    const subscription = await js.subscribe('legal.worker.*', {
      durable_name: `worker-${this.workerId}`,
      deliver_policy: 'new',
      ack_policy: 'explicit'
    });
    
    // Handle coordination messages
    (async () => {
      for await (const message of subscription) {
        await this.handleCoordinationMessage(message);
        message.ack();
      }
    })();
    
    console.log(`[WORKER-${this.pid}] NATS subscriptions setup complete`);
  }
  
  async handleCoordinationMessage(message) {
    const subject = message.subject;
    const data = JSON.parse(new TextDecoder().decode(message.data));
    
    console.log(`[WORKER-${this.pid}] Received coordination message: ${subject}`, data);
    
    switch (subject) {
      case 'legal.worker.memory-cleanup':
        await this.performMemoryCleanup();
        break;
        
      case 'legal.worker.health-check':
        await this.reportHealth();
        break;
        
      case 'legal.worker.document-process':
        await this.processDocument(data);
        break;
        
      case 'legal.worker.scale-signal':
        await this.handleScaleSignal(data);
        break;
        
      default:
        console.log(`[WORKER-${this.pid}] Unknown coordination message: ${subject}`);
    }
  }
  
  async setupSvelteKitApp() {
    // Load SvelteKit build output
    const { handler } = require('../build/handler.js');
    
    this.svelteKitHandler = handler;
    
    console.log(`[WORKER-${this.pid}] SvelteKit application loaded`);
  }
  
  setupRequestHandling() {
    this.server = createServer(async (req, res) => {
      try {
        this.requestCount++;
        const startTime = Date.now();
        
        // Add worker identification headers
        res.setHeader('X-Worker-ID', this.workerId);
        res.setHeader('X-Worker-PID', this.pid);
        res.setHeader('X-Request-Count', this.requestCount);
        
        // Handle special worker endpoints
        const url = parse(req.url, true);
        
        if (url.pathname === '/_worker/health') {
          return this.handleHealthCheck(req, res);
        }
        
        if (url.pathname === '/_worker/metrics') {
          return this.handleMetrics(req, res);
        }
        
        if (url.pathname.startsWith('/_worker/')) {
          return this.handleWorkerAPI(req, res, url);
        }
        
        // Route to SvelteKit for all other requests
        await this.svelteKitHandler(req, res);
        
        // Log request metrics
        const duration = Date.now() - startTime;
        this.logRequestMetrics(req, duration);
        
      } catch (error) {
        console.error(`[WORKER-${this.pid}] Request error:`, error);
        res.statusCode = 500;
        res.end('Internal Server Error');
      }
    });
  }
  
  handleHealthCheck(req, res) {
    const memUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    const health = {
      status: 'healthy',
      workerId: this.workerId,
      pid: this.pid,
      uptime: process.uptime(),
      requests: this.requestCount,
      memory: {
        rss: memUsage.rss,
        heapUsed: memUsage.heapUsed,
        heapTotal: memUsage.heapTotal,
        external: memUsage.external
      },
      cpu: {
        user: cpuUsage.user,
        system: cpuUsage.system
      },
      services: {
        nats: this.natsConnection ? 'connected' : 'disconnected',
        kratos: this.kratosClient ? 'connected' : 'disconnected'
      },
      timestamp: Date.now()
    };
    
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(health, null, 2));
  }
  
  handleMetrics(req, res) {
    const metrics = {
      workerId: this.workerId,
      pid: this.pid,
      startTime: this.startTime,
      uptime: Date.now() - this.startTime,
      requestCount: this.requestCount,
      lastHealthReport: this.lastHealthReport,
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
      eventLoopDelay: this.getEventLoopDelay()
    };
    
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(metrics, null, 2));
  }
  
  async handleWorkerAPI(req, res, url) {
    const action = url.pathname.replace('/_worker/', '');
    
    switch (action) {
      case 'gc':
        global.gc && global.gc();
        res.end('Garbage collection triggered');
        break;
        
      case 'reload':
        // Graceful worker reload
        process.send({ type: 'worker-reload', pid: this.pid });
        res.end('Worker reload initiated');
        break;
        
      case 'status':
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({
          workerId: this.workerId,
          status: 'active',
          uptime: process.uptime(),
          requests: this.requestCount
        }));
        break;
        
      default:
        res.statusCode = 404;
        res.end('Worker API endpoint not found');
    }
  }
  
  setupClusterCommunication() {
    // Handle messages from cluster master
    process.on('message', async (message) => {
      switch (message.type) {
        case 'memory-cleanup':
          await this.performMemoryCleanup();
          break;
          
        case 'health-request':
          await this.reportHealth();
          break;
          
        case 'graceful-shutdown':
          await this.gracefulShutdown();
          break;
          
        case 'document-process':
          await this.processDocument(message.data);
          break;
          
        default:
          console.log(`[WORKER-${this.pid}] Unknown cluster message: ${message.type}`);
      }
    });
    
    // Send periodic health reports
    setInterval(() => {
      this.reportHealth();
    }, 30000);
  }
  
  async performMemoryCleanup() {
    console.log(`[WORKER-${this.pid}] Performing memory cleanup`);
    
    try {
      // Trigger garbage collection
      if (global.gc) {
        global.gc();
      }
      
      // Clear internal caches
      if (this.svelteKitHandler && this.svelteKitHandler.clearCache) {
        this.svelteKitHandler.clearCache();
      }
      
      // Report memory usage after cleanup
      const memUsage = process.memoryUsage();
      console.log(`[WORKER-${this.pid}] Memory after cleanup:`, {
        rss: Math.round(memUsage.rss / 1024 / 1024) + 'MB',
        heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024) + 'MB'
      });
      
    } catch (error) {
      console.error(`[WORKER-${this.pid}] Memory cleanup error:`, error);
    }
  }
  
  async reportHealth() {
    const healthData = {
      workerId: this.workerId,
      pid: this.pid,
      uptime: process.uptime(),
      requestCount: this.requestCount,
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
      timestamp: Date.now()
    };
    
    this.lastHealthReport = Date.now();
    
    // Send to cluster master
    process.send({
      type: 'health-report',
      data: healthData
    });
    
    // Publish to NATS for broader monitoring
    if (this.natsConnection) {
      try {
        this.natsConnection.publish('legal.worker.health', JSON.stringify(healthData));
      } catch (error) {
        console.error(`[WORKER-${this.pid}] Failed to publish health to NATS:`, error);
      }
    }
  }
  
  async processDocument(documentData) {
    console.log(`[WORKER-${this.pid}] Processing document:`, documentData.documentId);
    
    try {
      // Coordinate with Go-Kratos service for document processing
      if (this.kratosClient) {
        const result = await this.callKratosService('ProcessDocument', documentData);
        
        // Publish processing result
        if (this.natsConnection) {
          this.natsConnection.publish('legal.document.processed', JSON.stringify({
            workerId: this.workerId,
            documentId: documentData.documentId,
            result: result,
            timestamp: Date.now()
          }));
        }
      }
      
    } catch (error) {
      console.error(`[WORKER-${this.pid}] Document processing error:`, error);
    }
  }
  
  async callKratosService(method, data) {
    return new Promise((resolve, reject) => {
      if (!this.kratosClient) {
        return reject(new Error('Kratos client not connected'));
      }
      
      this.kratosClient[method](data, (error, response) => {
        if (error) {
          reject(error);
        } else {
          resolve(response);
        }
      });
    });
  }
  
  logRequestMetrics(req, duration) {
    // Log request for performance monitoring
    const logData = {
      workerId: this.workerId,
      method: req.method,
      url: req.url,
      duration: duration,
      timestamp: Date.now(),
      userAgent: req.headers['user-agent']
    };
    
    // Send to logging system via NATS
    if (this.natsConnection) {
      this.natsConnection.publish('legal.logs.request', JSON.stringify(logData));
    }
  }
  
  getEventLoopDelay() {
    const start = process.hrtime.bigint();
    setImmediate(() => {
      const delta = process.hrtime.bigint() - start;
      return Number(delta / 1000000n); // Convert to milliseconds
    });
    return 0; // Simplified for this example
  }
  
  async startServer() {
    const port = process.env.PORT || 5173;
    
    this.server.listen(port, () => {
      console.log(`[WORKER-${this.pid}] Server listening on port ${port}`);
      
      // Notify cluster master that worker is ready
      process.send({
        type: 'worker-ready',
        pid: this.pid,
        workerId: this.workerId,
        port: port
      });
    });
    
    // Handle server errors
    this.server.on('error', (error) => {
      console.error(`[WORKER-${this.pid}] Server error:`, error);
      process.exit(1);
    });
  }
  
  async gracefulShutdown() {
    console.log(`[WORKER-${this.pid}] Starting graceful shutdown`);
    
    // Stop accepting new connections
    this.server.close(() => {
      console.log(`[WORKER-${this.pid}] Server closed`);
    });
    
    // Close service connections
    if (this.natsConnection) {
      await this.natsConnection.close();
    }
    
    // Exit process
    setTimeout(() => {
      process.exit(0);
    }, 5000);
  }
}

// Start worker application
const workerApp = new WorkerApplication();

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error(`[WORKER-${process.pid}] Uncaught exception:`, error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error(`[WORKER-${process.pid}] Unhandled rejection:`, reason);
  process.exit(1);
});

module.exports = WorkerApplication;