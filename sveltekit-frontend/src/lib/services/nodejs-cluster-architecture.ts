/**
 * Node.js Cluster Architecture for SvelteKit 2
 * Provides horizontal scaling with intelligent load balancing and health monitoring
 */

import cluster from 'node:cluster';
import { cpus } from 'node:os';
import { EventEmitter } from 'node:events';
import { createServer } from 'node:http';
import { handler } from '../../../build/handler.js';
import { writable, type Writable } from 'svelte/store';

// Cluster configuration interfaces
export interface ClusterConfig {
  workers: number;
  port: number;
  host: string;
  gracefulShutdownTimeout: number;
  healthCheckInterval: number;
  maxMemoryUsage: number; // MB 
  restartOnHighMemory: boolean;
  loadBalancingStrategy: 'round-robin' | 'least-connections' | 'cpu-based';
  enableStickySession: boolean;
  redisUrl?: string; // For session storage
}

export interface WorkerMetrics {
  workerId: number;
  pid: number;
  status: 'online' | 'disconnected' | 'dead' | 'starting';
  connections: number;
  requestsHandled: number;
  memoryUsage: NodeJS.MemoryUsage;
  cpuUsage: NodeJS.CpuUsage;
  lastHealthCheck: number;
  errors: number;
  uptime: number;
}

export interface ClusterHealth {
  totalWorkers: number;
  healthyWorkers: number;
  totalRequests: number;
  averageResponseTime: number;
  memoryUsage: {
    total: number;
    average: number;
    peak: number;
  };
  cpuUsage: {
    total: number;
    average: number;
  };
  errors: {
    total: number;
    rate: number; // errors per minute
  };
}

/**
 * SvelteKit 2 Cluster Manager
 * Manages worker processes with intelligent load balancing
 */
export class SvelteKitClusterManager extends EventEmitter {
  private config: ClusterConfig;
  private workers = new Map<number, WorkerMetrics>();
  private requestCounter = 0;
  private startTime = Date.now();
  
  // Reactive stores for monitoring
  public health: Writable<ClusterHealth>;
  public workerMetrics: Writable<WorkerMetrics[]>;
  
  // Load balancing state
  private roundRobinIndex = 0;
  private connectionCounts = new Map<number, number>();

  constructor(config: Partial<ClusterConfig> = {}) {
    super();
    
    this.config = {
      workers: config.workers || cpus().length,
      port: config.port || 3000,
      host: config.host || '0.0.0.0',
      gracefulShutdownTimeout: config.gracefulShutdownTimeout || 10000,
      healthCheckInterval: config.healthCheckInterval || 5000,
      maxMemoryUsage: config.maxMemoryUsage || 512,
      restartOnHighMemory: config.restartOnHighMemory || true,
      loadBalancingStrategy: config.loadBalancingStrategy || 'round-robin',
      enableStickySession: config.enableStickySession || false,
      redisUrl: config.redisUrl
    };

    // Initialize reactive stores
    this.health = writable(this.getInitialHealth());
    this.workerMetrics = writable([]);
    
    this.setupCluster();
  }

  /**
   * Initialize cluster based on primary/worker role
   */
  private setupCluster(): void {
    if (cluster.isPrimary) {
      this.setupPrimary();
    } else {
      this.setupWorker();
    }
  }

  /**
   * Primary process setup - manages workers and load balancing
   */
  private setupPrimary(): void {
    console.log(`🚀 SvelteKit Cluster Manager starting with ${this.config.workers} workers`);
    console.log(`📊 Load balancing strategy: ${this.config.loadBalancingStrategy}`);
    
    // Create worker processes
    for (let i = 0; i < this.config.workers; i++) {
      this.createWorker();
    }

    // Setup cluster event handlers
    cluster.on('online', (worker) => {
      console.log(`✅ Worker ${worker.process.pid} is online`);
      this.updateWorkerStatus(worker.id, 'online');
    });

    cluster.on('disconnect', (worker) => {
      console.log(`⚠️ Worker ${worker.process.pid} disconnected`);
      this.updateWorkerStatus(worker.id, 'disconnected');
    });

    cluster.on('exit', (worker, code, signal) => {
      console.log(`❌ Worker ${worker.process.pid} died (${signal || code})`);
      this.handleWorkerExit(worker);
    });

    // Setup health monitoring
    this.startHealthMonitoring();
    
    // Setup graceful shutdown
    this.setupGracefulShutdown();

    // Create load balancer server
    this.createLoadBalancer();
  }

  /**
   * Worker process setup - runs SvelteKit application
   */
  private setupWorker(): void {
    const server = createServer(handler);
    
    server.listen(this.config.port, this.config.host, () => {
      console.log(`🔧 Worker ${process.pid} listening on ${this.config.host}:${this.config.port}`);
    });

    // Worker health reporting
    this.setupWorkerHealthReporting();
    
    // Handle worker-specific shutdown
    process.on('SIGTERM', () => {
      console.log(`🛑 Worker ${process.pid} received SIGTERM, shutting down gracefully`);
      server.close(() => {
        process.exit(0);
      });
    });
  }

  /**
   * Create a new worker process
   */
  private createWorker(): void {
    const worker = cluster.fork();
    
    this.workers.set(worker.id, {
      workerId: worker.id,
      pid: worker.process.pid!,
      status: 'starting',
      connections: 0,
      requestsHandled: 0,
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
      lastHealthCheck: Date.now(),
      errors: 0,
      uptime: 0
    });

    this.connectionCounts.set(worker.id, 0);
  }

  /**
   * Handle worker process exit and restart
   */
  private handleWorkerExit(worker: cluster.Worker): void {
    this.workers.delete(worker.id);
    this.connectionCounts.delete(worker.id);
    
    // Restart worker unless shutting down
    if (!worker.exitedAfterDisconnect) {
      console.log('🔄 Restarting worker...');
      setTimeout(() => this.createWorker(), 1000);
    }
  }

  /**
   * Create load balancer server in primary process
   */
  private createLoadBalancer(): void {
    const server = createServer((req, res) => {
      const workerId = this.selectWorker(req);
      const worker = cluster.workers![workerId];
      
      if (worker && worker.isDead() === false) {
        this.forwardRequest(worker, req, res);
      } else {
        // Fallback to any available worker
        const availableWorker = this.getAvailableWorker();
        if (availableWorker) {
          this.forwardRequest(availableWorker, req, res);
        } else {
          res.writeHead(503, { 'Content-Type': 'text/plain' });
          res.end('Service Temporarily Unavailable');
        }
      }
    });

    server.listen(this.config.port, this.config.host, () => {
      console.log(`⚖️ Load balancer listening on ${this.config.host}:${this.config.port}`);
    });
  }

  /**
   * Select worker based on load balancing strategy
   */
  private selectWorker(req: any): number {
    const availableWorkers = Array.from(this.workers.keys()).filter(
      id => cluster.workers![id] && !cluster.workers![id]!.isDead()
    );

    if (availableWorkers.length === 0) {
      throw new Error('No available workers');
    }

    switch (this.config.loadBalancingStrategy) {
      case 'round-robin':
        return this.selectRoundRobin(availableWorkers);
      
      case 'least-connections':
        return this.selectLeastConnections(availableWorkers);
      
      case 'cpu-based':
        return this.selectByCpuUsage(availableWorkers);
      
      default:
        return availableWorkers[0];
    }
  }

  /**
   * Round-robin worker selection
   */
  private selectRoundRobin(workers: number[]): number {
    const worker = workers[this.roundRobinIndex % workers.length];
    this.roundRobinIndex++;
    return worker;
  }

  /**
   * Select worker with least connections
   */
  private selectLeastConnections(workers: number[]): number {
    return workers.reduce((minWorker, workerId) => {
      const minConnections = this.connectionCounts.get(minWorker) || 0;
      const currentConnections = this.connectionCounts.get(workerId) || 0;
      return currentConnections < minConnections ? workerId : minWorker;
    });
  }

  /**
   * Select worker based on CPU usage
   */
  private selectByCpuUsage(workers: number[]): number {
    return workers.reduce((minWorker, workerId) => {
      const minMetrics = this.workers.get(minWorker);
      const currentMetrics = this.workers.get(workerId);
      
      if (!minMetrics || !currentMetrics) return minWorker;
      
      const minCpu = minMetrics.cpuUsage.user + minMetrics.cpuUsage.system;
      const currentCpu = currentMetrics.cpuUsage.user + currentMetrics.cpuUsage.system;
      
      return currentCpu < minCpu ? workerId : minWorker;
    });
  }

  /**
   * Forward request to selected worker
   */
  private forwardRequest(worker: cluster.Worker, req: any, res: any): void {
    // Track connection count
    const connections = this.connectionCounts.get(worker.id) || 0;
    this.connectionCounts.set(worker.id, connections + 1);
    
    // Update request counter
    this.requestCounter++;
    
    // Forward request to worker (simplified - in practice use http-proxy)
    worker.send({
      type: 'http-request',
      req: {
        url: req.url,
        method: req.method,
        headers: req.headers
      }
    });

    // Handle response (simplified)
    res.on('close', () => {
      const newConnections = this.connectionCounts.get(worker.id) || 1;
      this.connectionCounts.set(worker.id, Math.max(0, newConnections - 1));
    });
  }

  /**
   * Get any available worker
   */
  private getAvailableWorker(): cluster.Worker | null {
    for (const worker of Object.values(cluster.workers!)) {
      if (worker && !worker.isDead()) {
        return worker;
      }
    }
    return null;
  }

  /**
   * Start health monitoring for all workers
   */
  private startHealthMonitoring(): void {
    setInterval(() => {
      this.collectWorkerMetrics();
      this.updateHealthStore();
      this.checkWorkerHealth();
    }, this.config.healthCheckInterval);
  }

  /**
   * Collect metrics from all workers
   */
  private collectWorkerMetrics(): void {
    for (const [workerId, worker] of Object.entries(cluster.workers!)) {
      if (worker && !worker.isDead()) {
        worker.send({ type: 'health-check' });
      }
    }
  }

  /**
   * Setup worker health reporting
   */
  private setupWorkerHealthReporting(): void {
    process.on('message', (msg: any) => {
      if (msg.type === 'health-check') {
        // Send health metrics back to primary
        process.send!({
          type: 'health-response',
          workerId: process.env.CLUSTER_WORKER_ID,
          metrics: {
            memoryUsage: process.memoryUsage(),
            cpuUsage: process.cpuUsage(),
            uptime: process.uptime(),
            timestamp: Date.now()
          }
        });
      }
    });
  }

  /**
   * Update worker status
   */
  private updateWorkerStatus(workerId: number, status: WorkerMetrics['status']): void {
    const worker = this.workers.get(workerId);
    if (worker) {
      worker.status = status;
      worker.lastHealthCheck = Date.now();
    }
  }

  /**
   * Check worker health and restart if necessary
   */
  private checkWorkerHealth(): void {
    for (const [workerId, metrics] of this.workers.entries()) {
      const worker = cluster.workers![workerId];
      
      if (!worker || worker.isDead()) {
        continue;
      }

      // Check memory usage
      if (this.config.restartOnHighMemory) {
        const memoryMB = metrics.memoryUsage.heapUsed / 1024 / 1024;
        if (memoryMB > this.config.maxMemoryUsage) {
          console.log(`🔄 Restarting worker ${workerId} due to high memory usage: ${memoryMB.toFixed(2)}MB`);
          this.restartWorker(workerId);
          continue;
        }
      }

      // Check last health check time
      const timeSinceLastCheck = Date.now() - metrics.lastHealthCheck;
      if (timeSinceLastCheck > this.config.healthCheckInterval * 3) {
        console.log(`🔄 Restarting unresponsive worker ${workerId}`);
        this.restartWorker(workerId);
      }
    }
  }

  /**
   * Restart a specific worker
   */
  private restartWorker(workerId: number): void {
    const worker = cluster.workers![workerId];
    if (worker) {
      worker.disconnect();
      setTimeout(() => {
        if (!worker.isDead()) {
          worker.kill();
        }
      }, this.config.gracefulShutdownTimeout);
    }
  }

  /**
   * Update health store with current metrics
   */
  private updateHealthStore(): void {
    const healthyWorkers = Array.from(this.workers.values()).filter(
      w => w.status === 'online'
    );

    const totalMemory = healthyWorkers.reduce((sum, w) => sum + w.memoryUsage.heapUsed, 0);
    const averageMemory = healthyWorkers.length > 0 ? totalMemory / healthyWorkers.length : 0;

    const health: ClusterHealth = {
      totalWorkers: this.workers.size,
      healthyWorkers: healthyWorkers.length,
      totalRequests: this.requestCounter,
      averageResponseTime: this.calculateAverageResponseTime(),
      memoryUsage: {
        total: totalMemory,
        average: averageMemory,
        peak: Math.max(...healthyWorkers.map(w => w.memoryUsage.heapUsed))
      },
      cpuUsage: {
        total: healthyWorkers.reduce((sum, w) => sum + w.cpuUsage.user + w.cpuUsage.system, 0),
        average: healthyWorkers.length > 0 ? 
          healthyWorkers.reduce((sum, w) => sum + w.cpuUsage.user + w.cpuUsage.system, 0) / healthyWorkers.length : 0
      },
      errors: {
        total: healthyWorkers.reduce((sum, w) => sum + w.errors, 0),
        rate: this.calculateErrorRate()
      }
    };

    this.health.set(health);
    this.workerMetrics.set(Array.from(this.workers.values()));
  }

  /**
   * Calculate average response time (simplified)
   */
  private calculateAverageResponseTime(): number {
    // In a real implementation, this would track actual response times
    return Math.random() * 100 + 50; // Mock: 50-150ms
  }

  /**
   * Calculate error rate per minute
   */
  private calculateErrorRate(): number {
    const uptime = (Date.now() - this.startTime) / 1000 / 60; // minutes
    const totalErrors = Array.from(this.workers.values()).reduce((sum, w) => sum + w.errors, 0);
    return uptime > 0 ? totalErrors / uptime : 0;
  }

  /**
   * Get initial health state
   */
  private getInitialHealth(): ClusterHealth {
    return {
      totalWorkers: 0,
      healthyWorkers: 0,
      totalRequests: 0,
      averageResponseTime: 0,
      memoryUsage: { total: 0, average: 0, peak: 0 },
      cpuUsage: { total: 0, average: 0 },
      errors: { total: 0, rate: 0 }
    };
  }

  /**
   * Setup graceful shutdown
   */
  private setupGracefulShutdown(): void {
    const shutdown = () => {
      console.log('🛑 Initiating graceful cluster shutdown...');
      
      for (const worker of Object.values(cluster.workers!)) {
        if (worker) {
          worker.disconnect();
        }
      }

      setTimeout(() => {
        console.log('💀 Force killing remaining workers');
        for (const worker of Object.values(cluster.workers!)) {
          if (worker && !worker.isDead()) {
            worker.kill();
          }
        }
        process.exit(0);
      }, this.config.gracefulShutdownTimeout);
    };

    process.on('SIGTERM', shutdown);
    process.on('SIGINT', shutdown);
  }

  /**
   * Get cluster health information
   */
  public getHealth(): ClusterHealth {
    return this.health;
  }

  /**
   * Get worker metrics
   */
  public getWorkerMetrics(): WorkerMetrics[] {
    return Array.from(this.workers.values());
  }

  /**
   * Manually restart all workers (rolling restart)
   */
  public async rollingRestart(): Promise<void> {
    console.log('🔄 Starting rolling restart of all workers...');
    
    const workerIds = Array.from(this.workers.keys());
    
    for (const workerId of workerIds) {
      console.log(`🔄 Restarting worker ${workerId}...`);
      this.restartWorker(workerId);
      
      // Wait for worker to be replaced
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    console.log('✅ Rolling restart completed');
  }

  /**
   * Scale cluster up or down
   */
  public scaleCluster(targetWorkers: number): void {
    const currentWorkers = this.workers.size;
    
    if (targetWorkers > currentWorkers) {
      // Scale up
      const workersToAdd = targetWorkers - currentWorkers;
      console.log(`📈 Scaling up: adding ${workersToAdd} workers`);
      
      for (let i = 0; i < workersToAdd; i++) {
        this.createWorker();
      }
    } else if (targetWorkers < currentWorkers) {
      // Scale down
      const workersToRemove = currentWorkers - targetWorkers;
      console.log(`📉 Scaling down: removing ${workersToRemove} workers`);
      
      const workerIds = Array.from(this.workers.keys()).slice(0, workersToRemove);
      
      for (const workerId of workerIds) {
        const worker = cluster.workers![workerId];
        if (worker) {
          worker.disconnect();
        }
      }
    }

    this.config.workers = targetWorkers;
  }
}

/**
 * Factory function to create and start cluster manager
 */
export function createSvelteKitCluster(config?: Partial<ClusterConfig>): SvelteKitClusterManager {
  return new SvelteKitClusterManager(config);
}

/**
 * Express/Connect middleware for cluster-aware session handling
 */
export function createClusterSessionMiddleware(redisUrl?: string) {
  return (req: any, res: any, next: any) => {
    // Add cluster-aware session handling
    // This would integrate with Redis for shared session storage
    req.clusterId = process.env.CLUSTER_WORKER_ID;
    next();
  };
}

/**
 * Health check endpoint for load balancers
 */
export function createHealthCheckEndpoint() {
  return (req: any, res: any) => {
    const health = {
      status: 'healthy',
      timestamp: Date.now(),
      worker: {
        pid: process.pid,
        memory: process.memoryUsage(),
        uptime: process.uptime()
      }
    };
    
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(health));
  };
}