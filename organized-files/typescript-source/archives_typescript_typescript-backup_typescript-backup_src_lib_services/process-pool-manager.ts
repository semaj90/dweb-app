/**
 * Advanced Process Pool Manager for Legal AI Platform
 * Caches and reuses Node.js processes, Go services, and Python/CUDA workers
 * Integrates with existing NATS messaging and Redis caching architecture
 */
import { Worker, isMainThread, MessageChannel } from 'worker_threads';
import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import Redis from 'ioredis';

export interface WorkerConfig {
  type: 'node' | 'go' | 'python-cuda';
  command: string;
  args?: string[];
  maxInstances: number;
  idleTimeout: number; // ms
  gpuMemoryMB?: number; // For CUDA workers
  port?: number; // For Go services
}

export interface WorkerInstance {
  id: string;
  type: string;
  process?: ChildProcess | Worker;
  port?: number;
  lastUsed: number;
  isIdle: boolean;
  memoryUsage?: number;
  gpuMemoryMB?: number;
  requestCount: number;
}

export interface PoolStats {
  totalWorkers: number;
  idleWorkers: number;
  busyWorkers: number;
  totalRequests: number;
  averageResponseTime: number;
  memoryUsage: {
    node: number;
    gpu: number;
  };
}

export class ProcessPoolManager extends EventEmitter {
  private pools: Map<string, WorkerInstance[]> = new Map();
  private configs: Map<string, WorkerConfig> = new Map();
  private redis: Redis;
  private stats: Map<string, number[]> = new Map(); // Response times
  private cleanupInterval: NodeJS.Timeout;

  constructor(private redisUrl: string = 'redis://localhost:6379') {
    super();
    this.redis = new Redis(redisUrl);
    
    // Cleanup idle workers every 30 seconds
    this.cleanupInterval = setInterval(() => this.cleanupIdleWorkers(), 30000);
  }

  /**
   * Register worker configuration for legal AI services
   */
  async registerWorker(poolName: string, config: WorkerConfig): Promise<any> {
    this.configs.set(poolName, config);
    this.pools.set(poolName, []);
    
    // Pre-warm pool with initial workers
    const initialCount = Math.min(2, config.maxInstances);
    for (let i = 0; i < initialCount; i++) {
      await this.createWorker(poolName, config);
    }

    this.emit('pool:registered', { poolName, config });
  }

  /**
   * Get or create worker from pool with intelligent load balancing
   */
  async getWorker(poolName: string, requestData?: any): Promise<WorkerInstance> {
    const config = this.configs.get(poolName);
    const pool = this.pools.get(poolName);
    
    if (!config || !pool) {
      throw new Error(`Pool ${poolName} not registered`);
    }

    // Find idle worker
    let worker = pool.find(w => w.isIdle);
    
    // Create new worker if none available and under limit
    if (!worker && pool.length < config.maxInstances) {
      worker = await this.createWorker(poolName, config);
      pool.push(worker);
    }
    
    // Wait for worker or use least busy one
    if (!worker) {
      worker = pool.reduce((prev, curr) => 
        prev.requestCount < curr.requestCount ? prev : curr
      );
    }

    // Mark as busy and update usage
    worker.isIdle = false;
    worker.lastUsed = Date.now();
    worker.requestCount++;

    // Cache worker assignment in Redis for debugging
    await this.redis.setex(
      `legal_ai:worker_assignment:${worker.id}`, 
      300, 
      JSON.stringify({ poolName, requestData: requestData?.type || 'unknown' })
    );

    this.emit('worker:assigned', { poolName, workerId: worker.id });
    return worker;
  }

  /**
   * Release worker back to pool
   */
  async releaseWorker(poolName: string, workerId: string, responseTime?: number): Promise<any> {
    const pool = this.pools.get(poolName);
    const worker = pool?.find(w => w.id === workerId);
    
    if (worker) {
      worker.isIdle = true;
      worker.lastUsed = Date.now();
      
      // Track response time for analytics
      if (responseTime) {
        const times = this.stats.get(poolName) || [];
        times.push(responseTime);
        if (times.length > 100) times.shift(); // Keep last 100
        this.stats.set(poolName, times);
      }
      
      this.emit('worker:released', { poolName, workerId, responseTime });
    }
  }

  /**
   * Create worker instance based on type
   */
  private async createWorker(poolName: string, config: WorkerConfig): Promise<WorkerInstance> {
    const workerId = `${poolName}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    let worker: WorkerInstance;

    switch (config.type) {
      case 'node':
        worker = await this.createNodeWorker(workerId, config);
        break;
        
      case 'go':
        worker = await this.createGoService(workerId, config);
        break;
        
      case 'python-cuda':
        worker = await this.createPythonCudaWorker(workerId, config);
        break;
        
      default:
        throw new Error(`Unknown worker type: ${config.type}`);
    }

    this.emit('worker:created', { poolName, workerId });
    return worker;
  }

  /**
   * Create Node.js worker thread for legal document processing
   */
  private async createNodeWorker(workerId: string, config: WorkerConfig): Promise<WorkerInstance> {
    const worker = new Worker(config.command, {
      workerData: { workerId, ...config.args }
    });
    
    return {
      id: workerId,
      type: 'node',
      process: worker,
      lastUsed: Date.now(),
      isIdle: true,
      requestCount: 0
    };
  }

  /**
   * Create Go service process for enhanced RAG and vector operations
   */
  private async createGoService(workerId: string, config: WorkerConfig): Promise<WorkerInstance> {
    // Find available port for Go service
    const port = config.port || await this.findAvailablePort(8090);
    
    const process = spawn(config.command, [
      ...config.args || [],
      '--port', port.toString(),
      '--worker-id', workerId
    ], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PORT: port.toString() }
    });

    // Wait for service to be ready
    await this.waitForService(`http://localhost:${port}/health`);
    
    return {
      id: workerId,
      type: 'go',
      process,
      port,
      lastUsed: Date.now(),
      isIdle: true,
      requestCount: 0
    };
  }

  /**
   * Create Python CUDA worker for GPU-accelerated legal analysis
   */
  private async createPythonCudaWorker(workerId: string, config: WorkerConfig): Promise<WorkerInstance> {
    const process = spawn('python', [
      config.command,
      '--worker-id', workerId,
      '--gpu-memory', config.gpuMemoryMB?.toString() || '1024',
      ...config.args || []
    ], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { 
        ...process.env, 
        CUDA_VISIBLE_DEVICES: '0',
        PYTHONUNBUFFERED: '1'
      }
    });

    // Monitor GPU memory usage
    const gpuMemoryMB = config.gpuMemoryMB || 1024;
    
    return {
      id: workerId,
      type: 'python-cuda',
      process,
      lastUsed: Date.now(),
      isIdle: true,
      gpuMemoryMB,
      requestCount: 0
    };
  }

  /**
   * Legal AI specific pool configurations
   */
  async initializeLegalAIPools(): Promise<any> {
    // Enhanced RAG Go service pool
    await this.registerWorker('enhanced-rag', {
      type: 'go',
      command: './go-microservice/bin/enhanced-rag.exe',
      maxInstances: 3,
      idleTimeout: 300000, // 5 minutes
      port: 8094
    });

    // Document processing Node.js pool
    await this.registerWorker('document-processor', {
      type: 'node',
      command: './src/workers/document-processor.js',
      maxInstances: 4,
      idleTimeout: 180000 // 3 minutes
    });

    // GPU-accelerated vector analysis
    await this.registerWorker('vector-analysis', {
      type: 'python-cuda',
      command: './python-workers/vector_analysis.py',
      maxInstances: 2,
      idleTimeout: 600000, // 10 minutes (keep GPU warm)
      gpuMemoryMB: 2048
    });

    // Legal entity extraction
    await this.registerWorker('entity-extraction', {
      type: 'python-cuda',
      command: './python-workers/legal_nlp.py',
      maxInstances: 2,
      idleTimeout: 300000,
      gpuMemoryMB: 1024
    });

    console.log('🏛️ Legal AI process pools initialized');
  }

  /**
   * Get comprehensive pool statistics
   */
  getStats(): Record<string, PoolStats> {
    const stats: Record<string, PoolStats> = {};
    
    for (const [poolName, pool] of this.pools) {
      const responseTimes = this.stats.get(poolName) || [];
      const avgResponseTime = responseTimes.length > 0 
        ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length 
        : 0;
      
      const totalRequests = pool.reduce((sum, w) => sum + w.requestCount, 0);
      const idleWorkers = pool.filter(w => w.isIdle).length;
      const nodeMemory = pool.filter(w => w.type === 'node').length * 50; // Estimate 50MB per Node worker
      const gpuMemory = pool.reduce((sum, w) => sum + (w.gpuMemoryMB || 0), 0);
      
      stats[poolName] = {
        totalWorkers: pool.length,
        idleWorkers,
        busyWorkers: pool.length - idleWorkers,
        totalRequests,
        averageResponseTime: avgResponseTime,
        memoryUsage: {
          node: nodeMemory,
          gpu: gpuMemory
        }
      };
    }
    
    return stats;
  }

  /**
   * Cleanup idle workers to free resources
   */
  private async cleanupIdleWorkers(): Promise<any> {
    const now = Date.now();
    
    for (const [poolName, pool] of this.pools) {
      const config = this.configs.get(poolName)!;
      
      // Remove workers idle longer than timeout
      const toRemove = pool.filter(w => 
        w.isIdle && (now - w.lastUsed) > config.idleTimeout
      );
      
      for (const worker of toRemove) {
        await this.terminateWorker(worker);
        const index = pool.indexOf(worker);
        if (index > -1) pool.splice(index, 1);
        
        this.emit('worker:terminated', { poolName, workerId: worker.id, reason: 'idle_timeout' });
      }
    }
  }

  /**
   * Terminate worker process
   */
  private async terminateWorker(worker: WorkerInstance): Promise<any> {
    if (worker.process) {
      if (worker.type === 'node') {
        await (worker.process as Worker).terminate();
      } else {
        (worker.process as ChildProcess).kill('SIGTERM');
      }
    }
    
    // Clean up Redis cache
    await this.redis.del(`legal_ai:worker_assignment:${worker.id}`);
  }

  /**
   * Utility: Find available port
   */
  private async findAvailablePort(startPort: number): Promise<number> {
    const net = await import('net');
    
    return new Promise((resolve) => {
      const server = net.createServer();
      server.listen(startPort, () => {
        const port = (server.address() as any).port;
        server.close(() => resolve(port));
      });
      server.on('error', () => {
        resolve(this.findAvailablePort(startPort + 1));
      });
    });
  }

  /**
   * Utility: Wait for service to be ready
   */
  private async waitForService(url: string, maxAttempts: number = 30): Promise<any> {
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const response = await fetch(url);
        if (response.ok) return;
      } catch (error: any) {
        // Service not ready yet
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    throw new Error(`Service at ${url} failed to start`);
  }

  /**
   * Shutdown all pools
   */
  async shutdown(): Promise<any> {
    clearInterval(this.cleanupInterval);
    
    for (const [poolName, pool] of this.pools) {
      for (const worker of pool) {
        await this.terminateWorker(worker);
      }
    }
    
    this.pools.clear();
    await this.redis.quit();
    this.emit('pools:shutdown');
  }
}

// Global singleton for the legal AI platform
export const legalAIProcessPool = new ProcessPoolManager();

// Initialize on module load
if (isMainThread) {
  legalAIProcessPool.initializeLegalAIPools().catch(console.error);
}