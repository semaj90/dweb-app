import cluster from 'node:cluster';
import { cpus } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { env } from '$env/dynamic/private';

/**
 * Advanced Node.js Clustering + Worker Threads for Legal AI System
 * Optimized for high-concurrency legal document processing and AI inference
 */

export interface ClusterConfig {
  workers: number;
  maxMemoryUsage: number;
  restartThreshold: number;
  enableGracefulShutdown: boolean;
  workerPoolSize: number;
  maxWorkerThreads: number;
}

export interface WorkerTask {
  id: string;
  type: 'document_analysis' | 'vector_search' | 'ai_inference' | 'data_processing';
  payload: any;
  priority: 'low' | 'medium' | 'high' | 'critical';
  timeout?: number;
}

export interface WorkerResult {
  taskId: string;
  success: boolean;
  result?: any;
  error?: string;
  processingTime: number;
  workerId: number;
  memoryUsage: number;
}

export class LegalAIClusterManager {
  private config: ClusterConfig;
  private workerPool: Worker[] = [];
  private taskQueue: WorkerTask[] = [];
  private activeTasks = new Map<string, { worker: Worker; startTime: number }>();
  private workerMetrics = new Map<number, {
    tasksCompleted: number;
    totalProcessingTime: number;
    memoryUsage: number;
    errors: number;
  }>();

  constructor(config?: Partial<ClusterConfig>) {
    this.config = {
      workers: parseInt(env.CLUSTER_WORKERS || '4'),
      maxMemoryUsage: parseInt(env.MAX_MEMORY_USAGE || '2048') * 1024 * 1024, // Convert MB to bytes
      restartThreshold: 1000, // Restart worker after 1000 tasks
      enableGracefulShutdown: true,
      workerPoolSize: parseInt(env.WORKER_POOL_SIZE || '4'),
      maxWorkerThreads: parseInt(env.MAX_WORKER_THREADS || '8'),
      ...config
    };
  }

  /**
   * Initialize the cluster manager
   */
  async initialize(): Promise<void> {
    if (cluster.isPrimary && env.ENABLE_CLUSTERING === 'true') {
      await this.initializePrimaryProcess();
    } else {
      await this.initializeWorkerProcess();
    }
  }

  private async initializePrimaryProcess(): Promise<void> {
    console.log(`🚀 Starting Legal AI Cluster Manager with ${this.config.workers} workers`);

    // Fork worker processes
    for (let i = 0; i < this.config.workers; i++) {
      const worker = cluster.fork();
      this.setupWorkerEventHandlers(worker);
    }

    // Initialize worker thread pool
    await this.initializeWorkerThreadPool();

    // Setup cluster event handlers
    this.setupClusterEventHandlers();

    // Start health monitoring
    this.startHealthMonitoring();

    console.log('✅ Legal AI Cluster Manager initialized successfully');
  }

  private async initializeWorkerProcess(): Promise<void> {
    console.log(`👷 Worker ${process.pid} started`);
    
    // Initialize worker thread pool for this process
    await this.initializeWorkerThreadPool();
    
    // Setup graceful shutdown
    this.setupGracefulShutdown();
  }

  private async initializeWorkerThreadPool(): Promise<void> {
    for (let i = 0; i < this.config.workerPoolSize; i++) {
      const worker = new Worker(new URL('./worker-thread.js', import.meta.url), {
        workerData: { workerId: i }
      });

      this.setupWorkerThreadHandlers(worker, i);
      this.workerPool.push(worker);
      
      // Initialize worker metrics
      this.workerMetrics.set(i, {
        tasksCompleted: 0,
        totalProcessingTime: 0,
        memoryUsage: 0,
        errors: 0
      });
    }

    console.log(`✅ Worker thread pool initialized with ${this.config.workerPoolSize} threads`);
  }

  private setupWorkerEventHandlers(worker: any): void {
    worker.on('exit', (code: number | null, signal: string | null) => {
      console.log(`⚠️ Worker ${worker.process.pid} died (${signal || code}). Restarting...`);
      
      if (!worker.isDead()) {
        worker.kill();
      }
      
      // Fork a new worker
      const newWorker = cluster.fork();
      this.setupWorkerEventHandlers(newWorker);
    });

    worker.on('error', (error: Error) => {
      console.error(`❌ Worker ${worker.process.pid} error:`, error);
    });

    worker.on('message', (message: any) => {
      this.handleWorkerMessage(worker, message);
    });
  }

  private setupWorkerThreadHandlers(worker: Worker, workerId: number): void {
    worker.on('message', (result: WorkerResult) => {
      this.handleWorkerThreadResult(result);
    });

    worker.on('error', (error) => {
      console.error(`❌ Worker thread ${workerId} error:`, error);
      this.updateWorkerMetrics(workerId, { errors: 1 });
    });

    worker.on('exit', (code) => {
      if (code !== 0) {
        console.log(`⚠️ Worker thread ${workerId} stopped with exit code ${code}`);
        // Restart worker thread
        this.restartWorkerThread(workerId);
      }
    });
  }

  private setupClusterEventHandlers(): void {
    cluster.on('exit', (worker, code, signal) => {
      console.log(`Worker ${worker.process.pid} died`);
    });

    // Handle graceful shutdown
    process.on('SIGTERM', () => this.gracefulShutdown());
    process.on('SIGINT', () => this.gracefulShutdown());
  }

  private setupGracefulShutdown(): void {
    process.on('SIGTERM', () => {
      console.log('🔄 Worker process received SIGTERM, shutting down gracefully...');
      this.shutdownWorkerThreads();
    });

    process.on('SIGINT', () => {
      console.log('🔄 Worker process received SIGINT, shutting down gracefully...');
      this.shutdownWorkerThreads();
    });
  }

  /**
   * Submit a task to the worker thread pool
   */
  async submitTask(task: WorkerTask): Promise<WorkerResult> {
    return new Promise((resolve, reject) => {
      // Add timeout handling
      const timeout = task.timeout || 30000; // Default 30 seconds
      const timeoutId = setTimeout(() => {
        reject(new Error(`Task ${task.id} timed out after ${timeout}ms`));
      }, timeout);

      // Find available worker
      const availableWorker = this.findAvailableWorker();
      
      if (!availableWorker) {
        // Queue the task if no workers available
        this.taskQueue.push(task);
        reject(new Error('No available workers, task queued'));
        return;
      }

      // Set up one-time result handler
      const resultHandler = (result: WorkerResult) => {
        if (result.taskId === task.id) {
          clearTimeout(timeoutId);
          availableWorker.off('message', resultHandler);
          this.activeTasks.delete(task.id);
          
          if (result.success) {
            resolve(result);
          } else {
            reject(new Error(result.error));
          }
          
          // Process next queued task if any
          this.processQueuedTasks();
        }
      };

      availableWorker.on('message', resultHandler);
      
      // Send task to worker
      this.activeTasks.set(task.id, { worker: availableWorker, startTime: Date.now() });
      availableWorker.postMessage(task);
    });
  }

  /**
   * Submit multiple tasks in parallel
   */
  async submitBatchTasks(tasks: WorkerTask[]): Promise<WorkerResult[]> {
    const taskPromises = tasks.map(task => this.submitTask(task));
    return Promise.allSettled(taskPromises).then(results => 
      results.map((result, index) => {
        if (result.status === 'fulfilled') {
          return result.value;
        } else {
          return {
            taskId: tasks[index].id,
            success: false,
            error: result.reason.message,
            processingTime: 0,
            workerId: -1,
            memoryUsage: 0
          };
        }
      })
    );
  }

  private findAvailableWorker(): Worker | null {
    // Find worker with least active tasks
    let bestWorker: Worker | null = null;
    let minActiveTasks = Infinity;

    for (const worker of this.workerPool) {
      const activeTasks = Array.from(this.activeTasks.values())
        .filter(task => task.worker === worker).length;
      
      if (activeTasks < minActiveTasks) {
        minActiveTasks = activeTasks;
        bestWorker = worker;
      }
    }

    return minActiveTasks < 2 ? bestWorker : null; // Max 2 tasks per worker
  }

  private processQueuedTasks(): void {
    if (this.taskQueue.length === 0) return;

    // Sort queue by priority
    this.taskQueue.sort((a, b) => {
      const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });

    const availableWorker = this.findAvailableWorker();
    if (availableWorker && this.taskQueue.length > 0) {
      const task = this.taskQueue.shift()!;
      this.submitTask(task);
    }
  }

  private handleWorkerMessage(worker: any, message: any): void {
    // Handle inter-process communication between primary and worker processes
    switch (message.type) {
      case 'health_check':
        worker.send({ type: 'health_response', status: 'healthy' });
        break;
      case 'metrics_request':
        worker.send({ 
          type: 'metrics_response', 
          metrics: this.getClusterMetrics() 
        });
        break;
    }
  }

  private handleWorkerThreadResult(result: WorkerResult): void {
    // Update worker metrics
    this.updateWorkerMetrics(result.workerId, {
      tasksCompleted: 1,
      totalProcessingTime: result.processingTime,
      memoryUsage: result.memoryUsage,
      errors: result.success ? 0 : 1
    });

    // Check if worker needs restart due to memory usage or task count
    const metrics = this.workerMetrics.get(result.workerId);
    if (metrics) {
      if (metrics.tasksCompleted >= this.config.restartThreshold ||
          metrics.memoryUsage > this.config.maxMemoryUsage) {
        this.restartWorkerThread(result.workerId);
      }
    }
  }

  private updateWorkerMetrics(workerId: number, updates: Partial<{
    tasksCompleted: number;
    totalProcessingTime: number;
    memoryUsage: number;
    errors: number;
  }>): void {
    const current = this.workerMetrics.get(workerId) || {
      tasksCompleted: 0,
      totalProcessingTime: 0,
      memoryUsage: 0,
      errors: 0
    };

    this.workerMetrics.set(workerId, {
      tasksCompleted: current.tasksCompleted + (updates.tasksCompleted || 0),
      totalProcessingTime: current.totalProcessingTime + (updates.totalProcessingTime || 0),
      memoryUsage: Math.max(current.memoryUsage, updates.memoryUsage || 0),
      errors: current.errors + (updates.errors || 0)
    });
  }

  private restartWorkerThread(workerId: number): void {
    console.log(`🔄 Restarting worker thread ${workerId}`);
    
    const oldWorker = this.workerPool[workerId];
    if (oldWorker) {
      oldWorker.terminate();
    }

    const newWorker = new Worker(new URL('./worker-thread.js', import.meta.url), {
      workerData: { workerId }
    });

    this.setupWorkerThreadHandlers(newWorker, workerId);
    this.workerPool[workerId] = newWorker;

    // Reset metrics
    this.workerMetrics.set(workerId, {
      tasksCompleted: 0,
      totalProcessingTime: 0,
      memoryUsage: 0,
      errors: 0
    });
  }

  private startHealthMonitoring(): void {
    setInterval(() => {
      this.performHealthCheck();
    }, 30000); // Check every 30 seconds
  }

  private performHealthCheck(): void {
    const clusterMetrics = this.getClusterMetrics();
    
    // Log health status
    console.log('🏥 Cluster Health:', {
      totalWorkers: clusterMetrics.totalWorkers,
      activeThreads: clusterMetrics.activeWorkerThreads,
      queuedTasks: this.taskQueue.length,
      avgResponseTime: clusterMetrics.avgResponseTime,
      memoryUsage: clusterMetrics.totalMemoryUsage
    });

    // Alert on high memory usage
    if (clusterMetrics.totalMemoryUsage > this.config.maxMemoryUsage * 0.8) {
      console.warn('⚠️ High memory usage detected:', clusterMetrics.totalMemoryUsage);
    }

    // Alert on high queue length
    if (this.taskQueue.length > 100) {
      console.warn('⚠️ High task queue length:', this.taskQueue.length);
    }
  }

  /**
   * Get cluster metrics
   */
  getClusterMetrics() {
    const workerMetrics = Array.from(this.workerMetrics.values());
    
    return {
      totalWorkers: this.config.workers,
      activeWorkerThreads: this.workerPool.length,
      queuedTasks: this.taskQueue.length,
      activeTasks: this.activeTasks.size,
      totalTasksCompleted: workerMetrics.reduce((sum, m) => sum + m.tasksCompleted, 0),
      totalErrors: workerMetrics.reduce((sum, m) => sum + m.errors, 0),
      avgResponseTime: workerMetrics.reduce((sum, m) => sum + m.totalProcessingTime, 0) / 
                      Math.max(1, workerMetrics.reduce((sum, m) => sum + m.tasksCompleted, 0)),
      totalMemoryUsage: workerMetrics.reduce((sum, m) => sum + m.memoryUsage, 0),
      successRate: (() => {
        const total = workerMetrics.reduce((sum, m) => sum + m.tasksCompleted, 0);
        const errors = workerMetrics.reduce((sum, m) => sum + m.errors, 0);
        return total > 0 ? ((total - errors) / total) * 100 : 100;
      })()
    };
  }

  private async gracefulShutdown(): Promise<void> {
    console.log('🔄 Starting graceful shutdown...');
    
    // Stop accepting new tasks
    this.taskQueue = [];
    
    // Wait for active tasks to complete (with timeout)
    const shutdownTimeout = 30000; // 30 seconds
    const startTime = Date.now();
    
    while (this.activeTasks.size > 0 && (Date.now() - startTime) < shutdownTimeout) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    // Terminate worker threads
    await this.shutdownWorkerThreads();
    
    // Exit process
    process.exit(0);
  }

  private async shutdownWorkerThreads(): Promise<void> {
    console.log('🔄 Shutting down worker threads...');
    
    const shutdownPromises = this.workerPool.map(worker => 
      worker.terminate().catch(err => console.error('Worker termination error:', err))
    );
    
    await Promise.allSettled(shutdownPromises);
    console.log('✅ All worker threads terminated');
  }
}

// Export singleton instance
export const clusterManager = new LegalAIClusterManager();

// Auto-initialize if clustering is enabled
if (env.ENABLE_CLUSTERING === 'true') {
  clusterManager.initialize().catch(error => {
    console.error('❌ Cluster initialization failed:', error);
    process.exit(1);
  });
}