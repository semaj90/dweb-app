/**
 * Node.js Cluster Manager for Enhanced RAG Service
 * Standalone version that doesn't require VS Code dependencies
 */

import cluster from 'cluster';
import type { Worker } from 'cluster';
import * as os from 'os';
import { EventEmitter } from 'events';

export interface ClusterConfig {
  workers: number;
  maxMemoryPerWorker: number;
  restartThreshold: number;
  enableAutoRestart: boolean;
  workloadDistribution: 'round-robin' | 'least-loaded' | 'hash-based';
}

export interface WorkerTask {
  id: string;
  type: 'rag-query' | 'agent-orchestrate' | 'embed-cache' | 'auto-fix';
  data: unknown;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  timeout?: number;
}

export interface WorkerResponse {
  taskId: string;
  success: boolean;
  result?: unknown;
  error?: string;
  processingTime: number;
  workerId: number;
}

export interface WorkerStats {
  workerId: number;
  tasksProcessed: number;
  currentLoad: number;
  memoryUsage: NodeJS.MemoryUsage;
  uptime: number;
  status: 'idle' | 'busy' | 'overloaded' | 'error';
}

export class NodeClusterManager extends EventEmitter {
  private config: ClusterConfig;
  private workers: Map<number, Worker> = new Map();
  private taskQueue: Map<string, WorkerTask> = new Map();
  private workerStats: Map<number, WorkerStats> = new Map();
  private currentWorkerIndex = 0;
  private isShuttingDown = false;

  constructor(config?: Partial<ClusterConfig>) {
    super();
    
    this.config = {
      workers: config?.workers || Math.min(os.cpus().length, 4),
      maxMemoryPerWorker: config?.maxMemoryPerWorker || 512 * 1024 * 1024,
      restartThreshold: config?.restartThreshold || 0.9,
      enableAutoRestart: config?.enableAutoRestart ?? true,
      workloadDistribution: config?.workloadDistribution || 'least-loaded'
    };
  }

  /**
   * Initialize the cluster manager
   */
  async initialize(): Promise<void> {
    if (cluster.isPrimary) {
      await this.setupPrimary();
    } else {
      await this.setupWorker();
    }
  }

  /**
   * Set up primary process
   */
  private async setupPrimary(): Promise<void> {
    console.log(`Initializing cluster with ${this.config.workers} workers...`);

    // Fork workers
    for (let i = 0; i < this.config.workers; i++) {
      await this.forkWorker();
    }

    // Set up cluster event handlers
    cluster.on('exit', (worker, code, signal) => {
      console.log(`Worker ${worker.process.pid} died (${signal || code})`);
      this.handleWorkerExit(worker);
    });

    cluster.on('message', (worker, message) => {
      this.handleWorkerMessage(worker, message);
    });

    // Start monitoring workers
    this.startWorkerMonitoring();

    this.emit('cluster-ready', {
      workers: this.workers.size,
      config: this.config
    });
  }

  /**
   * Set up worker process
   */
  private async setupWorker(): Promise<void> {
    // Set up message handler for tasks
    process.on('message', async (message: unknown) => {
      if (message.type === 'task') {
        await this.executeWorkerTask(message.task);
      } else if (message.type === 'stats-request') {
        this.sendWorkerStats();
      }
    });

    // Send ready signal
    if (process.send) {
      process.send({
        type: 'worker-ready',
        workerId: process.pid,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Fork a new worker
   */
  private async forkWorker(): Promise<Worker> {
    const worker = cluster.fork({
      WORKER_TYPE: 'rag-agent',
      WORKER_ID: `worker-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    });

    this.workers.set(worker.id, worker);
    
    // Initialize worker stats
    this.workerStats.set(worker.id, {
      workerId: worker.id,
      tasksProcessed: 0,
      currentLoad: 0,
      memoryUsage: process.memoryUsage(),
      uptime: 0,
      status: 'idle'
    });

    return worker;
  }

  /**
   * Execute a task using the cluster
   */
  async executeTask(task: WorkerTask): Promise<WorkerResponse> {
    return new Promise((resolve, reject) => {
      const timeout = task.timeout || 30000;
      const timeoutId = setTimeout(() => {
        reject(new Error(`Task ${task.id} timed out after ${timeout}ms`));
      }, timeout);

      const worker = this.selectWorker(task);
      
      if (!worker) {
        clearTimeout(timeoutId);
        reject(new Error('No available workers'));
        return;
      }

      // Set up response handler
      const responseHandler = (message: unknown) => {
        if (message.type === 'task-complete' && message.taskId === task.id) {
          clearTimeout(timeoutId);
          worker.off('message', responseHandler);
          
          // Update worker stats
          const stats = this.workerStats.get(worker.id);
          if (stats) {
            stats.tasksProcessed++;
            stats.currentLoad = Math.max(0, stats.currentLoad - 1);
          }

          resolve({
            taskId: task.id,
            success: message.success,
            result: message.result,
            error: message.error,
            processingTime: message.processingTime,
            workerId: worker.id
          });
        }
      };

      worker.on('message', responseHandler);

      // Send task to worker
      worker.send({
        type: 'task',
        task: task
      });

      // Update worker load
      const stats = this.workerStats.get(worker.id);
      if (stats) {
        stats.currentLoad++;
        stats.status = stats.currentLoad > 2 ? 'busy' : 'idle';
      }
    });
  }

  /**
   * Select the best worker for a task
   */
  private selectWorker(task: WorkerTask): Worker | null {
    const availableWorkers = Array.from(this.workers.values()).filter(w => w.isDead() === false);
    
    if (availableWorkers.length === 0) {
      return null;
    }

    switch (this.config.workloadDistribution) {
      case 'round-robin':
        const worker = availableWorkers[this.currentWorkerIndex % availableWorkers.length];
        this.currentWorkerIndex++;
        return worker;

      case 'least-loaded':
        return availableWorkers.reduce((best, current) => {
          const bestStats = this.workerStats.get(best.id);
          const currentStats = this.workerStats.get(current.id);
          
          if (!bestStats) return current;
          if (!currentStats) return best;
          
          return currentStats.currentLoad < bestStats.currentLoad ? current : best;
        });

      case 'hash-based':
        const hash = this.hashString(task.id);
        return availableWorkers[hash % availableWorkers.length];

      default:
        return availableWorkers[0];
    }
  }

  /**
   * Execute a task in worker process
   */
  private async executeWorkerTask(task: WorkerTask): Promise<void> {
    const startTime = Date.now();
    let result: unknown;
    let success = true;
    let error: string | undefined;

    try {
      switch (task.type) {
        case 'rag-query':
          // Import and execute RAG query
          const { legalRAG } = await import('../sveltekit-frontend/src/lib/ai/langchain-rag.js');
          result = await legalRAG.query(task.data.question, task.data.options);
          break;

        case 'agent-orchestrate':
          // Import and execute agent orchestration
          try {
            const agentType = task.data.agent || 'claude';
            let agent;
            
            switch (agentType) {
              case 'claude':
                const { claudeAgent } = await import('../agents/claude-agent.js');
                agent = claudeAgent;
                break;
              case 'autogen':
                const { autoGenAgent } = await import('../agents/autogen-agent.js');
                agent = autoGenAgent;
                break;
              case 'crewai':
                const { crewAIAgent } = await import('../agents/crewai-agent.js');
                agent = crewAIAgent;
                break;
              default:
                throw new Error(`Unknown agent type: ${agentType}`);
            }

            result = await agent.execute(task.data);
          } catch (importError) {
            // Fallback to direct RAG if agents not available
            const { legalRAG } = await import('../sveltekit-frontend/src/lib/ai/langchain-rag.js');
            result = await legalRAG.query(task.data.prompt, task.data.options);
          }
          break;

        case 'auto-fix':
          // Import and execute auto-fix
          try {
            const { runAutoFix } = await import('../sveltekit-frontend/js_tests/sveltekit-best-practices-fix.mjs');
            result = await runAutoFix(task.data);
          } catch (importError) {
            result = {
              summary: { filesFixed: 0, totalIssues: 0 },
              fixes: {},
              recommendations: ['Auto-fix module not available']
            };
          }
          break;

        case 'embed-cache':
          // Handle embedding caching
          result = {
            cached: true,
            embeddings: task.data.text.length, // Mock embedding count
            similarity: 0.8
          };
          break;

        default:
          throw new Error(`Unknown task type: ${task.type}`);
      }

    } catch (err) {
      success = false;
      error = err instanceof Error ? err.message : 'Unknown error';

    } finally {
      const processingTime = Date.now() - startTime;

      // Send response back to primary
      if (process.send) {
        process.send({
          type: 'task-complete',
          taskId: task.id,
          success,
          result,
          error,
          processingTime
        });
      }
    }
  }

  /**
   * Handle worker exit
   */
  private handleWorkerExit(worker: Worker): void {
    if (this.isShuttingDown) return;

    console.log(`Worker ${worker.id} exited, cleaning up...`);
    
    this.workers.delete(worker.id);
    this.workerStats.delete(worker.id);

    // Restart worker if auto-restart is enabled
    if (this.config.enableAutoRestart) {
      setTimeout(() => {
        this.forkWorker().then(() => {
          console.log(`Worker ${worker.id} restarted`);
        });
      }, 1000);
    }
  }

  /**
   * Handle worker messages
   */
  private handleWorkerMessage(worker: Worker, message: unknown): void {
    switch (message.type) {
      case 'worker-ready':
        console.log(`Worker ${worker.id} is ready`);
        break;
      
      case 'stats-update':
        this.workerStats.set(worker.id, message.stats);
        break;
      
      case 'error':
        console.error(`Worker ${worker.id} error:`, message.error);
        this.emit('worker-error', { workerId: worker.id, error: message.error });
        break;
    }
  }

  /**
   * Start monitoring workers
   */
  private startWorkerMonitoring(): void {
    setInterval(() => {
      if (this.isShuttingDown) return;

      for (const [workerId, worker] of this.workers) {
        // Request stats from worker
        worker.send({ type: 'stats-request' });
        
        // Check memory usage and restart if needed
        const stats = this.workerStats.get(workerId);
        if (stats && stats.memoryUsage.heapUsed > this.config.maxMemoryPerWorker * this.config.restartThreshold) {
          console.log(`Worker ${workerId} memory usage too high, restarting...`);
          this.restartWorker(workerId);
        }
      }
    }, 10000); // Check every 10 seconds
  }

  /**
   * Send worker stats from worker process
   */
  private sendWorkerStats(): void {
    if (process.send) {
      process.send({
        type: 'stats-update',
        stats: {
          workerId: process.pid,
          memoryUsage: process.memoryUsage(),
          uptime: process.uptime(),
          timestamp: Date.now()
        }
      });
    }
  }

  /**
   * Restart a worker
   */
  private async restartWorker(workerId: number): Promise<void> {
    const worker = this.workers.get(workerId);
    if (worker) {
      worker.kill();
    }
  }

  /**
   * Get cluster statistics
   */
  getClusterStats(): {
    totalWorkers: number;
    activeWorkers: number;
    totalTasksProcessed: number;
    averageLoad: number;
    workerStats: WorkerStats[];
  } {
    const workerStats = Array.from(this.workerStats.values());
    const activeWorkers = workerStats.filter(s => s.status !== 'error').length;
    const totalTasks = workerStats.reduce((sum, s) => sum + s.tasksProcessed, 0);
    const averageLoad = workerStats.length > 0 
      ? workerStats.reduce((sum, s) => sum + s.currentLoad, 0) / workerStats.length 
      : 0;

    return {
      totalWorkers: this.workers.size,
      activeWorkers,
      totalTasksProcessed: totalTasks,
      averageLoad,
      workerStats
    };
  }

  /**
   * Shutdown cluster gracefully
   */
  async shutdown(): Promise<void> {
    this.isShuttingDown = true;
    
    console.log('Shutting down cluster...');
    
    // Send shutdown signal to all workers
    for (const worker of this.workers.values()) {
      worker.send({ type: 'shutdown' });
    }

    // Wait for workers to exit gracefully
    await new Promise<void>((resolve) => {
      let workersRemaining = this.workers.size;
      
      if (workersRemaining === 0) {
        resolve();
        return;
      }

      const checkWorkers = () => {
        workersRemaining--;
        if (workersRemaining === 0) {
          resolve();
        }
      };

      for (const worker of this.workers.values()) {
        worker.once('exit', checkWorkers);
      }

      // Force kill after 5 seconds
      setTimeout(() => {
        for (const worker of this.workers.values()) {
          if (!worker.isDead()) {
            worker.kill('SIGKILL');
          }
        }
        resolve();
      }, 5000);
    });

    console.log('Cluster shutdown complete');
  }

  /**
   * Hash string for consistent worker selection
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash);
  }
}

// Export singleton instance for Node.js environments
export const nodeClusterManager = new NodeClusterManager();