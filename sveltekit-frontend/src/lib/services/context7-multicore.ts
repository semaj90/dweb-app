/**
 * Context7 Multicore Service - Legal AI Integration
 * Production-ready multicore processing with Go-SIMD, Go-LLAMA, and MCP integration
 */

import { EventEmitter } from 'events';

export interface Context7MulticoreConfig {
  workerCount?: number;
  basePort?: number;
  loadBalancerPort?: number;
  enableGPU?: boolean;
  enableGoLlama?: boolean;
  enableLegalBert?: boolean;
  maxConcurrentTasks?: number;
  enableMCP?: boolean;
}

export interface WorkerInfo {
  id: string;
  port: number;
  status: 'initializing' | 'healthy' | 'busy' | 'error';
  lastHealth: Date;
  tasksProcessed: number;
  currentLoad: number;
  capabilities: string[];
}

export interface ProcessingTask {
  id: string;
  type:
    | 'tokenize'
    | 'semantic_analysis'
    | 'legal_classification'
    | 'tensor_parse'
    | 'json_parse'
    | 'recommendation';
  data: any;
  priority: 'low' | 'medium' | 'high' | 'critical';
  createdAt: Date;
  workerId?: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  result?: any;
  error?: string;
}

export interface LoadBalancerStatus {
  totalWorkers: number;
  healthyWorkers: number;
  totalRequests: number;
  requestsPerSecond: number;
  averageResponseTime: number;
  strategy: string;
  systemLoad: number;
}

export interface TensorData {
  shape: number[];
  dtype: 'float32' | 'float64' | 'int32' | 'int64';
  data: number[];
  metadata?: Record<string, any>;
}

export interface JSONParsingResult {
  valid: boolean;
  data?: any;
  error?: string;
  schema?: string;
  complexity: number;
}

export interface RecommendationRequest {
  context: string;
  errorType?: string;
  codeSnippet?: string;
  stackTrace?: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

export interface RecommendationResult {
  recommendations: Array<{
    solution: string;
    confidence: number;
    category: string;
    implementation: string;
    estimatedEffort: string;
  }>;
  context7Insights: string[];
  relatedErrors: string[];
  bestPractices: string[];
}

class Context7MulticoreService extends EventEmitter {
  private config: Required<Context7MulticoreConfig>;
  private workers: Map<string, WorkerInfo> = new Map();
  private taskQueue: ProcessingTask[] = [];
  private activeTasks: Map<string, ProcessingTask> = new Map();
  private loadBalancerHealth: LoadBalancerStatus | null = null;
  // Use ReturnType<typeof setInterval> for cross-environment compatibility (Node / Browser)
  private healthCheckInterval: ReturnType<typeof setInterval> | null = null;
  private taskProcessorInterval: ReturnType<typeof setInterval> | null = null;
  private metrics = {
    totalTasks: 0,
    completedTasks: 0,
    failedTasks: 0,
    averageProcessingTime: 0,
    systemUptime: 0,
  };

  constructor(config: Context7MulticoreConfig = {}) {
    super();

    this.config = {
      workerCount: config.workerCount || Math.min(8, require('os').cpus().length),
      basePort: config.basePort || 4100,
      loadBalancerPort: config.loadBalancerPort || 8099,
      enableGPU: config.enableGPU ?? true,
      enableGoLlama: config.enableGoLlama ?? true,
      enableLegalBert: config.enableLegalBert ?? true,
      maxConcurrentTasks: config.maxConcurrentTasks || 50,
      enableMCP: config.enableMCP ?? true,
    };

    this.initialize();
  }

  private async initialize() {
    console.log('🚀 Initializing Context7 Multicore Service');

    // Initialize workers
    await this.discoverWorkers();

    // Start health monitoring
    this.startHealthMonitoring();

    // Start task processor
    this.startTaskProcessor();

    // Check load balancer
    await this.checkLoadBalancer();

    console.log(`✅ Context7 Multicore Service initialized with ${this.workers.size} workers`);
    this.emit('initialized', { workerCount: this.workers.size });
  }

  private async discoverWorkers() {
    const discoveries = [];

    for (let i = 0; i < this.config.workerCount; i++) {
      const port = this.config.basePort + i;
      const workerId = `worker_${i + 1}`;

      discoveries.push(this.checkWorker(workerId, port));
    }

    const results = await Promise.allSettled(discoveries);

    results.forEach((result, index) => {
      if (result.status === 'fulfilled' && result.value) {
        const workerId = `worker_${index + 1}`;
        const port = this.config.basePort + index;

        this.workers.set(workerId, {
          id: workerId,
          port,
          status: 'healthy',
          lastHealth: new Date(),
          tasksProcessed: 0,
          currentLoad: 0,
          capabilities: this.getWorkerCapabilities(),
        });
      }
    });
  }

  private async checkWorker(workerId: string, port: number): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 3000);

      const response = await fetch(`http://localhost:${port}/health`, {
        signal: controller.signal,
      });

      clearTimeout(timeout);
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  private getWorkerCapabilities(): string[] {
    const capabilities = ['tokenize', 'semantic_analysis'];

    if (this.config.enableLegalBert) {
      capabilities.push('legal_classification', 'legal_ner', 'legal_sentiment');
    }

    if (this.config.enableGoLlama) {
      capabilities.push('llm_processing', 'text_generation');
    }

    if (this.config.enableGPU) {
      capabilities.push('gpu_acceleration', 'tensor_processing');
    }

    capabilities.push('json_parsing', 'recommendation_generation');

    return capabilities;
  }

  private startHealthMonitoring() {
    this.healthCheckInterval = setInterval(async () => {
      await this.performHealthChecks();
    }, 10000); // Check every 10 seconds
  }

  private async performHealthChecks() {
    const healthPromises = Array.from(this.workers.entries()).map(async ([workerId, worker]) => {
      const isHealthy = await this.checkWorker(workerId, worker.port);

      if (isHealthy) {
        worker.status = worker.currentLoad > 0.8 ? 'busy' : 'healthy';
        worker.lastHealth = new Date();
      } else {
        worker.status = 'error';
      }

      return { workerId, healthy: isHealthy };
    });

    const results = await Promise.allSettled(healthPromises);

    const healthyCount = results.filter(
      (result) => result.status === 'fulfilled' && result.value?.healthy
    ).length;

    this.emit('health_check_completed', {
      total: this.workers.size,
      healthy: healthyCount,
      timestamp: new Date(),
    });
  }

  private async checkLoadBalancer(): Promise<void> {
    try {
      const response = await fetch(`http://localhost:${this.config.loadBalancerPort}/status`);

      if (response.ok) {
        this.loadBalancerHealth = await response.json();
        console.log('📊 Load balancer is healthy');
      }
    } catch (error) {
      console.warn('⚠️ Load balancer not available');
      this.loadBalancerHealth = null;
    }
  }

  private startTaskProcessor() {
    this.taskProcessorInterval = setInterval(() => {
      this.processQueuedTasks();
    }, 1000);
  }

  private async processQueuedTasks() {
    if (this.taskQueue.length === 0 || this.activeTasks.size >= this.config.maxConcurrentTasks) {
      return;
    }

    // Sort tasks by priority and age
    this.taskQueue.sort((a, b) => {
      const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];

      if (priorityDiff !== 0) return priorityDiff;

      return a.createdAt.getTime() - b.createdAt.getTime();
    });

    const availableWorkers = Array.from(this.workers.values()).filter(
      (worker) => worker.status === 'healthy' && worker.currentLoad < 0.8
    );

    if (availableWorkers.length === 0) return;

    const tasksToProcess = Math.min(
      this.taskQueue.length,
      availableWorkers.length,
      this.config.maxConcurrentTasks - this.activeTasks.size
    );

    for (let i = 0; i < tasksToProcess; i++) {
      const task = this.taskQueue.shift()!;
      const worker = this.selectBestWorker(task, availableWorkers);

      if (worker) {
        await this.assignTaskToWorker(task, worker);
      } else {
        // Put task back in queue if no suitable worker found
        this.taskQueue.unshift(task);
        break;
      }
    }
  }

  private selectBestWorker(
    task: ProcessingTask,
    availableWorkers: WorkerInfo[]
  ): WorkerInfo | null {
    // Filter workers by capability
    const capableWorkers = availableWorkers.filter((worker) =>
      this.workerCanHandleTask(worker, task)
    );

    if (capableWorkers.length === 0) return null;

    // Select worker with lowest load
    return capableWorkers.reduce((best, current) =>
      current.currentLoad < best.currentLoad ? current : best
    );
  }

  private workerCanHandleTask(worker: WorkerInfo, task: ProcessingTask): boolean {
    const requiredCapabilities = this.getRequiredCapabilities(task.type);
    return requiredCapabilities.every((cap) => worker.capabilities.includes(cap));
  }

  private getRequiredCapabilities(taskType: string): string[] {
    switch (taskType) {
      case 'tokenize':
        return ['tokenize'];
      case 'semantic_analysis':
        return ['semantic_analysis'];
      case 'legal_classification':
        return ['legal_classification'];
      case 'tensor_parse':
        return ['tensor_processing'];
      case 'json_parse':
        return ['json_parsing'];
      case 'recommendation':
        return ['recommendation_generation'];
      default:
        return [];
    }
  }

  private async assignTaskToWorker(task: ProcessingTask, worker: WorkerInfo): Promise<void> {
    task.status = 'processing';
    task.workerId = worker.id;
    this.activeTasks.set(task.id, task);

    worker.currentLoad += 0.2; // Approximate load increase

    try {
      const result = await this.executeTaskOnWorker(task, worker);

      task.status = 'completed';
      task.result = result;
      this.metrics.completedTasks++;

      this.emit('task_completed', { task, result });
    } catch (error: any) {
      task.status = 'failed';
      task.error = error.message;
      this.metrics.failedTasks++;

      this.emit('task_failed', { task, error: error.message });
    } finally {
      this.activeTasks.delete(task.id);
      worker.currentLoad = Math.max(0, worker.currentLoad - 0.2);
      worker.tasksProcessed++;
    }
  }

  private async executeTaskOnWorker(task: ProcessingTask, worker: WorkerInfo): Promise<any> {
    const endpoint = this.getWorkerEndpoint(task.type);
    const url = `http://localhost:${worker.port}${endpoint}`;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000); // 30s timeout

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(task.data),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        throw new Error(`Worker request failed: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeout);
      throw error;
    }
  }

  private getWorkerEndpoint(taskType: string): string {
    switch (taskType) {
      case 'tokenize':
        return '/tokenize';
      case 'semantic_analysis':
        return '/semantic-analysis';
      case 'legal_classification':
        return '/legal-bert';
      case 'tensor_parse':
        return '/tensor-parse';
      case 'json_parse':
        return '/json-parse';
      case 'recommendation':
        return '/recommendation';
      default:
        throw new Error(`Unknown task type: ${taskType}`);
    }
  }

  // Public API Methods

  async processText(
    text: string,
    type: 'tokenize' | 'semantic_analysis' | 'legal_classification' = 'tokenize',
    priority: ProcessingTask['priority'] = 'medium'
  ): Promise<ProcessingTask> {
    const task: ProcessingTask = {
      id: this.generateTaskId(),
      type,
      data: { text },
      priority,
      createdAt: new Date(),
      status: 'queued',
    };

    this.taskQueue.push(task);
    this.metrics.totalTasks++;

    this.emit('task_queued', { task });

    return task;
  }

  async parseJSON(
    jsonString: string,
    schema?: string,
    priority: ProcessingTask['priority'] = 'medium'
  ): Promise<ProcessingTask> {
    const task: ProcessingTask = {
      id: this.generateTaskId(),
      type: 'json_parse',
      data: { jsonString, schema },
      priority,
      createdAt: new Date(),
      status: 'queued',
    };

    this.taskQueue.push(task);
    this.metrics.totalTasks++;

    return task;
  }

  async parseTensor(
    tensorData: TensorData,
    priority: ProcessingTask['priority'] = 'medium'
  ): Promise<ProcessingTask> {
    const task: ProcessingTask = {
      id: this.generateTaskId(),
      type: 'tensor_parse',
      data: tensorData,
      priority,
      createdAt: new Date(),
      status: 'queued',
    };

    this.taskQueue.push(task);
    this.metrics.totalTasks++;

    return task;
  }

  async generateRecommendations(
    request: RecommendationRequest,
    priority: ProcessingTask['priority'] = 'high'
  ): Promise<ProcessingTask> {
    const task: ProcessingTask = {
      id: this.generateTaskId(),
      type: 'recommendation',
      data: request,
      priority,
      createdAt: new Date(),
      status: 'queued',
    };

    this.taskQueue.push(task);
    this.metrics.totalTasks++;

    return task;
  }

  async getTaskStatus(taskId: string): Promise<ProcessingTask | null> {
    const activeTask = this.activeTasks.get(taskId);
    if (activeTask) return activeTask;

    const queuedTask = this.taskQueue.find((task) => task.id === taskId);
    return queuedTask || null;
  }

  async waitForTask(taskId: string, timeoutMs: number = 30000): Promise<ProcessingTask> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.removeListener('task_completed', onCompleted);
        this.removeListener('task_failed', onFailed);
        reject(new Error('Task timeout'));
      }, timeoutMs);

      const onCompleted = ({ task }: { task: ProcessingTask }) => {
        if (task.id === taskId) {
          clearTimeout(timeout);
          this.removeListener('task_failed', onFailed);
          resolve(task);
        }
      };

      const onFailed = ({ task }: { task: ProcessingTask }) => {
        if (task.id === taskId) {
          clearTimeout(timeout);
          this.removeListener('task_completed', onCompleted);
          reject(new Error(task.error || 'Task failed'));
        }
      };

      this.on('task_completed', onCompleted);
      this.on('task_failed', onFailed);

      // Check if task is already completed
      this.getTaskStatus(taskId).then((task) => {
        if (task && (task.status === 'completed' || task.status === 'failed')) {
          clearTimeout(timeout);
          this.removeListener('task_completed', onCompleted);
          this.removeListener('task_failed', onFailed);

          if (task.status === 'completed') {
            resolve(task);
          } else {
            reject(new Error(task.error || 'Task failed'));
          }
        }
      });
    });
  }

  getSystemStatus(): {
    workers: WorkerInfo[];
    loadBalancer: LoadBalancerStatus | null;
    metrics: typeof this.metrics;
    queue: { size: number; activeTasks: number };
  } {
    return {
      workers: Array.from(this.workers.values()),
      loadBalancer: this.loadBalancerHealth,
      metrics: { ...this.metrics },
      queue: {
        size: this.taskQueue.length,
        activeTasks: this.activeTasks.size,
      },
    };
  }

  private generateTaskId(): string {
    return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down Context7 Multicore Service');

    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    if (this.taskProcessorInterval) {
      clearInterval(this.taskProcessorInterval);
    }

    // Wait for active tasks to complete (with timeout)
    const activeTaskIds = Array.from(this.activeTasks.keys());
    if (activeTaskIds.length > 0) {
      console.log(`⏳ Waiting for ${activeTaskIds.length} active tasks to complete...`);

      try {
        await Promise.race([
          Promise.all(activeTaskIds.map((id) => this.waitForTask(id, 5000))),
          new Promise((resolve) => setTimeout(resolve, 10000)), // Max 10s wait
        ]);
      } catch (error) {
        console.warn('⚠️ Some tasks did not complete before shutdown');
      }
    }

    this.emit('shutdown');
    console.log('✅ Context7 Multicore Service shutdown complete');
  }
}

// Singleton instance
let instance: Context7MulticoreService | null = null;

export function getContext7MulticoreService(
  config?: Context7MulticoreConfig
): Context7MulticoreService {
  if (!instance) {
    instance = new Context7MulticoreService(config);
  }
  return instance;
}

export function createContext7MulticoreService(
  config?: Context7MulticoreConfig
): Context7MulticoreService {
  return new Context7MulticoreService(config);
}

// Helper functions for JSON and Tensor parsing
export async function parseJSONAdvanced(
  jsonString: string,
  schema?: string
): Promise<JSONParsingResult> {
  try {
    const data = JSON.parse(jsonString);

    let complexity = 0;
    const calculateComplexity = (obj: any, depth = 0): number => {
      if (depth > 10) return complexity; // Prevent infinite recursion

      if (Array.isArray(obj)) {
        complexity += obj.length;
        obj.forEach((item) => calculateComplexity(item, depth + 1));
      } else if (typeof obj === 'object' && obj !== null) {
        complexity += Object.keys(obj).length;
        Object.values(obj).forEach((value) => calculateComplexity(value, depth + 1));
      }

      return complexity;
    };

    return {
      valid: true,
      data,
      schema: schema || 'auto-detected',
      complexity: calculateComplexity(data),
    };
  } catch (error: any) {
    return {
      valid: false,
      error: error.message,
      complexity: 0,
    };
  }
}

export function createTensorData(
  shape: number[],
  data: number[],
  dtype: TensorData['dtype'] = 'float32',
  metadata?: Record<string, any>
): TensorData {
  const expectedSize = shape.reduce((acc, dim) => acc * dim, 1);

  if (data.length !== expectedSize) {
    throw new Error(
      `Data length ${data.length} does not match shape ${shape} (expected ${expectedSize})`
    );
  }

  return {
    shape,
    dtype,
    data: [...data], // Copy array
    metadata: metadata ? { ...metadata } : undefined,
  };
}

export { Context7MulticoreService };
export default getContext7MulticoreService;
