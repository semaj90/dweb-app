// @ts-nocheck
/**
 * AI Worker Manager - Orchestrates multiple AI workers for parallel processing
 * Handles task distribution, load balancing, and worker lifecycle management
 */

import type {
  AITask,
  AIResponse,
  WorkerMessage,
  WorkerStatus,
  WorkerPool,
  AIServiceWorkerManager,
  TaskResult,
  ProcessingMetrics,
  WorkerConfiguration,
} from "$lib/types/ai-worker.js";

export class AIWorkerManager implements AIServiceWorkerManager {
  private workerPool: WorkerPool;
  private activeTasks: Map<
    string,
    {
      task: AITask;
      workerId: number;
      startTime: number;
      resolve: (result: TaskResult) => void;
      reject: (error: Error) => void;
    }
  > = new Map();

  private config: WorkerConfiguration;
  private metrics: Map<string, ProcessingMetrics> = new Map();
  private isInitialized = false;

  // Event handlers
  onTaskComplete?: (taskId: string, response: AIResponse) => void;
  onTaskError?: (taskId: string, error: Error) => void;
  onStatusUpdate?: (status: WorkerStatus) => void;

  constructor(config: Partial<WorkerConfiguration> = {}) {
    this.config = {
      maxConcurrentTasks: 3,
      defaultTimeout: 30000,
      retryAttempts: 2,
      enableMetrics: true,
      enableLogging: true,
      providers: [],
      ...config,
    };

    this.workerPool = {
      workers: [],
      taskDistribution: "least-loaded",
      maxWorkers: Math.min(navigator.hardwareConcurrency || 4, 6),
      currentLoad: [],
      totalTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
    };
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      // Create worker pool
      for (let i = 0; i < this.workerPool.maxWorkers; i++) {
        const worker = new Worker(
          new URL("../workers/ai-service-worker.ts", import.meta.url),
          { type: "module" },
        );

        this.setupWorkerEventHandlers(worker, i);
        this.workerPool.workers.push(worker);
        this.workerPool.currentLoad.push(0);

        // Send initial configuration
        worker.postMessage({
          type: "UPDATE_PROVIDER_CONFIG",
          taskId: `init-${i}`,
          payload: this.config.providers,
        });
      }

      this.isInitialized = true;

      if (this.config.enableLogging) {
        console.log(
          `AI Worker Manager initialized with ${this.workerPool.maxWorkers} workers`,
        );
      }
    } catch (error) {
      console.error("Failed to initialize AI Worker Manager:", error);
      throw error;
    }
  }

  private setupWorkerEventHandlers(worker: Worker, workerId: number) {
    worker.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
      const { type, taskId, payload } = event.data;

      switch (type) {
        case "TASK_STARTED":
          this.handleTaskStarted(taskId, workerId);
          break;
        case "TASK_COMPLETED":
          this.handleTaskCompleted(taskId, payload as AIResponse, workerId);
          break;
        case "TASK_ERROR":
          this.handleTaskError(taskId, payload, workerId);
          break;
        case "TASK_CANCELLED":
          this.handleTaskCancelled(taskId, workerId);
          break;
        case "STATUS_UPDATE":
          this.handleStatusUpdate(payload);
          break;
        default:
          if (this.config.enableLogging) {
            console.log(`Worker ${workerId} message:`, type, payload);
          }
      }
    });

    worker.addEventListener("error", (error) => {
      console.error(`Worker ${workerId} error:`, error);
      this.handleWorkerError(workerId, error);
    });
  }

  async submitTask(task: AITask): Promise<string> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const taskId = task.taskId || crypto.randomUUID();
    const enhancedTask: AITask = {
      ...task,
      taskId,
      timestamp: Date.now(),
    };

    return new Promise((resolve, reject) => {
      const workerId = this.selectWorker(enhancedTask);
      const worker = this.workerPool.workers[workerId];

      if (!worker) {
        reject(new Error("No available workers"));
        return;
      }

      // Track active task
      this.activeTasks.set(taskId, {
        task: enhancedTask,
        workerId,
        startTime: Date.now(),
        resolve: (result: TaskResult) => resolve(result.taskId),
        reject,
      });

      // Update worker load
      this.workerPool.currentLoad[workerId]++;
      this.workerPool.totalTasks++;

      // Send task to worker
      worker.postMessage({
        type: "PROCESS_AI_TASK",
        taskId,
        payload: enhancedTask,
      });

      // Set timeout
      setTimeout(() => {
        if (this.activeTasks.has(taskId)) {
          this.cancelTask(taskId);
          reject(new Error(`Task ${taskId} timed out`));
        }
      }, this.config.defaultTimeout);

      if (this.config.enableLogging) {
        console.log(`Task ${taskId} submitted to worker ${workerId}`);
      }
    });
  }

  private selectWorker(task: AITask): number {
    switch (this.workerPool.taskDistribution) {
      case "round-robin":
        return this.workerPool.totalTasks % this.workerPool.maxWorkers;

      case "least-loaded":
        return this.workerPool.currentLoad.indexOf(
          Math.min(...this.workerPool.currentLoad),
        );

      case "priority-based":
        // High priority tasks go to least loaded worker
        if (task.priority === "high" || task.priority === "critical") {
          return this.workerPool.currentLoad.indexOf(
            Math.min(...this.workerPool.currentLoad),
          );
        }
        // Low priority tasks use round-robin
        return this.workerPool.totalTasks % this.workerPool.maxWorkers;

      default:
        return 0;
    }
  }

  async cancelTask(taskId: string): Promise<void> {
    const activeTask = this.activeTasks.get(taskId);
    if (!activeTask) {
      return;
    }

    const worker = this.workerPool.workers[activeTask.workerId];
    if (worker) {
      worker.postMessage({
        type: "CANCEL_TASK",
        taskId,
        payload: null,
      });
    }

    this.cleanupTask(taskId, activeTask.workerId);
  }

  async getStatus(): Promise<WorkerStatus> {
    const workerStatuses = await Promise.all(
      this.workerPool.workers.map((worker, index) =>
        this.getWorkerStatus(worker, index),
      ),
    );

    return {
      activeRequests: this.activeTasks.size,
      queueLength: workerStatuses.reduce(
        (sum, status) => sum + status.queueLength,
        0,
      ),
      providers: [], // Aggregated from workers
      maxConcurrent: this.config.maxConcurrentTasks,
      uptime: Date.now(),
      totalProcessed: this.workerPool.completedTasks,
      errors: this.workerPool.failedTasks,
    };
  }

  private async getWorkerStatus(
    worker: Worker,
    workerId: number,
  ): Promise<WorkerStatus> {
    return new Promise((resolve) => {
      const timeoutId = setTimeout(() => {
        resolve({
          activeRequests: this.workerPool.currentLoad[workerId],
          queueLength: 0,
          providers: [],
          maxConcurrent: this.config.maxConcurrentTasks,
          uptime: 0,
          totalProcessed: 0,
          errors: 0,
        });
      }, 1000);

      const messageHandler = (event: MessageEvent<WorkerMessage>) => {
        if (event.data.type === "STATUS_UPDATE") {
          clearTimeout(timeoutId);
          worker.removeEventListener("message", messageHandler);
          resolve(event.data.payload);
        }
      };

      worker.addEventListener("message", messageHandler);
      worker.postMessage({
        type: "GET_STATUS",
        taskId: `status-${workerId}`,
        payload: null,
      });
    });
  }

  async shutdown(): Promise<void> {
    // Cancel all active tasks
    const cancelPromises = Array.from(this.activeTasks.keys()).map((taskId) =>
      this.cancelTask(taskId),
    );
    await Promise.all(cancelPromises);

    // Terminate all workers
    this.workerPool.workers.forEach((worker) => worker.terminate());

    // Clear state
    this.workerPool.workers = [];
    this.workerPool.currentLoad = [];
    this.activeTasks.clear();
    this.metrics.clear();
    this.isInitialized = false;

    if (this.config.enableLogging) {
      console.log("AI Worker Manager shutdown completed");
    }
  }

  private handleTaskStarted(taskId: string, workerId: number) {
    const activeTask = this.activeTasks.get(taskId);
    if (activeTask && this.config.enableMetrics) {
      const metrics: ProcessingMetrics = {
        taskId,
        startTime: Date.now(),
        queueTime: Date.now() - activeTask.startTime,
        retries: 0,
        provider: activeTask.task.providerId,
        model: activeTask.task.model,
        tokensProcessed: 0,
        success: false,
      };
      this.metrics.set(taskId, metrics);
    }
  }

  private handleTaskCompleted(
    taskId: string,
    response: AIResponse,
    workerId: number,
  ) {
    const activeTask = this.activeTasks.get(taskId);
    if (!activeTask) return;

    const result: TaskResult = {
      taskId,
      status: "completed",
      response,
      metrics: this.updateMetrics(taskId, response, true),
    };

    activeTask.resolve(result);
    this.cleanupTask(taskId, workerId);
    this.workerPool.completedTasks++;

    if (this.onTaskComplete) {
      this.onTaskComplete(taskId, response);
    }
  }

  private handleTaskError(taskId: string, error: any, workerId: number) {
    const activeTask = this.activeTasks.get(taskId);
    if (!activeTask) return;

    const errorObj = new Error(error.message || "Unknown worker error");
    const result: TaskResult = {
      taskId,
      status: "failed",
      error: errorObj,
      metrics: this.updateMetrics(taskId, null, false, error.message),
    };

    activeTask.reject(errorObj);
    this.cleanupTask(taskId, workerId);
    this.workerPool.failedTasks++;

    if (this.onTaskError) {
      this.onTaskError(taskId, errorObj);
    }
  }

  private handleTaskCancelled(taskId: string, workerId: number) {
    const activeTask = this.activeTasks.get(taskId);
    if (!activeTask) return;

    const result: TaskResult = {
      taskId,
      status: "cancelled",
      metrics: this.updateMetrics(taskId, null, false, "Cancelled"),
    };

    activeTask.reject(new Error("Task was cancelled"));
    this.cleanupTask(taskId, workerId);
  }

  private handleStatusUpdate(status: WorkerStatus) {
    if (this.onStatusUpdate) {
      this.onStatusUpdate(status);
    }
  }

  private handleWorkerError(workerId: number, error: ErrorEvent) {
    console.error(`Worker ${workerId} encountered an error:`, error);

    // Restart worker if needed
    if (this.workerPool.workers[workerId]) {
      this.workerPool.workers[workerId].terminate();

      const newWorker = new Worker(
        new URL("../workers/ai-service-worker.ts", import.meta.url),
        { type: "module" },
      );

      this.setupWorkerEventHandlers(newWorker, workerId);
      this.workerPool.workers[workerId] = newWorker;
      this.workerPool.currentLoad[workerId] = 0;
    }
  }

  private cleanupTask(taskId: string, workerId: number) {
    this.activeTasks.delete(taskId);
    if (this.workerPool.currentLoad[workerId] > 0) {
      this.workerPool.currentLoad[workerId]--;
    }
  }

  private updateMetrics(
    taskId: string,
    response: AIResponse | null,
    success: boolean,
    error?: string,
  ): ProcessingMetrics {
    const existing = this.metrics.get(taskId);
    if (!existing) {
      return {
        taskId,
        startTime: Date.now(),
        endTime: Date.now(),
        processingTime: 0,
        queueTime: 0,
        retries: 0,
        provider: "unknown",
        model: "unknown",
        tokensProcessed: response?.tokensUsed || 0,
        success,
        error,
      };
    }

    const updated: ProcessingMetrics = {
      ...existing,
      endTime: Date.now(),
      processingTime: Date.now() - existing.startTime,
      tokensProcessed: response?.tokensUsed || 0,
      success,
      error,
    };

    this.metrics.set(taskId, updated);
    return updated;
  }

  // Public methods for configuration and monitoring
  updateConfiguration(config: Partial<WorkerConfiguration>) {
    this.config = { ...this.config, ...config };

    // Update workers with new config
    this.workerPool.workers.forEach((worker, index) => {
      worker.postMessage({
        type: "UPDATE_PROVIDER_CONFIG",
        taskId: `config-update-${index}`,
        payload: this.config.providers,
      });
    });
  }

  getMetrics(): ProcessingMetrics[] {
    return Array.from(this.metrics.values());
  }

  getWorkerPoolStatus(): WorkerPool {
    return { ...this.workerPool };
  }

  // Helper method to submit multiple tasks in parallel
  async submitBatchTasks(tasks: AITask[]): Promise<string[]> {
    const promises = tasks.map((task) => this.submitTask(task));
    return Promise.all(promises);
  }

  // Helper method to wait for specific task completion
  async waitForTask(taskId: string): Promise<TaskResult> {
    return new Promise((resolve, reject) => {
      const checkInterval = setInterval(() => {
        const metrics = this.metrics.get(taskId);
        if (metrics && metrics.endTime) {
          clearInterval(checkInterval);
          resolve({
            taskId,
            status: metrics.success ? "completed" : "failed",
            metrics,
          });
        }
      }, 100);

      // Timeout after 2x the default timeout
      setTimeout(() => {
        clearInterval(checkInterval);
        reject(new Error(`Timeout waiting for task ${taskId}`));
      }, this.config.defaultTimeout * 2);
    });
  }
}

// Singleton instance for global use
export const aiWorkerManager = new AIWorkerManager();

// Helper functions for common task types
export function createGenerationTask(
  prompt: string,
  model: string,
  providerId: string,
  options: Partial<AITask> = {},
): AITask {
  return {
    taskId: crypto.randomUUID(),
    type: "generate",
    providerId,
    model,
    prompt,
    timestamp: Date.now(),
    priority: "medium",
    ...options,
  };
}

export function createAnalysisTask(
  content: string,
  analysisType: string,
  model: string,
  providerId: string,
  options: Partial<AITask> = {},
): AITask {
  return {
    taskId: crypto.randomUUID(),
    type: "analyze",
    providerId,
    model,
    prompt: `Analyze the following content for ${analysisType}:\n\n${content}`,
    timestamp: Date.now(),
    priority: "medium",
    ...options,
  };
}

export function createEmbeddingTask(
  text: string,
  model: string = "nomic-embed-text",
  providerId: string = "ollama",
  options: Partial<AITask> = {},
): AITask {
  return {
    taskId: crypto.randomUUID(),
    type: "embed",
    providerId,
    model,
    prompt: text,
    timestamp: Date.now(),
    priority: "low",
    ...options,
  };
}
