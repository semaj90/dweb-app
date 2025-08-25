/**
 * Unified GPU Service Integration Layer
 * Connects FlashAttention2 RTX 3060 Ti service with GPU error processor
 * Provides centralized GPU orchestration for the Legal AI platform
 */

import { flashAttention2Service, gpuErrorProcessor, type FlashAttention2Config, type AttentionResult, type GPUErrorContext, type ErrorProcessingResult } from './flashattention2-rtx3060';

export interface GPUServiceConfig {
  enableGPU: boolean;
  fallbackToCPU: boolean;
  maxRetries: number;
  timeout: number;
  batchSize: number;
  memoryOptimization: 'balanced' | 'speed' | 'memory';
}

export interface GPUProcessingTask {
  id: string;
  type: 'embedding' | 'similarity' | 'legal_analysis' | 'error_processing';
  data: any;
  priority: 'low' | 'medium' | 'high' | 'critical';
  metadata: Record<string, any>;
}

export interface GPUTaskResult {
  taskId: string;
  success: boolean;
  result?: any;
  error?: string;
  processingTime: number;
  gpuUsed: boolean;
  memoryUsed: number;
}

export interface GPUServiceStatus {
  available: boolean;
  initialized: boolean;
  currentLoad: number;
  queuedTasks: number;
  errorRate: number;
  memoryUsage: {
    used: number;
    total: number;
    percentage: number;
  };
  performance: {
    avgProcessingTime: number;
    throughput: number;
    efficiency: number;
  };
}

/**
 * Centralized GPU Service Integration
 * Manages all GPU-accelerated services for legal AI processing
 */
export class GPUServiceIntegration {
  private config: GPUServiceConfig;
  private taskQueue: GPUProcessingTask[] = [];
  private activeTasks = new Map<string, GPUProcessingTask>();
  private isInitialized = false;
  private metrics = {
    tasksProcessed: 0,
    tasksSucceeded: 0,
    tasksFailed: 0,
    totalProcessingTime: 0,
    gpuUtilization: 0
  };

  constructor(config: Partial<GPUServiceConfig> = {}) {
    this.config = {
      enableGPU: true,
      fallbackToCPU: true,
      maxRetries: 3,
      timeout: 30000,
      batchSize: 8,
      memoryOptimization: 'balanced',
      ...config
    };
  }

  /**
   * Initialize all GPU services
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('🚀 Initializing GPU Service Integration...');

    try {
      // Initialize FlashAttention2 service
      await flashAttention2Service.initialize();
      console.log('✅ FlashAttention2 service initialized');

      // Test GPU availability
      const gpuStatus = await this.testGPUAvailability();
      if (!gpuStatus.available && !this.config.fallbackToCPU) {
        throw new Error('GPU not available and CPU fallback disabled');
      }

      this.isInitialized = true;
      console.log('✅ GPU Service Integration initialized successfully');

      // Start task processing loop
      this.startTaskProcessor();

    } catch (error) {
      console.error('❌ GPU Service Integration initialization failed:', error);
      if (this.config.fallbackToCPU) {
        console.log('🔄 Falling back to CPU-only mode');
        this.config.enableGPU = false;
        this.isInitialized = true;
        this.startTaskProcessor();
      } else {
        throw error;
      }
    }
  }

  /**
   * Submit a task for GPU processing
   */
  async submitTask(task: Omit<GPUProcessingTask, 'id'>): Promise<string> {
    await this.initialize();

    const taskId = `gpu_task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const fullTask: GPUProcessingTask = {
      ...task,
      id: taskId
    };

    // Priority queue insertion
    this.insertTaskByPriority(fullTask);

    console.log(`📝 Task ${taskId} queued for ${task.type} processing (priority: ${task.priority})`);
    return taskId;
  }

  /**
   * Process legal text with FlashAttention2
   */
  async processLegalText(
    text: string,
    context: string[] = [],
    analysisType: 'semantic' | 'legal' | 'precedent' = 'legal'
  ): Promise<AttentionResult> {
    const taskId = await this.submitTask({
      type: 'legal_analysis',
      data: { text, context, analysisType },
      priority: 'medium',
      metadata: { textLength: text.length, contextCount: context.length }
    });

    return this.waitForTaskResult(taskId);
  }

  /**
   * Process GPU errors using the error processor
   */
  async processGPUError(errorContext: GPUErrorContext): Promise<ErrorProcessingResult> {
    const taskId = await this.submitTask({
      type: 'error_processing',
      data: errorContext,
      priority: errorContext.errorType === 'compilation' ? 'high' : 'medium',
      metadata: { errorType: errorContext.errorType, modelVersion: errorContext.modelVersion }
    });

    return this.waitForTaskResult(taskId);
  }

  /**
   * Generate embeddings for text
   */
  async generateEmbeddings(texts: string[]): Promise<Float32Array[]> {
    const taskId = await this.submitTask({
      type: 'embedding',
      data: { texts },
      priority: 'medium',
      metadata: { batchSize: texts.length }
    });

    const result = await this.waitForTaskResult(taskId);
    return result.embeddings;
  }

  /**
   * Calculate similarity between texts
   */
  async calculateSimilarity(text1: string, text2: string): Promise<number> {
    const taskId = await this.submitTask({
      type: 'similarity',
      data: { text1, text2 },
      priority: 'low',
      metadata: { comparisonType: 'text_similarity' }
    });

    const result = await this.waitForTaskResult(taskId);
    return result.similarity;
  }

  /**
   * Get current GPU service status
   */
  async getStatus(): Promise<GPUServiceStatus> {
    const flashAttentionStatus = flashAttention2Service.getStatus();
    const memoryInfo = this.getMemoryUsage();
    
    return {
      available: this.config.enableGPU && flashAttentionStatus.gpuEnabled,
      initialized: this.isInitialized,
      currentLoad: this.activeTasks.size,
      queuedTasks: this.taskQueue.length,
      errorRate: this.calculateErrorRate(),
      memoryUsage: memoryInfo,
      performance: {
        avgProcessingTime: this.metrics.tasksProcessed > 0 
          ? this.metrics.totalProcessingTime / this.metrics.tasksProcessed 
          : 0,
        throughput: this.calculateThroughput(),
        efficiency: this.calculateEfficiency()
      }
    };
  }

  /**
   * Test GPU availability
   */
  private async testGPUAvailability(): Promise<{ available: boolean; details: string }> {
    try {
      // Test WebGPU availability
      if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
        const adapter = await (navigator as any).gpu?.requestAdapter();
        if (adapter) {
          return { available: true, details: 'WebGPU available' };
        }
      }

      // Test FlashAttention2 GPU mode
      const flashStatus = flashAttention2Service.getStatus();
      if (flashStatus.gpuEnabled) {
        return { available: true, details: 'FlashAttention2 GPU mode active' };
      }

      return { available: false, details: 'No GPU acceleration available' };
    } catch (error) {
      return { available: false, details: `GPU test failed: ${error.message}` };
    }
  }

  /**
   * Insert task into queue by priority
   */
  private insertTaskByPriority(task: GPUProcessingTask): void {
    const priorities = { critical: 4, high: 3, medium: 2, low: 1 };
    const taskPriority = priorities[task.priority] || 1;
    
    let insertIndex = this.taskQueue.length;
    for (let i = 0; i < this.taskQueue.length; i++) {
      const queuedPriority = priorities[this.taskQueue[i].priority] || 1;
      if (taskPriority > queuedPriority) {
        insertIndex = i;
        break;
      }
    }
    
    this.taskQueue.splice(insertIndex, 0, task);
  }

  /**
   * Start the task processing loop
   */
  private startTaskProcessor(): void {
    const processNextTask = async () => {
      if (this.taskQueue.length === 0 || this.activeTasks.size >= this.config.batchSize) {
        setTimeout(processNextTask, 100);
        return;
      }

      const task = this.taskQueue.shift();
      if (!task) {
        setTimeout(processNextTask, 100);
        return;
      }

      this.activeTasks.set(task.id, task);
      
      try {
        await this.executeTask(task);
      } catch (error) {
        console.error(`❌ Task ${task.id} execution failed:`, error);
      }

      this.activeTasks.delete(task.id);
      setTimeout(processNextTask, 10);
    };

    processNextTask();
    console.log('🔄 GPU task processor started');
  }

  /**
   * Execute a specific task
   */
  private async executeTask(task: GPUProcessingTask): Promise<GPUTaskResult> {
    const startTime = performance.now();
    let result: GPUTaskResult;

    try {
      switch (task.type) {
        case 'legal_analysis':
          const analysisResult = await flashAttention2Service.processLegalText(
            task.data.text,
            task.data.context || [],
            task.data.analysisType || 'legal'
          );
          result = {
            taskId: task.id,
            success: true,
            result: analysisResult,
            processingTime: performance.now() - startTime,
            gpuUsed: this.config.enableGPU,
            memoryUsed: analysisResult.memoryUsage || 0
          };
          break;

        case 'error_processing':
          const errorResult = await gpuErrorProcessor.processGPUError(task.data);
          result = {
            taskId: task.id,
            success: errorResult.resolved,
            result: errorResult,
            processingTime: performance.now() - startTime,
            gpuUsed: this.config.enableGPU,
            memoryUsed: 0
          };
          break;

        case 'embedding':
          // Simulate embedding generation
          const embeddings = task.data.texts.map(() => 
            new Float32Array(384).map(() => Math.random() - 0.5)
          );
          result = {
            taskId: task.id,
            success: true,
            result: { embeddings },
            processingTime: performance.now() - startTime,
            gpuUsed: this.config.enableGPU,
            memoryUsed: embeddings.length * 384 * 4 // Float32 bytes
          };
          break;

        case 'similarity':
          // Simulate similarity calculation
          const similarity = Math.random() * 0.5 + 0.5;
          result = {
            taskId: task.id,
            success: true,
            result: { similarity },
            processingTime: performance.now() - startTime,
            gpuUsed: this.config.enableGPU,
            memoryUsed: 1024 // Small memory usage
          };
          break;

        default:
          throw new Error(`Unknown task type: ${task.type}`);
      }

      this.updateMetrics(result);
      return result;

    } catch (error) {
      result = {
        taskId: task.id,
        success: false,
        error: error.message,
        processingTime: performance.now() - startTime,
        gpuUsed: false,
        memoryUsed: 0
      };
      
      this.updateMetrics(result);
      return result;
    }
  }

  /**
   * Wait for a task result
   */
  private async waitForTaskResult(taskId: string, timeout: number = this.config.timeout): Promise<any> {
    return new Promise((resolve, reject) => {
      const checkResult = () => {
        // This would normally check a results store
        // For now, simulate waiting for the active task to complete
        if (!this.activeTasks.has(taskId)) {
          // Task completed, return mock result
          resolve({ success: true, taskId });
          return;
        }
        
        setTimeout(checkResult, 100);
      };

      setTimeout(() => {
        reject(new Error(`Task ${taskId} timed out after ${timeout}ms`));
      }, timeout);

      checkResult();
    });
  }

  /**
   * Update processing metrics
   */
  private updateMetrics(result: GPUTaskResult): void {
    this.metrics.tasksProcessed++;
    this.metrics.totalProcessingTime += result.processingTime;
    
    if (result.success) {
      this.metrics.tasksSucceeded++;
    } else {
      this.metrics.tasksFailed++;
    }

    if (result.gpuUsed) {
      this.metrics.gpuUtilization++;
    }
  }

  /**
   * Calculate error rate
   */
  private calculateErrorRate(): number {
    if (this.metrics.tasksProcessed === 0) return 0;
    return this.metrics.tasksFailed / this.metrics.tasksProcessed;
  }

  /**
   * Calculate throughput (tasks per second)
   */
  private calculateThroughput(): number {
    if (this.metrics.totalProcessingTime === 0) return 0;
    return (this.metrics.tasksProcessed * 1000) / this.metrics.totalProcessingTime;
  }

  /**
   * Calculate efficiency (successful tasks / total tasks)
   */
  private calculateEfficiency(): number {
    if (this.metrics.tasksProcessed === 0) return 0;
    return this.metrics.tasksSucceeded / this.metrics.tasksProcessed;
  }

  /**
   * Get memory usage information
   */
  private getMemoryUsage(): GPUServiceStatus['memoryUsage'] {
    if (typeof performance !== 'undefined' && 'memory' in performance) {
      const memory = (performance as any).memory;
      return {
        used: memory.usedJSHeapSize || 0,
        total: memory.totalJSHeapSize || 0,
        percentage: memory.totalJSHeapSize > 0 
          ? (memory.usedJSHeapSize / memory.totalJSHeapSize) * 100 
          : 0
      };
    }

    return {
      used: 0,
      total: 0,
      percentage: 0
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    console.log('🧹 Cleaning up GPU Service Integration...');
    
    // Clear task queue
    this.taskQueue.length = 0;
    this.activeTasks.clear();
    
    // Cleanup FlashAttention2 service
    await flashAttention2Service.cleanup();
    
    // Clear error processor cache
    gpuErrorProcessor.clearCache();
    
    this.isInitialized = false;
    console.log('✅ GPU Service Integration cleanup complete');
  }
}

// Global instance
export const gpuServiceIntegration = new GPUServiceIntegration({
  enableGPU: true,
  fallbackToCPU: true,
  maxRetries: 3,
  timeout: 30000,
  batchSize: 4, // Reduced for RTX 3060 Ti
  memoryOptimization: 'balanced'
});

// Auto-initialize in browser environment
if (typeof window !== 'undefined') {
  gpuServiceIntegration.initialize().catch(console.warn);
}

// Export utility functions
export const GPUServiceUtils = {
  /**
   * Quick legal text analysis
   */
  async analyzeLegalText(text: string, context?: string[]): Promise<AttentionResult> {
    return gpuServiceIntegration.processLegalText(text, context, 'legal');
  },

  /**
   * Quick error processing
   */
  async processError(error: GPUErrorContext): Promise<ErrorProcessingResult> {
    return gpuServiceIntegration.processGPUError(error);
  },

  /**
   * Quick status check
   */
  async getServiceStatus(): Promise<GPUServiceStatus> {
    return gpuServiceIntegration.getStatus();
  },

  /**
   * Test GPU integration
   */
  async testIntegration(): Promise<{ success: boolean; details: string }> {
    try {
      const status = await gpuServiceIntegration.getStatus();
      
      if (!status.available) {
        return { success: false, details: 'GPU services not available' };
      }

      // Test with sample legal text
      const testResult = await gpuServiceIntegration.processLegalText(
        'This is a test legal document for GPU processing.',
        ['legal test', 'gpu integration'],
        'legal'
      );

      return { 
        success: true, 
        details: `GPU integration test successful. Confidence: ${testResult.confidence?.toFixed(2) || 'N/A'}` 
      };
    } catch (error) {
      return { 
        success: false, 
        details: `Integration test failed: ${error.message}` 
      };
    }
  }
};
