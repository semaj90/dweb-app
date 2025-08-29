/**
 * GPU Kernel Manager with Preloading and Memory Persistence
 * Optimizes CUDA kernel loading and keeps models warm for 10-100x performance improvement
 * Integrates with legal AI platform's RTX 3060 Ti GPU memory management
 */
import { EventEmitter } from 'events';
import { spawn, ChildProcess } from 'child_process';
import { performance } from 'perf_hooks';
import { legalAIGPUManager } from './gpu-memory-manager';
import { legalAIResultCache } from './advanced-result-cache';

export interface KernelConfig {
  name: string;
  modelPath: string;
  memoryRequiredMB: number;
  preloadScript: string;
  warmupInputs: any[];
  maxBatchSize: number;
  quantization: 'fp32' | 'fp16' | 'int8' | 'int4';
  deviceId: number;
}

export interface PreloadedKernel {
  name: string;
  processId: number;
  memoryHandle: string;
  isWarm: boolean;
  lastAccessed: number;
  batchQueue: KernelJob[];
  stats: {
    totalRuns: number;
    totalComputeTimeMs: number;
    averageComputeTimeMs: number;
    cacheHitRate: number;
  };
}

export interface KernelJob {
  id: string;
  input: any;
  priority: 'low' | 'medium' | 'high' | 'critical';
  timestamp: number;
  timeout: number;
  resolve: (result: any) => void;
  reject: (error: Error) => void;
}

export interface GPUKernelStats {
  preloadedKernels: number;
  totalMemoryMB: number;
  queuedJobs: number;
  averageLatency: number;
  throughput: number; // jobs per second
  kernels: Record<string, {
    runs: number;
    averageTime: number;
    memoryMB: number;
    isWarm: boolean;
  }>;
}

export class GPUKernelManager extends EventEmitter {
  private kernels: Map<string, PreloadedKernel> = new Map();
  private kernelConfigs: Map<string, KernelConfig> = new Map();
  private jobQueue: Map<string, KernelJob[]> = new Map();
  private processes: Map<string, ChildProcess> = new Map();
  
  private readonly batchProcessor: NodeJS.Timeout;
  private readonly maxQueueSize = 1000;
  private readonly batchInterval = 10; // ms

  constructor() {
    super();
    this.initializeLegalAIKernels();
    
    // Process batched jobs every 10ms for high throughput
    this.batchProcessor = setInterval(() => this.processBatchedJobs(), this.batchInterval);
  }

  /**
   * Initialize legal AI specific GPU kernels
   */
  private initializeLegalAIKernels(): void {
    // Legal document embedding kernel (nomic-embed-text optimized)
    this.kernelConfigs.set('legal-embedding', {
      name: 'legal-embedding',
      modelPath: './models/nomic-embed-text-cuda.bin',
      memoryRequiredMB: 512,
      preloadScript: './python-workers/embedding_kernel.py',
      warmupInputs: [
        'This is a sample legal contract clause for warmup.',
        'Legal entity recognition warmup text.',
        'Contract termination and liability provisions.'
      ],
      maxBatchSize: 32,
      quantization: 'fp16',
      deviceId: 0
    });

    // Legal NER kernel with optimized attention
    this.kernelConfigs.set('legal-ner', {
      name: 'legal-ner',
      modelPath: './models/legal-ner-cuda.bin', 
      memoryRequiredMB: 1024,
      preloadScript: './python-workers/ner_kernel.py',
      warmupInputs: [
        'Plaintiff John Doe vs Defendant ABC Corporation filed in California Superior Court.',
        'The contract shall be governed by the laws of New York State.',
        'Indemnification clause effective upon termination of agreement.'
      ],
      maxBatchSize: 16,
      quantization: 'fp16',
      deviceId: 0
    });

    // Vector similarity kernel (optimized FAISS GPU)
    this.kernelConfigs.set('vector-similarity', {
      name: 'vector-similarity',
      modelPath: './models/faiss-gpu-index.bin',
      memoryRequiredMB: 2048,
      preloadScript: './python-workers/similarity_kernel.py',
      warmupInputs: [
        // Sample 384-dim vectors for warmup
        new Array(384).fill(0).map(() => Math.random()),
        new Array(384).fill(0).map(() => Math.random() * 0.5),
        new Array(384).fill(0).map(() => Math.random() * 2)
      ],
      maxBatchSize: 128, // Large batches for efficiency
      quantization: 'fp32', // Keep precision for similarity
      deviceId: 0
    });

    // Legal document classification kernel
    this.kernelConfigs.set('legal-classifier', {
      name: 'legal-classifier',
      modelPath: './models/legal-bert-classifier-cuda.bin',
      memoryRequiredMB: 768,
      preloadScript: './python-workers/classifier_kernel.py',
      warmupInputs: [
        'This employment agreement contains non-compete clauses.',
        'Real estate purchase agreement with financing contingencies.',
        'Software licensing terms and intellectual property provisions.'
      ],
      maxBatchSize: 24,
      quantization: 'int8', // Quantized for speed
      deviceId: 0
    });

    // Legal summarization kernel (optimized transformer)
    this.kernelConfigs.set('legal-summarizer', {
      name: 'legal-summarizer',
      modelPath: './models/legal-summarizer-cuda.bin',
      memoryRequiredMB: 1536,
      preloadScript: './python-workers/summarizer_kernel.py',
      warmupInputs: [
        `This is a comprehensive legal document containing multiple sections including definitions, 
         obligations, termination clauses, and governing law provisions that requires summarization.`,
        `Employment contract with salary provisions, benefit packages, non-disclosure agreements, 
         and termination conditions for executive-level position.`
      ],
      maxBatchSize: 8, // Smaller batches due to memory requirements
      quantization: 'fp16',
      deviceId: 0
    });

    console.log('🔥 Legal AI GPU kernels configured for preloading');
  }

  /**
   * Preload and warm up GPU kernels
   */
  async preloadKernel(kernelName: string): Promise<any> {
    const config = this.kernelConfigs.get(kernelName);
    if (!config) {
      throw new Error(`Kernel config not found: ${kernelName}`);
    }

    const startTime = performance.now();

    try {
      // Allocate GPU memory
      const memoryHandle = await legalAIGPUManager.allocateGPUMemory(
        `kernel_${kernelName}`,
        kernelName,
        'high'
      );

      // Spawn Python worker process with GPU affinity
      const process = spawn('python', [
        config.preloadScript,
        '--model-path', config.modelPath,
        '--memory-handle', memoryHandle.workerId,
        '--batch-size', config.maxBatchSize.toString(),
        '--quantization', config.quantization,
        '--device-id', config.deviceId.toString(),
        '--preload-mode'
      ], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          CUDA_VISIBLE_DEVICES: config.deviceId.toString(),
          PYTHONUNBUFFERED: '1',
          CUDA_LAUNCH_BLOCKING: '1' // For debugging
        }
      });

      this.processes.set(kernelName, process);

      // Wait for kernel to load and initialize
      await this.waitForKernelReady(process, kernelName);

      // Perform warmup runs to initialize CUDA context and memory
      await this.warmupKernel(kernelName, config.warmupInputs);

      const preloadTime = performance.now() - startTime;

      const kernel: PreloadedKernel = {
        name: kernelName,
        processId: process.pid!,
        memoryHandle: memoryHandle.workerId,
        isWarm: true,
        lastAccessed: Date.now(),
        batchQueue: [],
        stats: {
          totalRuns: config.warmupInputs.length,
          totalComputeTimeMs: preloadTime,
          averageComputeTimeMs: preloadTime / config.warmupInputs.length,
          cacheHitRate: 0
        }
      };

      this.kernels.set(kernelName, kernel);
      this.jobQueue.set(kernelName, []);

      this.emit('kernel:preloaded', {
        kernelName,
        preloadTime,
        memoryMB: config.memoryRequiredMB,
        processId: process.pid
      });

      console.log(`🚀 Kernel ${kernelName} preloaded in ${preloadTime.toFixed(0)}ms`);

    } catch (error: any) {
      this.emit('kernel:preload_failed', {
        kernelName,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      throw error;
    }
  }

  /**
   * Execute computation on preloaded kernel with batching
   */
  async executeKernel<T = any>(
    kernelName: string,
    input: any,
    options: {
      priority?: 'low' | 'medium' | 'high' | 'critical';
      timeout?: number;
      useCache?: boolean;
    } = {}
  ): Promise<T> {
    const kernel = this.kernels.get(kernelName);
    if (!kernel) {
      throw new Error(`Kernel not preloaded: ${kernelName}. Call preloadKernel() first.`);
    }

    // Check result cache first
    if (options.useCache !== false) {
      const cached = await legalAIResultCache.get(`kernel_${kernelName}_${JSON.stringify(input)}`, `gpu_${kernelName}`);
      if (cached) {
        kernel.stats.cacheHitRate = (kernel.stats.cacheHitRate + 1) / 2; // Running average
        return cached.result;
      }
    }

    return new Promise<T>((resolve, reject) => {
      const job: KernelJob = {
        id: `${kernelName}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        input,
        priority: options.priority || 'medium',
        timestamp: Date.now(),
        timeout: options.timeout || 30000,
        resolve: (result) => {
          // Cache successful results
          if (options.useCache !== false) {
            const computeTime = Date.now() - job.timestamp;
            legalAIResultCache.set(
              `kernel_${kernelName}_${JSON.stringify(input)}`,
              `gpu_${kernelName}`,
              result,
              computeTime,
              { fromGPU: true }
            );
          }
          resolve(result);
        },
        reject
      };

      // Add to batch queue
      const queue = this.jobQueue.get(kernelName)!;
      if (queue.length >= this.maxQueueSize) {
        reject(new Error(`Queue full for kernel ${kernelName}`));
        return;
      }

      queue.push(job);

      // Set timeout
      setTimeout(() => {
        const index = queue.indexOf(job);
        if (index > -1) {
          queue.splice(index, 1);
          reject(new Error(`Kernel execution timeout: ${kernelName}`));
        }
      }, job.timeout);
    });
  }

  /**
   * Process batched jobs for optimal GPU utilization
   */
  private async processBatchedJobs(): Promise<any> {
    for (const [kernelName, queue] of this.jobQueue) {
      if (queue.length === 0) continue;

      const config = this.kernelConfigs.get(kernelName)!;
      const kernel = this.kernels.get(kernelName)!;
      const process = this.processes.get(kernelName)!;

      // Sort by priority and take batch
      queue.sort((a, b) => {
        const priorityOrder = { 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      });

      const batch = queue.splice(0, Math.min(config.maxBatchSize, queue.length));
      if (batch.length === 0) continue;

      try {
        const startTime = performance.now();
        
        // Execute batch on GPU kernel
        const results = await this.executeBatch(process, kernelName, batch);
        
        const computeTime = performance.now() - startTime;

        // Resolve individual jobs
        batch.forEach((job, index) => {
          job.resolve(results[index]);
        });

        // Update kernel stats
        kernel.stats.totalRuns += batch.length;
        kernel.stats.totalComputeTimeMs += computeTime;
        kernel.stats.averageComputeTimeMs = kernel.stats.totalComputeTimeMs / kernel.stats.totalRuns;
        kernel.lastAccessed = Date.now();

        this.emit('batch:processed', {
          kernelName,
          batchSize: batch.length,
          computeTime,
          averagePerJob: computeTime / batch.length
        });

      } catch (error: any) {
        // Reject all jobs in failed batch
        batch.forEach(job => {
          job.reject(error instanceof Error ? error : new Error('Batch execution failed'));
        });

        this.emit('batch:failed', {
          kernelName,
          batchSize: batch.length,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }
  }

  /**
   * Execute batch of jobs on GPU kernel process
   */
  private async executeBatch(process: ChildProcess, kernelName: string, batch: KernelJob[]): Promise<any[]> {
    return new Promise((resolve, reject) => {
      const batchId = `batch_${Date.now()}`;
      const inputs = batch.map(job => job.input);

      let responseBuffer = '';
      const timeout = setTimeout(() => {
        reject(new Error(`Batch execution timeout for ${kernelName}`));
      }, Math.max(...batch.map(job => job.timeout)));

      const dataHandler = (data: Buffer) => {
        responseBuffer += data.toString();
        
        // Look for complete batch response
        const lines = responseBuffer.split('\n');
        for (const line of lines) {
          if (line.trim().startsWith('{')) {
            try {
              const response = JSON.parse(line.trim());
              if (response.batchId === batchId && response.results) {
                clearTimeout(timeout);
                process.stdout!.off('data', dataHandler);
                resolve(response.results);
                return;
              }
            } catch (parseError) {
              // Continue parsing
            }
          }
        }
      };

      process.stdout!.on('data', dataHandler);

      // Send batch request
      const batchRequest = {
        batchId,
        inputs,
        kernelName
      };

      process.stdin!.write(JSON.stringify(batchRequest) + '\n');
    });
  }

  /**
   * Wait for kernel process to be ready
   */
  private async waitForKernelReady(process: ChildProcess, kernelName: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Kernel ${kernelName} failed to initialize within 30 seconds`));
      }, 30000);

      let outputBuffer = '';
      const dataHandler = (data: Buffer) => {
        outputBuffer += data.toString();
        if (outputBuffer.includes('KERNEL_READY')) {
          clearTimeout(timeout);
          process.stdout!.off('data', dataHandler);
          resolve();
        }
      };

      process.stdout!.on('data', dataHandler);
      
      process.stderr!.on('data', (data) => {
        const errorText = data.toString();
        if (errorText.includes('ERROR') || errorText.includes('CUDA_ERROR')) {
          clearTimeout(timeout);
          reject(new Error(`Kernel initialization error: ${errorText}`));
        }
      });
    });
  }

  /**
   * Warm up kernel with sample inputs
   */
  private async warmupKernel(kernelName: string, warmupInputs: any[]): Promise<any> {
    const startTime = performance.now();
    
    // Execute warmup computations
    const warmupPromises = warmupInputs.map((input, index) =>
      this.executeKernel(kernelName, input, { 
        priority: 'low',
        useCache: false,
        timeout: 10000 
      }).catch(error => {
        console.warn(`Warmup ${index} failed for ${kernelName}:`, error.message);
        return null;
      })
    );

    await Promise.all(warmupPromises);
    
    const warmupTime = performance.now() - startTime;
    this.emit('kernel:warmed', { kernelName, warmupTime, inputs: warmupInputs.length });
  }

  /**
   * Preload all legal AI kernels for optimal performance
   */
  async preloadAllKernels(): Promise<any> {
    const kernelNames = Array.from(this.kernelConfigs.keys());
    
    console.log(`🔥 Preloading ${kernelNames.length} legal AI GPU kernels...`);
    
    // Preload in order of memory usage (largest first)
    const sortedKernels = kernelNames.sort((a, b) => {
      const configA = this.kernelConfigs.get(a)!;
      const configB = this.kernelConfigs.get(b)!;
      return configB.memoryRequiredMB - configA.memoryRequiredMB;
    });

    for (const kernelName of sortedKernels) {
      try {
        await this.preloadKernel(kernelName);
      } catch (error: any) {
        console.error(`Failed to preload kernel ${kernelName}:`, error);
      }
    }

    this.emit('kernels:all_preloaded', { 
      total: kernelNames.length,
      successful: this.kernels.size 
    });
  }

  /**
   * Get GPU kernel statistics
   */
  getKernelStats(): GPUKernelStats {
    const kernelStats: Record<string, any> = {};
    let totalJobs = 0;
    let totalLatency = 0;

    for (const [name, kernel] of this.kernels) {
      kernelStats[name] = {
        runs: kernel.stats.totalRuns,
        averageTime: kernel.stats.averageComputeTimeMs,
        memoryMB: this.kernelConfigs.get(name)?.memoryRequiredMB || 0,
        isWarm: kernel.isWarm
      };
      totalJobs += kernel.stats.totalRuns;
      totalLatency += kernel.stats.averageComputeTimeMs;
    }

    const queuedJobs = Array.from(this.jobQueue.values())
      .reduce((sum, queue) => sum + queue.length, 0);

    return {
      preloadedKernels: this.kernels.size,
      totalMemoryMB: Array.from(this.kernelConfigs.values())
        .reduce((sum, config) => sum + config.memoryRequiredMB, 0),
      queuedJobs,
      averageLatency: totalJobs > 0 ? totalLatency / totalJobs : 0,
      throughput: totalJobs > 0 ? 1000 / (totalLatency / totalJobs) : 0,
      kernels: kernelStats
    };
  }

  /**
   * Legal AI specific kernel execution methods
   */
  
  // Fast document embedding with preloaded kernel
  async embedDocument(text: string): Promise<number[]> {
    return this.executeKernel('legal-embedding', { text }, { useCache: true });
  }

  // Fast legal entity recognition
  async extractLegalEntities(text: string): Promise<any[]> {
    return this.executeKernel('legal-ner', { text }, { useCache: true });
  }

  // Optimized vector similarity search
  async computeVectorSimilarity(query: number[], candidates: number[][]): Promise<number[]> {
    return this.executeKernel('vector-similarity', { query, candidates }, { useCache: true });
  }

  // Legal document classification
  async classifyLegalDocument(text: string): Promise<{ category: string; confidence: number }> {
    return this.executeKernel('legal-classifier', { text }, { useCache: true });
  }

  // Legal document summarization
  async summarizeLegalDocument(text: string, maxLength: number = 200): Promise<string> {
    return this.executeKernel('legal-summarizer', { text, maxLength }, { useCache: true });
  }

  /**
   * Shutdown all kernels and cleanup
   */
  async shutdown(): Promise<any> {
    clearInterval(this.batchProcessor);

    // Terminate all processes
    for (const [kernelName, process] of this.processes) {
      try {
        process.kill('SIGTERM');
        await legalAIGPUManager.releaseGPUMemory(`kernel_${kernelName}`);
      } catch (error: any) {
        console.warn(`Error shutting down kernel ${kernelName}:`, error);
      }
    }

    this.kernels.clear();
    this.processes.clear();
    this.jobQueue.clear();

    this.emit('kernels:shutdown');
  }
}

// Global GPU kernel manager for legal AI platform
export const legalAIKernelManager = new GPUKernelManager();

// Auto-preload kernels on startup for immediate availability
legalAIKernelManager.preloadAllKernels().catch(console.error);