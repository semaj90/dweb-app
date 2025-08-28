/**
 * GPU Job Queue for Legal AI Platform
 * Advanced communication queue system for optimal GPU resource utilization
 * Prevents GPU overcommit and ensures fair scheduling across legal AI workloads
 */
import { EventEmitter } from 'events';
import { performance } from 'perf_hooks';
import Redis from 'ioredis';
import { legalAIKernelManager } from './gpu-kernel-manager';
import { legalAIGPUManager } from './gpu-memory-manager';

export interface GPUJob {
  id: string;
  type: 'embedding' | 'ner' | 'classification' | 'similarity' | 'summarization' | 'custom';
  kernelName: string;
  payload: any;
  priority: 'low' | 'medium' | 'high' | 'critical';
  userId?: string;
  sessionId?: string;
  timeout: number;
  estimatedMemoryMB: number;
  estimatedComputeTimeMs: number;
  createdAt: number;
  startedAt?: number;
  completedAt?: number;
  retryCount: number;
  maxRetries: number;
}

export interface GPUJobResult {
  jobId: string;
  success: boolean;
  result?: any;
  error?: string;
  executionTime: number;
  memoryUsed: number;
  queueTime: number;
  fromCache: boolean;
}

export interface QueueStats {
  pending: number;
  running: number;
  completed: number;
  failed: number;
  averageQueueTime: number;
  averageExecutionTime: number;
  throughputPerSecond: number;
  memoryUtilization: number;
  queuesByPriority: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  queuesByType: Record<string, number>;
}

export interface SchedulingPolicy {
  maxConcurrentJobs: number;
  priorityWeights: Record<string, number>;
  memoryOvercommitRatio: number; // Allow slight overcommit
  batchingEnabled: boolean;
  fairShareEnabled: boolean;
  preemptionEnabled: boolean;
}

export class GPUJobQueue extends EventEmitter {
  private redis: Redis;
  private pendingJobs: Map<string, GPUJob> = new Map();
  private runningJobs: Map<string, GPUJob> = new Map();
  private completedJobs: Map<string, GPUJobResult> = new Map();
  private userJobCounts: Map<string, number> = new Map(); // Fair sharing
  
  private scheduler: NodeJS.Timeout;
  private statsCollector: NodeJS.Timeout;
  
  private policy: SchedulingPolicy = {
    maxConcurrentJobs: 8, // Based on RTX 3060 Ti capabilities
    priorityWeights: { critical: 10, high: 5, medium: 2, low: 1 },
    memoryOvercommitRatio: 1.1, // Allow 10% overcommit
    batchingEnabled: true,
    fairShareEnabled: true,
    preemptionEnabled: false // Disabled for legal AI stability
  };

  private stats: QueueStats = {
    pending: 0,
    running: 0,
    completed: 0,
    failed: 0,
    averageQueueTime: 0,
    averageExecutionTime: 0,
    throughputPerSecond: 0,
    memoryUtilization: 0,
    queuesByPriority: { critical: 0, high: 0, medium: 0, low: 0 },
    queuesByType: {}
  };

  constructor(redisUrl: string = 'redis://localhost:6379') {
    super();
    this.redis = new Redis(redisUrl);
    
    // Schedule jobs every 50ms for responsive scheduling
    this.scheduler = setInterval(() => this.scheduleJobs(), 50);
    
    // Collect stats every 10 seconds
    this.statsCollector = setInterval(() => this.collectStats(), 10000);
    
    // Listen to GPU events
    this.setupGPUEventHandlers();
  }

  /**
   * Submit job to GPU queue
   */
  async submitJob(job: Omit<GPUJob, 'id' | 'createdAt' | 'retryCount'>): Promise<string> {
    const jobId = `gpu_job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const fullJob: GPUJob = {
      ...job,
      id: jobId,
      createdAt: Date.now(),
      retryCount: 0
    };

    // Validate job
    if (!this.validateJob(fullJob)) {
      throw new Error(`Invalid job parameters for job ${jobId}`);
    }

    // Check queue capacity
    if (this.pendingJobs.size >= 10000) {
      throw new Error('Queue capacity exceeded');
    }

    this.pendingJobs.set(jobId, fullJob);
    
    // Store in Redis for persistence
    await this.redis.hset(`legal_ai:gpu_jobs:${jobId}`, {
      job: JSON.stringify(fullJob),
      status: 'pending',
      created_at: fullJob.createdAt
    });

    // Update user job count for fair sharing
    if (fullJob.userId) {
      const currentCount = this.userJobCounts.get(fullJob.userId) || 0;
      this.userJobCounts.set(fullJob.userId, currentCount + 1);
    }

    this.emit('job:submitted', { jobId, type: fullJob.type, priority: fullJob.priority });
    
    // Immediate scheduling attempt for critical jobs
    if (fullJob.priority === 'critical') {
      setTimeout(() => this.scheduleJobs(), 0);
    }

    return jobId;
  }

  /**
   * Execute job and return result (with Promise interface)
   */
  async executeJob(job: Omit<GPUJob, 'id' | 'createdAt' | 'retryCount'>): Promise<GPUJobResult> {
    const jobId = await this.submitJob(job);
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Job ${jobId} timed out after ${job.timeout}ms`));
      }, job.timeout);

      const resultHandler = (result: GPUJobResult) => {
        if (result.jobId === jobId) {
          clearTimeout(timeout);
          this.off('job:completed', resultHandler);
          this.off('job:failed', resultHandler);
          
          if (result.success) {
            resolve(result);
          } else {
            reject(new Error(result.error || 'Job execution failed'));
          }
        }
      };

      this.on('job:completed', resultHandler);
      this.on('job:failed', resultHandler);
    });
  }

  /**
   * Legal AI specific job submission methods
   */

  // Submit document embedding job
  async embedDocument(text: string, options: {
    priority?: 'low' | 'medium' | 'high' | 'critical';
    userId?: string;
    timeout?: number;
  } = {}): Promise<number[]> {
    const result = await this.executeJob({
      type: 'embedding',
      kernelName: 'legal-embedding',
      payload: { text },
      priority: options.priority || 'medium',
      userId: options.userId,
      timeout: options.timeout || 15000,
      estimatedMemoryMB: 512,
      estimatedComputeTimeMs: 100,
      maxRetries: 2
    });
    
    return result.result;
  }

  // Submit legal entity extraction job
  async extractEntities(text: string, options: {
    priority?: 'low' | 'medium' | 'high' | 'critical';
    userId?: string;
  } = {}): Promise<any[]> {
    const result = await this.executeJob({
      type: 'ner',
      kernelName: 'legal-ner',
      payload: { text },
      priority: options.priority || 'medium',
      userId: options.userId,
      timeout: 20000,
      estimatedMemoryMB: 1024,
      estimatedComputeTimeMs: 300,
      maxRetries: 2
    });
    
    return result.result;
  }

  // Submit vector similarity job
  async computeSimilarity(query: number[], candidates: number[][], options: {
    priority?: 'low' | 'medium' | 'high' | 'critical';
    userId?: string;
  } = {}): Promise<number[]> {
    const result = await this.executeJob({
      type: 'similarity',
      kernelName: 'vector-similarity',
      payload: { query, candidates },
      priority: options.priority || 'high',
      userId: options.userId,
      timeout: 30000,
      estimatedMemoryMB: 2048,
      estimatedComputeTimeMs: 150,
      maxRetries: 1
    });
    
    return result.result;
  }

  // Submit document classification job
  async classifyDocument(text: string, options: {
    priority?: 'low' | 'medium' | 'high' | 'critical';
    userId?: string;
  } = {}): Promise<{ category: string; confidence: number }> {
    const result = await this.executeJob({
      type: 'classification',
      kernelName: 'legal-classifier',
      payload: { text },
      priority: options.priority || 'medium',
      userId: options.userId,
      timeout: 25000,
      estimatedMemoryMB: 768,
      estimatedComputeTimeMs: 200,
      maxRetries: 2
    });
    
    return result.result;
  }

  /**
   * Intelligent job scheduling with priority and resource awareness
   */
  private async scheduleJobs(): Promise<void> {
    try {
      // Check GPU memory availability
      const gpuStats = await legalAIGPUManager.getGPUStats();
      const availableMemoryMB = gpuStats.freeMemoryMB;
      const maxConcurrent = Math.min(
        this.policy.maxConcurrentJobs,
        Math.floor((gpuStats.totalMemoryMB * this.policy.memoryOvercommitRatio) / 512) // Min 512MB per job
      );

      if (this.runningJobs.size >= maxConcurrent) {
        return; // Wait for running jobs to complete
      }

      // Get schedulable jobs
      const schedulableJobs = this.getSchedulableJobs(availableMemoryMB);
      
      if (schedulableJobs.length === 0) {
        return;
      }

      // Schedule jobs up to capacity
      const slotsAvailable = maxConcurrent - this.runningJobs.size;
      const jobsToSchedule = schedulableJobs.slice(0, slotsAvailable);

      for (const job of jobsToSchedule) {
        await this.executeScheduledJob(job);
      }

    } catch (error) {
      this.emit('scheduler:error', {
        error: error instanceof Error ? error.message : 'Unknown scheduling error'
      });
    }
  }

  /**
   * Get jobs ready for scheduling based on priority, memory, and fair share
   */
  private getSchedulableJobs(availableMemoryMB: number): GPUJob[] {
    const pendingJobs = Array.from(this.pendingJobs.values());
    
    // Filter jobs that can fit in available memory
    const fittableJobs = pendingJobs.filter(job => 
      job.estimatedMemoryMB <= availableMemoryMB * this.policy.memoryOvercommitRatio
    );

    // Apply fair sharing if enabled
    let candidateJobs = fittableJobs;
    if (this.policy.fairShareEnabled) {
      candidateJobs = this.applyFairShare(fittableJobs);
    }

    // Sort by priority and creation time
    candidateJobs.sort((a, b) => {
      const priorityDiff = this.policy.priorityWeights[b.priority] - this.policy.priorityWeights[a.priority];
      if (priorityDiff !== 0) return priorityDiff;
      return a.createdAt - b.createdAt; // FIFO within same priority
    });

    return candidateJobs;
  }

  /**
   * Apply fair share scheduling to prevent user monopolization
   */
  private applyFairShare(jobs: GPUJob[]): GPUJob[] {
    if (!this.policy.fairShareEnabled) return jobs;

    const userJobCounts = new Map<string, number>();
    
    // Count running jobs per user
    for (const job of this.runningJobs.values()) {
      if (job.userId) {
        const count = userJobCounts.get(job.userId) || 0;
        userJobCounts.set(job.userId, count + 1);
      }
    }

    // Prioritize users with fewer running jobs
    return jobs.sort((a, b) => {
      const aRunningJobs = userJobCounts.get(a.userId || '') || 0;
      const bRunningJobs = userJobCounts.get(b.userId || '') || 0;
      
      if (aRunningJobs !== bRunningJobs) {
        return aRunningJobs - bRunningJobs;
      }
      
      // Fallback to normal priority sorting
      return this.policy.priorityWeights[b.priority] - this.policy.priorityWeights[a.priority];
    });
  }

  /**
   * Execute scheduled job on GPU kernel
   */
  private async executeScheduledJob(job: GPUJob): Promise<void> {
    const startTime = performance.now();
    
    try {
      // Move from pending to running
      this.pendingJobs.delete(job.id);
      job.startedAt = Date.now();
      this.runningJobs.set(job.id, job);

      // Update Redis status
      await this.redis.hset(`legal_ai:gpu_jobs:${job.id}`, {
        status: 'running',
        started_at: job.startedAt
      });

      this.emit('job:started', { jobId: job.id, type: job.type });

      // Execute on appropriate kernel
      const result = await this.executeOnKernel(job);
      
      const executionTime = performance.now() - startTime;
      const queueTime = job.startedAt - job.createdAt;

      // Create result
      const jobResult: GPUJobResult = {
        jobId: job.id,
        success: true,
        result,
        executionTime,
        memoryUsed: job.estimatedMemoryMB, // Actual would come from GPU monitoring
        queueTime,
        fromCache: false // Would be set by kernel manager if from cache
      };

      await this.completeJob(job, jobResult);

    } catch (error) {
      const executionTime = performance.now() - startTime;
      const queueTime = (job.startedAt || Date.now()) - job.createdAt;

      const jobResult: GPUJobResult = {
        jobId: job.id,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        executionTime,
        memoryUsed: 0,
        queueTime,
        fromCache: false
      };

      await this.handleJobFailure(job, jobResult);
    }
  }

  /**
   * Execute job on appropriate GPU kernel
   */
  private async executeOnKernel(job: GPUJob): Promise<any> {
    switch (job.type) {
      case 'embedding':
        return legalAIKernelManager.embedDocument(job.payload.text);
        
      case 'ner':
        return legalAIKernelManager.extractLegalEntities(job.payload.text);
        
      case 'similarity':
        return legalAIKernelManager.computeVectorSimilarity(
          job.payload.query,
          job.payload.candidates
        );
        
      case 'classification':
        return legalAIKernelManager.classifyLegalDocument(job.payload.text);
        
      case 'summarization':
        return legalAIKernelManager.summarizeLegalDocument(
          job.payload.text,
          job.payload.maxLength
        );
        
      default:
        throw new Error(`Unknown job type: ${job.type}`);
    }
  }

  /**
   * Complete successful job
   */
  private async completeJob(job: GPUJob, result: GPUJobResult): Promise<void> {
    // Move from running to completed
    this.runningJobs.delete(job.id);
    this.completedJobs.set(job.id, result);
    job.completedAt = Date.now();

    // Update user job count
    if (job.userId) {
      const currentCount = this.userJobCounts.get(job.userId) || 0;
      this.userJobCounts.set(job.userId, Math.max(0, currentCount - 1));
    }

    // Update Redis
    await this.redis.hset(`legal_ai:gpu_jobs:${job.id}`, {
      status: 'completed',
      completed_at: job.completedAt,
      result: JSON.stringify(result)
    });

    this.emit('job:completed', result);
  }

  /**
   * Handle job failure with retry logic
   */
  private async handleJobFailure(job: GPUJob, result: GPUJobResult): Promise<void> {
    this.runningJobs.delete(job.id);

    // Retry logic
    if (job.retryCount < job.maxRetries) {
      job.retryCount++;
      job.createdAt = Date.now(); // Reset creation time for fair scheduling
      this.pendingJobs.set(job.id, job);
      
      this.emit('job:retrying', { jobId: job.id, attempt: job.retryCount + 1 });
      return;
    }

    // Final failure
    this.completedJobs.set(job.id, result);
    job.completedAt = Date.now();

    // Update user job count
    if (job.userId) {
      const currentCount = this.userJobCounts.get(job.userId) || 0;
      this.userJobCounts.set(job.userId, Math.max(0, currentCount - 1));
    }

    // Update Redis
    await this.redis.hset(`legal_ai:gpu_jobs:${job.id}`, {
      status: 'failed',
      completed_at: job.completedAt,
      error: result.error || 'Unknown error'
    });

    this.emit('job:failed', result);
  }

  /**
   * Setup GPU event handlers
   */
  private setupGPUEventHandlers(): void {
    legalAIGPUManager.on('gpu:high_memory_usage', () => {
      // Pause scheduling when GPU memory is critically high
      this.policy.maxConcurrentJobs = Math.max(1, Math.floor(this.policy.maxConcurrentJobs * 0.5));
      
      setTimeout(() => {
        this.policy.maxConcurrentJobs = 8; // Reset after 1 minute
      }, 60000);
    });

    legalAIKernelManager.on('batch:processed', (data) => {
      // Increase concurrency after successful batch processing
      if (data.averagePerJob < 100) { // Fast processing
        this.policy.maxConcurrentJobs = Math.min(12, this.policy.maxConcurrentJobs + 1);
      }
    });
  }

  /**
   * Collect queue statistics
   */
  private collectStats(): void {
    this.stats.pending = this.pendingJobs.size;
    this.stats.running = this.runningJobs.size;
    this.stats.completed = this.completedJobs.size;

    // Calculate averages from completed jobs
    const completedResults = Array.from(this.completedJobs.values());
    if (completedResults.length > 0) {
      this.stats.averageQueueTime = completedResults.reduce((sum, r) => sum + r.queueTime, 0) / completedResults.length;
      this.stats.averageExecutionTime = completedResults.reduce((sum, r) => sum + r.executionTime, 0) / completedResults.length;
      this.stats.throughputPerSecond = completedResults.length / ((Date.now() - (completedResults[0]?.queueTime || 0)) / 1000);
    }

    // Count by priority
    this.stats.queuesByPriority = { critical: 0, high: 0, medium: 0, low: 0 };
    for (const job of this.pendingJobs.values()) {
      this.stats.queuesByPriority[job.priority]++;
    }

    // Count by type
    this.stats.queuesByType = {};
    for (const job of this.pendingJobs.values()) {
      this.stats.queuesByType[job.type] = (this.stats.queuesByType[job.type] || 0) + 1;
    }

    this.emit('stats:updated', this.stats);
  }

  /**
   * Validate job parameters
   */
  private validateJob(job: GPUJob): boolean {
    return !!(
      job.type &&
      job.kernelName &&
      job.payload &&
      job.priority &&
      job.timeout > 0 &&
      job.estimatedMemoryMB > 0 &&
      job.maxRetries >= 0
    );
  }

  /**
   * Get queue statistics
   */
  getQueueStats(): QueueStats {
    this.collectStats();
    return { ...this.stats };
  }

  /**
   * Cancel pending job
   */
  async cancelJob(jobId: string): Promise<boolean> {
    const job = this.pendingJobs.get(jobId);
    if (job) {
      this.pendingJobs.delete(jobId);
      await this.redis.hset(`legal_ai:gpu_jobs:${jobId}`, 'status', 'cancelled');
      this.emit('job:cancelled', { jobId });
      return true;
    }
    return false;
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId: string): Promise<{ status: string; job?: GPUJob; result?: GPUJobResult }> {
    if (this.pendingJobs.has(jobId)) {
      return { status: 'pending', job: this.pendingJobs.get(jobId) };
    }
    if (this.runningJobs.has(jobId)) {
      return { status: 'running', job: this.runningJobs.get(jobId) };
    }
    if (this.completedJobs.has(jobId)) {
      return { status: 'completed', result: this.completedJobs.get(jobId) };
    }
    
    // Check Redis for historical jobs
    const redisData = await this.redis.hget(`legal_ai:gpu_jobs:${jobId}`, 'status');
    return { status: redisData || 'not_found' };
  }

  /**
   * Shutdown queue system
   */
  async shutdown(): Promise<void> {
    clearInterval(this.scheduler);
    clearInterval(this.statsCollector);
    
    // Cancel all pending jobs
    for (const jobId of this.pendingJobs.keys()) {
      await this.cancelJob(jobId);
    }
    
    await this.redis.quit();
    this.emit('queue:shutdown');
  }
}

// Global GPU job queue for legal AI platform
export const legalAIGPUQueue = new GPUJobQueue();