/**
 * GPU Memory Manager for CUDA Worker Persistence
 * Optimizes GPU memory allocation and keeps models warm for legal AI processing
 * Integrates with RTX 3060 Ti (8GB VRAM) and legal AI platform architecture
 */
import { EventEmitter } from 'events';
import { spawn, ChildProcess } from 'child_process';
import { performance } from 'perf_hooks';

export interface GPUMemoryAllocation {
  workerId: string;
  processId: number;
  memoryMB: number;
  reservedMemoryMB: number;
  modelName: string;
  lastAccessed: number;
  isWarm: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

export interface GPUStats {
  totalMemoryMB: number;
  usedMemoryMB: number;
  freeMemoryMB: number;
  temperature: number;
  utilization: number;
  allocations: GPUMemoryAllocation[];
}

export interface ModelConfig {
  name: string;
  memoryRequiredMB: number;
  warmupTimeMs: number;
  keepWarmMs: number;
  maxInstances: number;
}

export class GPUMemoryManager extends EventEmitter {
  private allocations: Map<string, GPUMemoryAllocation> = new Map();
  private models: Map<string, ModelConfig> = new Map();
  private totalGPUMemoryMB: number = 8192; // RTX 3060 Ti
  private reservedSystemMemoryMB: number = 1024; // Reserve for system
  private monitoringProcess?: ChildProcess;
  private cleanupInterval: NodeJS.Timeout;

  constructor() {
    super();
    this.initializeLegalAIModels();
    this.startGPUMonitoring();
    
    // Cleanup stale allocations every 30 seconds
    this.cleanupInterval = setInterval(() => this.cleanupStaleAllocations(), 30000);
  }

  /**
   * Initialize legal AI model configurations
   */
  private initializeLegalAIModels(): void {
    // Legal document embedding model (nomic-embed-text)
    this.models.set('nomic-embed-text', {
      name: 'nomic-embed-text',
      memoryRequiredMB: 512,
      warmupTimeMs: 3000,
      keepWarmMs: 600000, // Keep warm for 10 minutes
      maxInstances: 2
    });

    // Legal entity recognition model
    this.models.set('legal-ner', {
      name: 'legal-ner',
      memoryRequiredMB: 1024,
      warmupTimeMs: 5000,
      keepWarmMs: 300000, // Keep warm for 5 minutes
      maxInstances: 1
    });

    // Legal document classification model
    this.models.set('legal-classifier', {
      name: 'legal-classifier',
      memoryRequiredMB: 768,
      warmupTimeMs: 4000,
      keepWarmMs: 400000, // Keep warm for 6.7 minutes
      maxInstances: 1
    });

    // Vector similarity analysis (for case clustering)
    this.models.set('vector-similarity', {
      name: 'vector-similarity',
      memoryRequiredMB: 2048,
      warmupTimeMs: 7000,
      keepWarmMs: 900000, // Keep warm for 15 minutes (expensive to reload)
      maxInstances: 1
    });

    // Legal summarization model
    this.models.set('legal-summarizer', {
      name: 'legal-summarizer',
      memoryRequiredMB: 1536,
      warmupTimeMs: 6000,
      keepWarmMs: 450000, // Keep warm for 7.5 minutes
      maxInstances: 1
    });

    console.log('🧠 Legal AI GPU models configured');
  }

  /**
   * Allocate GPU memory for a worker with specific model
   */
  async allocateGPUMemory(
    workerId: string,
    modelName: string,
    priority: 'low' | 'medium' | 'high' | 'critical' = 'medium'
  ): Promise<GPUMemoryAllocation> {
    const model = this.models.get(modelName);
    if (!model) {
      throw new Error(`Unknown model: ${modelName}`);
    }

    // Check if we have existing warm allocation for this model
    const existingWarmAllocation = Array.from(this.allocations.values()).find(
      alloc => alloc.modelName === modelName && alloc.isWarm
    );

    if (existingWarmAllocation) {
      // Reuse existing warm model
      existingWarmAllocation.workerId = workerId;
      existingWarmAllocation.lastAccessed = Date.now();
      existingWarmAllocation.priority = priority;
      
      this.emit('memory:reused', { workerId, modelName, memoryMB: existingWarmAllocation.memoryMB });
      return existingWarmAllocation;
    }

    // Check available memory
    const availableMemory = await this.getAvailableMemory();
    const requiredMemory = model.memoryRequiredMB + 256; // Add buffer

    if (availableMemory < requiredMemory) {
      // Try to free memory by evicting low-priority allocations
      await this.evictLowPriorityAllocations(requiredMemory);
      
      const availableAfterEviction = await this.getAvailableMemory();
      if (availableAfterEviction < requiredMemory) {
        throw new Error(
          `Insufficient GPU memory. Required: ${requiredMemory}MB, Available: ${availableAfterEviction}MB`
        );
      }
    }

    // Create new allocation
    const allocation: GPUMemoryAllocation = {
      workerId,
      processId: process.pid, // Will be updated when worker starts
      memoryMB: model.memoryRequiredMB,
      reservedMemoryMB: requiredMemory,
      modelName,
      lastAccessed: Date.now(),
      isWarm: false,
      priority
    };

    this.allocations.set(workerId, allocation);
    
    // Start model warmup
    await this.warmupModel(allocation);
    
    this.emit('memory:allocated', { 
      workerId, 
      modelName, 
      memoryMB: allocation.memoryMB,
      warmupTime: model.warmupTimeMs 
    });

    return allocation;
  }

  /**
   * Warm up model in GPU memory
   */
  private async warmupModel(allocation: GPUMemoryAllocation): Promise<any> {
    const model = this.models.get(allocation.modelName)!;
    const startTime = performance.now();

    try {
      // Simulate model loading - in practice, this would load the actual model
      await new Promise(resolve => setTimeout(resolve, model.warmupTimeMs));
      
      allocation.isWarm = true;
      allocation.lastAccessed = Date.now();
      
      const warmupTime = performance.now() - startTime;
      this.emit('model:warmed', { 
        workerId: allocation.workerId, 
        modelName: allocation.modelName,
        warmupTime 
      });

      // Schedule automatic cooldown
      setTimeout(() => {
        if (allocation.isWarm && (Date.now() - allocation.lastAccessed) >= model.keepWarmMs) {
          this.cooldownModel(allocation.workerId);
        }
      }, model.keepWarmMs);
      
    } catch (error: any) {
      this.emit('model:warmup_failed', { 
        workerId: allocation.workerId, 
        modelName: allocation.modelName,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      throw error;
    }
  }

  /**
   * Cool down model and free GPU memory
   */
  async cooldownModel(workerId: string): Promise<any> {
    const allocation = this.allocations.get(workerId);
    if (!allocation) return;

    allocation.isWarm = false;
    
    // In practice, this would unload the model from GPU
    // For now, we just mark it as not warm
    
    this.emit('model:cooled', { 
      workerId, 
      modelName: allocation.modelName,
      memoryFreed: allocation.memoryMB 
    });
  }

  /**
   * Release GPU memory allocation
   */
  async releaseGPUMemory(workerId: string): Promise<any> {
    const allocation = this.allocations.get(workerId);
    if (!allocation) return;

    // For warm models, don't immediately release - keep for reuse
    const model = this.models.get(allocation.modelName)!;
    if (allocation.isWarm && (Date.now() - allocation.lastAccessed) < model.keepWarmMs) {
      // Keep warm for potential reuse, but clear worker assignment
      allocation.workerId = `warm_${allocation.modelName}_${Date.now()}`;
      return;
    }

    // Release the allocation
    await this.cooldownModel(workerId);
    this.allocations.delete(workerId);
    
    this.emit('memory:released', { 
      workerId, 
      modelName: allocation.modelName,
      memoryFreed: allocation.memoryMB 
    });
  }

  /**
   * Get current GPU statistics
   */
  async getGPUStats(): Promise<GPUStats> {
    const allocations = Array.from(this.allocations.values());
    const usedMemoryMB = allocations.reduce((sum, alloc) => sum + alloc.reservedMemoryMB, 0);
    const freeMemoryMB = this.totalGPUMemoryMB - this.reservedSystemMemoryMB - usedMemoryMB;

    // Get GPU utilization and temperature (simulated for now)
    const { utilization, temperature } = await this.queryGPUMetrics();

    return {
      totalMemoryMB: this.totalGPUMemoryMB,
      usedMemoryMB,
      freeMemoryMB: Math.max(0, freeMemoryMB),
      temperature,
      utilization,
      allocations: allocations.slice() // Return copy
    };
  }

  /**
   * Get available GPU memory
   */
  private async getAvailableMemory(): Promise<number> {
    const stats = await this.getGPUStats();
    return stats.freeMemoryMB;
  }

  /**
   * Evict low-priority allocations to free memory
   */
  private async evictLowPriorityAllocations(requiredMemory: number): Promise<any> {
    const allocations = Array.from(this.allocations.values());
    
    // Sort by priority and last accessed time
    const candidates = allocations
      .filter(alloc => alloc.priority !== 'critical')
      .sort((a, b) => {
        const priorityOrder = { 'low': 0, 'medium': 1, 'high': 2, 'critical': 3 };
        if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
          return priorityOrder[a.priority] - priorityOrder[b.priority];
        }
        return a.lastAccessed - b.lastAccessed; // Older first
      });

    let freedMemory = 0;
    const evicted: string[] = [];

    for (const candidate of candidates) {
      if (freedMemory >= requiredMemory) break;

      await this.releaseGPUMemory(candidate.workerId);
      freedMemory += candidate.reservedMemoryMB;
      evicted.push(candidate.workerId);
    }

    if (evicted.length > 0) {
      this.emit('memory:evicted', { 
        evictedWorkers: evicted, 
        freedMemoryMB: freedMemory 
      });
    }
  }

  /**
   * Clean up stale allocations
   */
  private async cleanupStaleAllocations(): Promise<any> {
    const now = Date.now();
    const staleAllocations: string[] = [];

    for (const [workerId, allocation] of this.allocations) {
      const model = this.models.get(allocation.modelName)!;
      const staleDuration = now - allocation.lastAccessed;

      // Clean up allocations that haven't been used for twice the keep-warm duration
      if (staleDuration > (model.keepWarmMs * 2)) {
        staleAllocations.push(workerId);
      }
    }

    for (const workerId of staleAllocations) {
      await this.releaseGPUMemory(workerId);
    }

    if (staleAllocations.length > 0) {
      this.emit('memory:cleanup', { 
        cleanedAllocations: staleAllocations.length 
      });
    }
  }

  /**
   * Query GPU metrics (utilization, temperature, etc.)
   */
  private async queryGPUMetrics(): Promise<{ utilization: number; temperature: number }> {
    // In practice, this would use nvidia-ml-py or nvidia-smi
    // For now, return simulated values
    const allocations = Array.from(this.allocations.values());
    const warmAllocations = allocations.filter(a => a.isWarm).length;
    
    return {
      utilization: Math.min(95, warmAllocations * 20 + Math.random() * 10),
      temperature: 45 + (warmAllocations * 5) + Math.random() * 10
    };
  }

  /**
   * Start continuous GPU monitoring
   */
  private startGPUMonitoring(): void {
    // Monitor GPU stats every 5 seconds
    setInterval(async (): Promise<any> => {
      try {
        const stats = await this.getGPUStats();
        this.emit('gpu:stats', stats);
        
        // Alert on high memory usage
        const memoryUsagePercent = (stats.usedMemoryMB / stats.totalMemoryMB) * 100;
        if (memoryUsagePercent > 85) {
          this.emit('gpu:high_memory_usage', { 
            usagePercent: memoryUsagePercent.toFixed(1),
            usedMB: stats.usedMemoryMB,
            totalMB: stats.totalMemoryMB
          });
        }
        
        // Alert on high temperature
        if (stats.temperature > 80) {
          this.emit('gpu:high_temperature', { 
            temperature: stats.temperature 
          });
        }
        
      } catch (error: any) {
        this.emit('gpu:monitor_error', { 
          error: error instanceof Error ? error.message : 'Unknown error' 
        });
      }
    }, 5000);
  }

  /**
   * Optimize GPU memory layout for legal AI workloads
   */
  async optimizeForLegalWorkload(): Promise<any> {
    // Pre-warm frequently used models
    const frequentModels = ['nomic-embed-text', 'legal-classifier'];
    
    for (const modelName of frequentModels) {
      try {
        const preWarmWorkerId = `prewarm_${modelName}_${Date.now()}`;
        await this.allocateGPUMemory(preWarmWorkerId, modelName, 'low');
        
        this.emit('gpu:prewarmed', { modelName });
      } catch (error: any) {
        console.warn(`Failed to pre-warm ${modelName}:`, error);
      }
    }
  }

  /**
   * Get memory recommendations for legal AI workloads
   */
  getMemoryRecommendations(): {
    recommendation: string;
    currentUsage: number;
    suggestedModels: string[];
  } {
    const totalAllocated = Array.from(this.allocations.values())
      .reduce((sum, alloc) => sum + alloc.reservedMemoryMB, 0);
    
    const usagePercent = (totalAllocated / this.totalGPUMemoryMB) * 100;
    
    let recommendation: string;
    let suggestedModels: string[] = [];
    
    if (usagePercent < 50) {
      recommendation = 'GPU memory usage is low. Consider pre-warming additional models for better performance.';
      suggestedModels = ['legal-summarizer', 'vector-similarity'];
    } else if (usagePercent < 80) {
      recommendation = 'GPU memory usage is optimal for balanced performance and flexibility.';
    } else {
      recommendation = 'GPU memory usage is high. Consider evicting low-priority models or reducing batch sizes.';
    }
    
    return {
      recommendation,
      currentUsage: usagePercent,
      suggestedModels
    };
  }

  /**
   * Shutdown and cleanup
   */
  async shutdown(): Promise<any> {
    clearInterval(this.cleanupInterval);
    
    if (this.monitoringProcess) {
      this.monitoringProcess.kill();
    }
    
    // Release all allocations
    for (const workerId of this.allocations.keys()) {
      await this.releaseGPUMemory(workerId);
    }
    
    this.emit('gpu:shutdown');
  }
}

// Global GPU memory manager for legal AI platform
export const legalAIGPUManager = new GPUMemoryManager();

// Auto-optimize on startup
legalAIGPUManager.optimizeForLegalWorkload().catch(console.error);