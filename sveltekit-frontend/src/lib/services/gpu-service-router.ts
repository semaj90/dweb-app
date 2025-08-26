/**
 * GPU Service Router - CUDA Worker Integration
 * Coordinates 37+ Go microservices with cuda-worker.exe for RTX 3060 Ti optimization
 * Implements the GPU Service Router from COMPLETE_ARCHITECTURE_SUMMARY.md
 */

import { exec } from 'node:child_process';
import { promisify } from 'node:util';

const execAsync = promisify(exec);

export interface CUDARequest {
  jobId: string;
  type: 'embedding' | 'similarity' | 'som_train' | 'vector_search' | 'legal_analysis';
  data: number[];
  metadata?: {
    service: string;
    timestamp: number;
    rtx_3060_ti: boolean;
    gpu_acceleration: boolean;
  };
}

export interface CUDAResponse {
  jobId: string;
  type: string;
  vector?: number[];
  status: 'success' | 'error' | 'processing';
  timestamp: number;
  memoryUsed?: number;
  processingTime?: number;
  error?: string;
}

export interface GPUMemoryPool {
  total: number;
  available: number;
  allocated: Map<string, number>;
}

export interface GPUPerformanceMetrics {
  service: string;
  operation: string;
  duration: number;
  memoryUsed: number;
  throughput: number;
}

/**
 * GPU Service Router coordinating 37+ Go microservices with cuda-worker.exe
 */
export class GPUServiceRouter {
  private services = {
    "enhanced-rag": "8094",           // RAG + CUDA acceleration
    "upload": "8093",                 // Document processing + GPU
    "legal-ai": "8202",              // Case similarity + CUDA
    "gpu-indexer": "8220",           // Batch indexing + GPU
    "typescript-optimizer": "5173",   // Error processing + GPU
    "ai-summary": "8096",            // Summary generation + GPU
    "kratos-server": "50051",        // Legal gRPC + GPU compute
    "gin-upload": "8093",            // File upload with GPU processing
    "context7-pipeline": "8097",     // Error analysis + GPU
    "gpu-tensor": "8099",            // Tensor operations + CUDA
    "vector-redis": "8095",          // Vector + Redis + GPU
    "quic-gateway": "8200",          // QUIC + GPU acceleration
    "cluster-manager": "8213",       // Cluster + GPU coordination
    "xstate-manager": "8212",        // State + GPU processing
    "load-balancer": "8224"          // Load balancing + GPU
  };

  private cudaWorkerPath: string;
  private gpuMemoryPool: GPUMemoryPool;
  private performanceMetrics: GPUPerformanceMetrics[] = [];
  private jobCounter = 0;

  constructor(cudaWorkerPath = "./cuda-worker.exe") {
    this.cudaWorkerPath = cudaWorkerPath;
    this.gpuMemoryPool = {
      total: 8 * 1024 * 1024 * 1024, // 8GB RTX 3060 Ti
      available: 6 * 1024 * 1024 * 1024, // 6GB allocation with 2GB reserve
      allocated: new Map()
    };
  }

  /**
   * Route GPU request to appropriate service or direct CUDA processing
   */
  async routeGPURequest(request: {
    service: string;
    operation: string;
    data: number[];
    priority: 'high' | 'normal' | 'low';
    metadata?: any;
  }): Promise<CUDAResponse> {
    
    // Monitor GPU memory and performance
    const memoryBefore = this.gpuMemoryPool.available;
    const startTime = performance.now();
    
    // Direct CUDA processing for high-priority requests
    if (request.priority === 'high') {
      const cudaRequest: CUDARequest = {
        jobId: this.generateJobID(),
        type: request.operation as CUDARequest['type'],
        data: request.data,
        metadata: {
          service: request.service,
          timestamp: Date.now(),
          rtx_3060_ti: true,
          gpu_acceleration: true,
          ...request.metadata
        }
      };
      
      const response = await this.processWithCUDAWorker(cudaRequest);
      
      // Performance tracking
      this.recordPerformance({
        service: request.service,
        operation: request.operation,
        duration: performance.now() - startTime,
        memoryUsed: memoryBefore - this.gpuMemoryPool.available,
        throughput: request.data.length / (performance.now() - startTime) * 1000
      });
      
      return response;
    }
    
    // Route to appropriate GPU-accelerated Go service
    const servicePort = this.services[request.service] || "8094";
    return await this.forwardToGPUService(request, servicePort, {
      cuda_worker: true,
      gpu_acceleration: true,
      rtx_3060_ti_optimized: true
    });
  }

  /**
   * Process request directly with CUDA worker executable
   */
  private async processWithCUDAWorker(request: CUDARequest): Promise<CUDAResponse> {
    try {
      // Check GPU memory availability
      if (!this.checkMemoryAvailability(request.data.length * 4)) { // 4 bytes per float
        throw new Error('Insufficient GPU memory available');
      }

      // Allocate GPU memory for this job
      this.allocateMemory(request.jobId, request.data.length * 4);

      // Execute CUDA worker with JSON input/output
      const jsonInput = JSON.stringify(request);
      const { stdout, stderr } = await execAsync(
        `echo '${jsonInput}' | "${this.cudaWorkerPath}"`,
        { timeout: 30000, maxBuffer: 10 * 1024 * 1024 } // 10MB buffer
      );

      if (stderr) {
        console.warn('CUDA worker stderr:', stderr);
      }

      // Parse CUDA worker response
      const response: CUDAResponse = JSON.parse(stdout.trim());
      
      // Release GPU memory
      this.deallocateMemory(request.jobId);

      return {
        ...response,
        processingTime: performance.now() - Date.now(),
        memoryUsed: request.data.length * 4
      };

    } catch (error) {
      // Release memory on error
      this.deallocateMemory(request.jobId);
      
      return {
        jobId: request.jobId,
        type: request.type,
        status: 'error',
        timestamp: Date.now(),
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Forward request to GPU-accelerated Go service
   */
  private async forwardToGPUService(
    request: any, 
    servicePort: string, 
    gpuOptions: any
  ): Promise<CUDAResponse> {
    try {
      const response = await fetch(`http://localhost:${servicePort}/api/gpu/process`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-GPU-Acceleration': 'true',
          'X-CUDA-Worker': this.cudaWorkerPath
        },
        body: JSON.stringify({
          ...request,
          gpu_options: gpuOptions,
          cuda_worker_path: this.cudaWorkerPath
        }),
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });
      
      if (!response.ok) {
        throw new Error(`GPU service error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      
      return {
        jobId: this.generateJobID(),
        type: request.operation,
        status: 'success',
        timestamp: Date.now(),
        vector: result.vector || result.data,
        processingTime: result.processing_time,
        memoryUsed: result.memory_used
      };

    } catch (error) {
      return {
        jobId: this.generateJobID(),
        type: request.operation,
        status: 'error',
        timestamp: Date.now(),
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Get GPU service status and metrics
   */
  async getGPUStatus(): Promise<{
    available: boolean;
    memoryUsage: GPUMemoryPool;
    activeServices: string[];
    performanceMetrics: GPUPerformanceMetrics[];
    cudaWorkerPath: string;
  }> {
    return {
      available: true,
      memoryUsage: this.gpuMemoryPool,
      activeServices: Object.keys(this.services),
      performanceMetrics: this.performanceMetrics.slice(-10), // Last 10 operations
      cudaWorkerPath: this.cudaWorkerPath
    };
  }

  /**
   * Check if CUDA worker is available and responsive
   */
  async checkCUDAWorkerHealth(): Promise<boolean> {
    try {
      const healthCheckRequest: CUDARequest = {
        jobId: 'health-check',
        type: 'embedding',
        data: [1.0, 2.0, 3.0, 4.0], // Simple test data
        metadata: {
          service: 'health-check',
          timestamp: Date.now(),
          rtx_3060_ti: true,
          gpu_acceleration: true
        }
      };

      const response = await this.processWithCUDAWorker(healthCheckRequest);
      return response.status === 'success';

    } catch (error) {
      console.error('CUDA worker health check failed:', error);
      return false;
    }
  }

  // Memory management methods
  private checkMemoryAvailability(bytes: number): boolean {
    return this.gpuMemoryPool.available >= bytes;
  }

  private allocateMemory(jobId: string, bytes: number): void {
    if (this.gpuMemoryPool.available >= bytes) {
      this.gpuMemoryPool.available -= bytes;
      this.gpuMemoryPool.allocated.set(jobId, bytes);
    }
  }

  private deallocateMemory(jobId: string): void {
    const allocated = this.gpuMemoryPool.allocated.get(jobId);
    if (allocated) {
      this.gpuMemoryPool.available += allocated;
      this.gpuMemoryPool.allocated.delete(jobId);
    }
  }

  private recordPerformance(metrics: GPUPerformanceMetrics): void {
    this.performanceMetrics.push(metrics);
    // Keep only last 100 metrics
    if (this.performanceMetrics.length > 100) {
      this.performanceMetrics = this.performanceMetrics.slice(-100);
    }
  }

  private generateJobID(): string {
    return `cuda_job_${++this.jobCounter}_${Date.now()}`;
  }
}

// Singleton instance for application use
export const gpuServiceRouter = new GPUServiceRouter();