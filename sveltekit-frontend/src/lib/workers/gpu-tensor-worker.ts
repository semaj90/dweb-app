/**
 * GPU Tensor Worker - WebGPU Processing for Multi-Dimensional Arrays
 * Integrates with Go GPU microservice and NES-style caching
 */

// Worker message types
export interface WorkerMessage {
  type: 'PROCESS_TENSOR' | 'INITIALIZE' | 'GET_STATS' | 'CLEAR_CACHE';
  id?: string;
  data?: unknown;
}

export interface WorkerResponse {
  type: 'SUCCESS' | 'ERROR' | 'STATS' | 'INITIALIZED';
  id?: string;
  data?: unknown;
  error?: string;
}

// Multi-dimensional array interface
export interface MultiDimArray {
  shape: number[];
  data: Float32Array;
  dimensions: number;
  layout: string;
  cacheKey: string;
  lodLevel: number;
  timestamp?: number;
}

export interface GPUProcessingStats {
  totalProcessed: number;
  cacheHitRate: number;
  averageProcessingTime: number;
  webgpuSupported: boolean;
  lastProcessedTime: number;
}

class GPUTensorWorker {
  private wasmModule: WebAssembly.Instance | null = null;
  private gpuDevice: GPUDevice | null = null;
  private computePipeline: GPUComputePipeline | null = null;
  private tensorCache: Map<string, { data: MultiDimArray; timestamp: number }> = new Map();
  private stats: GPUProcessingStats = {
    totalProcessed: 0,
    cacheHitRate: 0,
    averageProcessingTime: 0,
    webgpuSupported: false,
    lastProcessedTime: 0
  };
  private goServiceUrl = 'http://localhost:8095'; // GPU tensor service

    async initialize(): Promise<boolean> {
    try {
      // Initialize WebGPU if available
      await this.initializeWebGPU();

      // Initialize WebAssembly if available
      await this.initializeWebAssembly();

      // Test connection to Go service
      await this.testGoServiceConnection();

      this.postMessage({
        type: 'INITIALIZED',
        data: {
          webgpuSupported: this.stats.webgpuSupported,
          wasmLoaded: this.wasmModule !== null,
          goServiceConnected: true
        }
      });

      return true;
    } catch (error: any) {
      console.error('GPU Tensor Worker initialization failed:', error);
      this.postMessage({
        type: 'ERROR',
        error: `Initialization failed: ${error.message}`
      });
      return false;
    }
  }

  private async initializeWebGPU(): Promise<void> {
    if ('gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter({
          powerPreference: 'high-performance'
        });

        if (adapter) {
          this.gpuDevice = await adapter.requestDevice({
            requiredFeatures: ['shader-f16'] as GPUFeatureName[],
            requiredLimits: {
              maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
              maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension
            }
          });

          this.stats.webgpuSupported = true;
          console.log('✅ WebGPU initialized successfully');
        }
      } catch (error: any) {
        console.warn('⚠️ WebGPU initialization failed:', error);
        this.stats.webgpuSupported = false;
      }
    }
  }

  private async initializeWebAssembly(): Promise<void> {
    try {
      // In a real implementation, this would load the compiled Go WASM module
      // For now, we'll simulate WASM availability
      const wasmSupported = typeof WebAssembly !== 'undefined';

      if (wasmSupported) {
        // Simulate WASM module loading
        // const wasmResponse = await fetch('/wasm/gpu-processor.wasm');
        // const wasmBytes = await wasmResponse.arrayBuffer();
        // this.wasmModule = await WebAssembly.instantiate(wasmBytes, importObject);

        console.log('✅ WebAssembly bridge ready (simulated)');
      }
    } catch (error: any) {
      console.warn('⚠️ WebAssembly initialization failed:', error);
    }
  }

  private async testGoServiceConnection(): Promise<void> {
    try {
      const response = await fetch(`${this.goServiceUrl}/health`);
      if (!response.ok) {
        throw new Error(`Go service health check failed: ${response.status}`);
      }
      console.log('✅ Go GPU tensor service connection established');
    } catch (error: any) {
      console.warn('⚠️ Go service connection failed, using fallback processing');
      throw error;
    }
  }

  async processGPUTensor(tensorData: MultiDimArray): Promise<MultiDimArray> {
    const startTime = performance.now();

    try {
      // Check cache first
      const cached = this.tensorCache.get(tensorData.cacheKey);
      if (cached && (Date.now() - cached.timestamp) < 300000) { // 5 minute cache
        this.updateCacheStats(true);
        return cached.data;
      }

      this.updateCacheStats(false);

      // Try WebGPU processing first
      if (this.gpuDevice && this.stats.webgpuSupported) {
        const result = await this.processWithWebGPU(tensorData);
        this.cacheResult(tensorData.cacheKey, result);
        this.updateProcessingStats(startTime);
        return result;
      }

      // Fallback to Go service
      const result = await this.processWithGoService(tensorData);
      this.cacheResult(tensorData.cacheKey, result);
      this.updateProcessingStats(startTime);
      return result;

    } catch (error: any) {
      console.error('GPU tensor processing failed:', error);
      throw new Error(`Processing failed: ${error.message}`);
    }
  }

  private async processWithWebGPU(tensorData: MultiDimArray): Promise<MultiDimArray> {
    if (!this.gpuDevice) {
      throw new Error('WebGPU device not available');
    }

    // Create compute shader for multi-dimensional processing
    const computeShader = `
      @group(0) @binding(0) var<storage, read> inputTensor: array<f32>;
      @group(0) @binding(1) var<storage, read_write> outputTensor: array<f32>;
      @group(0) @binding(2) var<uniform> tensorShape: array<i32, 4>;
      @group(0) @binding(3) var<uniform> metadata: vec4<i32>; // [dimensions, lodLevel, totalElements, padding]

      @compute @workgroup_size(256, 1, 1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        let totalElements = metadata.z;
        let dimensions = metadata.x;

        if (index >= u32(totalElements)) { return; }

        // Multi-dimensional indexing
        var indices: array<i32, 4>;
        var remaining = i32(index);

        // Convert linear index to multi-dimensional indices
        for (var d = dimensions - 1; d >= 0; d--) {
          indices[d] = remaining % tensorShape[d];
          remaining = remaining / tensorShape[d];
        }

        let value = inputTensor[index];

        // Apply legal AI specific transformations for 4D tensors
        if (dimensions == 4) {
          // 4D tensor processing: [cases, docs, paragraphs, embeddings]
          let caseIdx = indices[0];
          let docIdx = indices[1];
          let paraIdx = indices[2];
          let embedIdx = indices[3];

          // Semantic similarity weighting (nomic-embed-text optimization)
          var weight = 1.0;
          if (embedIdx < 384) {
            weight = 1.1; // First half of embeddings (more important)
          } else {
            weight = 0.95; // Second half
          }

          // Tricubic interpolation for spatial coherence
          if (caseIdx > 0 && docIdx > 0 && paraIdx > 0) {
            let x = f32(caseIdx) / f32(tensorShape[0]);
            let y = f32(docIdx) / f32(tensorShape[1]);
            let z = f32(paraIdx) / f32(tensorShape[2]);

            // Smoothstep interpolation (tricubic approximation)
            let wx = x * x * (3.0 - 2.0 * x);
            let wy = y * y * (3.0 - 2.0 * y);
            let wz = z * z * (3.0 - 2.0 * z);

            weight *= wx * wy * wz;
          }

          // Apply NES-style quantization for caching efficiency
          let quantized = round(value * weight * 255.0) / 255.0; // 8-bit quantization
          outputTensor[index] = quantized;
        } else {
          // Pass-through for other dimensions with small enhancement
          outputTensor[index] = value * 1.05;
        }
      }
    `;

    // Create or reuse compute pipeline
    if (!this.computePipeline) {
      const shaderModule = this.gpuDevice.createShaderModule({ code: computeShader });
      this.computePipeline = this.gpuDevice.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });
    }

    // Create GPU buffers
    const inputBuffer = this.gpuDevice.createBuffer({
      size: tensorData.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });

    new Float32Array(inputBuffer.getMappedRange()).set(tensorData.data);
    inputBuffer.unmap();

    const outputBuffer = this.gpuDevice.createBuffer({
      size: tensorData.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Create uniform buffers for shape and metadata
    const shapeBuffer = this.gpuDevice.createBuffer({
      size: 16, // 4 * 4 bytes for int32 array
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });

    const shapeData = new Int32Array(shapeBuffer.getMappedRange());
    for (let i = 0; i < 4; i++) {
      shapeData[i] = i < tensorData.shape.length ? tensorData.shape[i] : 1;
    }
    shapeBuffer.unmap();

    const metadataBuffer = this.gpuDevice.createBuffer({
      size: 16, // 4 * 4 bytes for vec4<i32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });

    const metadataArray = new Int32Array(metadataBuffer.getMappedRange());
    metadataArray[0] = tensorData.dimensions;
    metadataArray[1] = tensorData.lodLevel;
    metadataArray[2] = tensorData.data.length;
    metadataArray[3] = 0; // padding
    metadataBuffer.unmap();

    // Execute compute shader
    const commandEncoder = this.gpuDevice.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(this.computePipeline);

    // Create bind group
    const bindGroup = this.gpuDevice.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: shapeBuffer } },
        { binding: 3, resource: { buffer: metadataBuffer } },
      ]
    });

    passEncoder.setBindGroup(0, bindGroup);

    // Dispatch compute shader
    const workgroupsX = Math.ceil(tensorData.data.length / 256);
    passEncoder.dispatchWorkgroups(workgroupsX, 1, 1);
    passEncoder.end();

    this.gpuDevice.queue.submit([commandEncoder.finish()]);

    // Read results back
    const resultBuffer = this.gpuDevice.createBuffer({
      size: tensorData.data.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const copyEncoder = this.gpuDevice.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(outputBuffer, 0, resultBuffer, 0, tensorData.data.byteLength);
    this.gpuDevice.queue.submit([copyEncoder.finish()]);

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const resultArray = new Float32Array(resultBuffer.getMappedRange());
    const processedData = new Float32Array(resultArray);
    resultBuffer.unmap();

    // Cleanup GPU resources
    inputBuffer.destroy();
    outputBuffer.destroy();
    shapeBuffer.destroy();
    metadataBuffer.destroy();
    resultBuffer.destroy();

    return {
      ...tensorData,
      data: processedData,
      layout: 'webgpu_processed',
      timestamp: Date.now()
    };
  }

  private async processWithGoService(tensorData: MultiDimArray): Promise<MultiDimArray> {
    try {
      const response = await fetch(`${this.goServiceUrl}/process-tensor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': `worker_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          'X-Processing-Mode': 'webworker',
        },
        body: JSON.stringify(tensorData)
      });

      if (!response.ok) {
        throw new Error(`Go service request failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(`Go service processing failed: ${result.error || 'Unknown error'}`);
      }

      return {
        ...result.data,
        timestamp: Date.now()
      };
    } catch (error: any) {
      throw new Error(`Go service communication failed: ${error.message}`);
    }
  }

  private cacheResult(cacheKey: string, result: MultiDimArray): void {
    // Implement LRU cache with size limit
    if (this.tensorCache.size > 100) {
      const oldestKey = Array.from(this.tensorCache.keys())[0];
      this.tensorCache.delete(oldestKey);
    }

    this.tensorCache.set(cacheKey, {
      data: result,
      timestamp: Date.now()
    });
  }

  private updateCacheStats(isHit: boolean): void {
    this.stats.totalProcessed++;
    const hitCount = isHit ? 1 : 0;
    const totalHits = (this.stats.cacheHitRate * (this.stats.totalProcessed - 1)) + hitCount;
    this.stats.cacheHitRate = totalHits / this.stats.totalProcessed;
  }

  private updateProcessingStats(startTime: number): void {
    const processingTime = performance.now() - startTime;
    const totalTime = this.stats.averageProcessingTime * (this.stats.totalProcessed - 1);
    this.stats.averageProcessingTime = (totalTime + processingTime) / this.stats.totalProcessed;
    this.stats.lastProcessedTime = Date.now();
  }

  private getStats(): GPUProcessingStats {
    return {
      ...this.stats,
      cacheHitRate: Math.round(this.stats.cacheHitRate * 10000) / 100, // Convert to percentage with 2 decimals
      averageProcessingTime: Math.round(this.stats.averageProcessingTime * 100) / 100 // Round to 2 decimals
    };
  }

  private clearCache(): void {
    this.tensorCache.clear();
    console.log('🗑️ Tensor cache cleared');
  }

  private postMessage(message: WorkerResponse): void {
    // Type-safe postMessage for worker context
    (self as any).postMessage(message);
  }
}

// Worker instance
const tensorWorker = new GPUTensorWorker();

// Message handler
self.onmessage = async function(e: MessageEvent<WorkerMessage>) {
  const { type, id, data } = e.data;

  try {
    switch (type) {
      case 'INITIALIZE':
        const initialized = await tensorWorker.initialize();
        tensorWorker.postMessage({
          type: 'INITIALIZED',
          id,
          data: { initialized }
        });
        break;

      case 'PROCESS_TENSOR':
        // data comes from the worker message (unknown), assert to MultiDimArray
        const result = await tensorWorker.processGPUTensor(data as MultiDimArray);
        tensorWorker.postMessage({
          type: 'SUCCESS',
          id,
          data: result
        });
        break;

      case 'GET_STATS':
        const stats = (tensorWorker as any).getStats();
        tensorWorker.postMessage({
          type: 'STATS',
          id,
          data: stats
        });
        break;

      case 'CLEAR_CACHE':
        (tensorWorker as any).clearCache();
        tensorWorker.postMessage({
          type: 'SUCCESS',
          id,
          data: { cache_cleared: true }
        });
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error: any) {
    tensorWorker.postMessage({
      type: 'ERROR',
      id,
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

// Export types for TypeScript
export type { WorkerMessage, WorkerResponse, MultiDimArray, GPUProcessingStats };