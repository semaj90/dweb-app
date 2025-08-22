// WebGPU Manager - GPU Acceleration for Legal AI Platform
// Provides high-performance computing capabilities for embeddings, search, and document processing

import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

export interface WebGPUCapabilities {
  device: GPUDevice | null;
  adapter: GPUAdapter | null;
  features: Set<string>;
  limits: Record<string, number>;
  isSupported: boolean;
  maxComputeWorkgroupsPerDimension: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxStorageBufferBindingSize: number;
}

export interface GPUComputeTask {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startTime?: number;
  endTime?: number;
  inputSize: number;
  outputSize: number;
  workgroups: number;
  shader: string;
}

export interface WebGPUMetrics {
  totalTasks: number;
  completedTasks: number;
  failedTasks: number;
  averageExecutionTime: number;
  totalGPUMemoryUsed: number;
  throughput: number; // tasks per second
}

// WebGPU feature detection and initialization
export class WebGPUManager {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private capabilities: WebGPUCapabilities = {
    device: null,
    adapter: null,
    features: new Set(),
    limits: {},
    isSupported: false,
    maxComputeWorkgroupsPerDimension: 0,
    maxComputeInvocationsPerWorkgroup: 0,
    maxStorageBufferBindingSize: 0
  };
  private metrics: WebGPUMetrics = {
    totalTasks: 0,
    completedTasks: 0,
    failedTasks: 0,
    averageExecutionTime: 0,
    totalGPUMemoryUsed: 0,
    throughput: 0
  };
  private activeTasks = new Map<string, GPUComputeTask>();
  private taskHistory: GPUComputeTask[] = [];

  async initialize(): Promise<boolean> {
    if (!browser || !navigator.gpu) {
      console.warn('WebGPU not supported in this browser');
      return false;
    }

    try {
      console.log('🔄 Initializing WebGPU...');

      // Request adapter
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });

      if (!this.adapter) {
        throw new Error('No WebGPU adapter found');
      }

      // Request device with required features
      this.device = await this.adapter.requestDevice({
        requiredFeatures: ['shader-f16'] as GPUFeatureName[],
        requiredLimits: {
          maxComputeWorkgroupSizeX: 256,
          maxComputeWorkgroupSizeY: 256,
          maxComputeWorkgroupSizeZ: 64
        }
      });

      // Setup error handling
      this.device.lost.then((info) => {
        console.error('WebGPU device lost:', info);
      });

      // Update capabilities
      this.capabilities = {
        device: this.device,
        adapter: this.adapter,
        features: new Set(Array.from(this.adapter.features)),
        limits: this.adapter.limits as Record<string, number>,
        isSupported: true,
        maxComputeWorkgroupsPerDimension: this.adapter.limits.maxComputeWorkgroupsPerDimension,
        maxComputeInvocationsPerWorkgroup: this.adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize
      };

      // Update stores
      webgpuCapabilities.set(this.capabilities);
      webgpuMetrics.set(this.metrics);

      console.log('✓ WebGPU initialized successfully');
      console.log('📊 GPU Features:', Array.from(this.capabilities.features));
      console.log('🔧 GPU Limits:', this.capabilities.limits);

      return true;

    } catch (error) {
      console.error('❌ WebGPU initialization failed:', error);
      this.capabilities.isSupported = false;
      webgpuCapabilities.set(this.capabilities);
      return false;
    }
  }

  // ============ Vector Operations ============

  /**
   * Compute cosine similarity between vectors using GPU
   */
  async computeCosineSimilarity(
    vectorA: Float32Array,
    vectorB: Float32Array
  ): Promise<number> {
    if (!this.device || vectorA.length !== vectorB.length) {
      throw new Error('Invalid input or WebGPU not initialized');
    }

    const task = this.createTask('cosine-similarity', vectorA.length * 8);

    try {
      const shaderCode = `
        @group(0) @binding(0) var<storage, read> vectorA: array<f32>;
        @group(0) @binding(1) var<storage, read> vectorB: array<f32>;
        @group(0) @binding(2) var<storage, read_write> result: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          let size = arrayLength(&vectorA);
          
          if (index >= size) {
            return;
          }

          // Compute dot product components
          let dotProduct = vectorA[index] * vectorB[index];
          let normA = vectorA[index] * vectorA[index];
          let normB = vectorB[index] * vectorB[index];
          
          // Store partial results
          result[index] = dotProduct;
          result[index + size] = normA;
          result[index + size * 2u] = normB;
        }
      `;

      const module = this.device.createShaderModule({ code: shaderCode });
      const pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' }
      });

      // Create buffers
      const bufferA = this.createBuffer(vectorA);
      const bufferB = this.createBuffer(vectorB);
      const resultBuffer = this.device.createBuffer({
        size: vectorA.length * 3 * 4, // 3 arrays of f32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });

      // Create bind group
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: bufferA } },
          { binding: 1, resource: { buffer: bufferB } },
          { binding: 2, resource: { buffer: resultBuffer } }
        ]
      });

      // Execute compute shader
      const commandEncoder = this.device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(pipeline);
      computePass.setBindGroup(0, bindGroup);
      
      const workgroups = Math.ceil(vectorA.length / 256);
      computePass.dispatchWorkgroups(workgroups);
      computePass.end();

      // Read results
      const readBuffer = this.createReadBuffer(resultBuffer.size);
      commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultBuffer.size);
      
      this.device.queue.submit([commandEncoder.finish()]);
      // Wait for GPU operations to complete
      await new Promise(resolve => setTimeout(resolve, 10));

      const results = await this.readBuffer(readBuffer);
      const float32Results = new Float32Array(results);

      // Calculate final cosine similarity
      let dotProduct = 0;
      let normA = 0;
      let normB = 0;

      for (let i = 0; i < vectorA.length; i++) {
        dotProduct += float32Results[i];
        normA += float32Results[i + vectorA.length];
        normB += float32Results[i + vectorA.length * 2];
      }

      const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));

      this.completeTask(task, { similarity });
      return similarity;

    } catch (error) {
      this.failTask(task, error.message);
      throw error;
    }
  }

  /**
   * Batch vector similarity computation
   */
  async batchVectorSimilarity(
    queryVector: Float32Array,
    documentVectors: Float32Array[],
    options: { topK?: number; threshold?: number } = {}
  ): Promise<Array<{ index: number; similarity: number }>> {
    if (!this.device || documentVectors.length === 0) {
      throw new Error('Invalid input or WebGPU not initialized');
    }

    const { topK = 10, threshold = 0.0 } = options;
    const task = this.createTask('batch-similarity', queryVector.length * documentVectors.length * 4);

    try {
      const shaderCode = `
        @group(0) @binding(0) var<storage, read> queryVector: array<f32>;
        @group(0) @binding(1) var<storage, read> documentVectors: array<f32>;
        @group(0) @binding(2) var<storage, read_write> similarities: array<f32>;
        @group(0) @binding(3) var<uniform> params: Params;

        struct Params {
          vectorSize: u32,
          numDocuments: u32,
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let docIndex = global_id.x;
          
          if (docIndex >= params.numDocuments) {
            return;
          }

          var dotProduct: f32 = 0.0;
          var queryNorm: f32 = 0.0;
          var docNorm: f32 = 0.0;

          for (var i: u32 = 0u; i < params.vectorSize; i++) {
            let queryVal = queryVector[i];
            let docVal = documentVectors[docIndex * params.vectorSize + i];
            
            dotProduct += queryVal * docVal;
            queryNorm += queryVal * queryVal;
            docNorm += docVal * docVal;
          }

          let similarity = dotProduct / (sqrt(queryNorm) * sqrt(docNorm));
          similarities[docIndex] = similarity;
        }
      `;

      const module = this.device.createShaderModule({ code: shaderCode });
      const pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' }
      });

      // Flatten document vectors
      const flatDocVectors = new Float32Array(queryVector.length * documentVectors.length);
      documentVectors.forEach((vec, idx) => {
        flatDocVectors.set(vec, idx * queryVector.length);
      });

      // Create buffers
      const queryBuffer = this.createBuffer(queryVector);
      const docBuffer = this.createBuffer(flatDocVectors);
      const resultBuffer = this.device.createBuffer({
        size: documentVectors.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });

      // Parameters buffer
      const params = new Uint32Array([queryVector.length, documentVectors.length]);
      const paramsBuffer = this.createBuffer(params);

      // Create bind group
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: queryBuffer } },
          { binding: 1, resource: { buffer: docBuffer } },
          { binding: 2, resource: { buffer: resultBuffer } },
          { binding: 3, resource: { buffer: paramsBuffer } }
        ]
      });

      // Execute
      const commandEncoder = this.device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(pipeline);
      computePass.setBindGroup(0, bindGroup);
      
      const workgroups = Math.ceil(documentVectors.length / 256);
      computePass.dispatchWorkgroups(workgroups);
      computePass.end();

      // Read results
      const readBuffer = this.createReadBuffer(resultBuffer.size);
      commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultBuffer.size);
      
      this.device.queue.submit([commandEncoder.finish()]);
      // Wait for GPU operations to complete
      await new Promise(resolve => setTimeout(resolve, 10));

      const results = await this.readBuffer(readBuffer);
      const similarities = new Float32Array(results);

      // Process results
      const scoredResults = Array.from(similarities)
        .map((similarity, index) => ({ index, similarity }))
        .filter(result => result.similarity >= threshold)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topK);

      this.completeTask(task, { results: scoredResults.length });
      return scoredResults;

    } catch (error) {
      this.failTask(task, error.message);
      throw error;
    }
  }

  // ============ Text Processing ============

  /**
   * GPU-accelerated text chunking and preprocessing
   */
  async processTextChunks(
    text: string,
    chunkSize: number = 512,
    overlap: number = 50
  ): Promise<Array<{ text: string; position: number; length: number }>> {
    const task = this.createTask('text-processing', text.length);

    try {
      // For now, implement on CPU but structure for GPU acceleration
      // In a full implementation, this would use WebGPU compute shaders
      // for parallel text processing and tokenization
      
      const chunks: Array<{ text: string; position: number; length: number }> = [];
      const words = text.split(/\s+/);
      
      let currentChunk = '';
      let wordCount = 0;
      let position = 0;
      
      for (let i = 0; i < words.length; i++) {
        const word = words[i];
        
        if (wordCount >= chunkSize && currentChunk.length > 0) {
          chunks.push({
            text: currentChunk.trim(),
            position,
            length: currentChunk.length
          });
          
          // Handle overlap
          const overlapWords = words.slice(Math.max(0, i - overlap), i);
          currentChunk = overlapWords.join(' ') + ' ';
          position = text.indexOf(overlapWords[0], position);
          wordCount = overlapWords.length;
        }
        
        currentChunk += word + ' ';
        wordCount++;
      }
      
      // Add final chunk
      if (currentChunk.trim().length > 0) {
        chunks.push({
          text: currentChunk.trim(),
          position,
          length: currentChunk.length
        });
      }

      this.completeTask(task, { chunks: chunks.length });
      return chunks;

    } catch (error) {
      this.failTask(task, error.message);
      throw error;
    }
  }

  /**
   * GPU-accelerated semantic analysis
   */
  async analyzeSemanticPatterns(
    embeddings: Float32Array[],
    options: { clusters?: number; iterations?: number } = {}
  ): Promise<{ clusters: number[][]; centroids: Float32Array[] }> {
    const { clusters = 5, iterations = 10 } = options;
    const task = this.createTask('semantic-analysis', embeddings.length * embeddings[0].length * 4);

    try {
      // Simplified K-means clustering on GPU
      // In production, this would use more sophisticated clustering algorithms

      const results = await this.performKMeansClustering(embeddings, clusters, iterations);
      
      this.completeTask(task, { clusters: results.clusters.length });
      return results;

    } catch (error) {
      this.failTask(task, error.message);
      throw error;
    }
  }

  // ============ Utility Methods ============

  private async performKMeansClustering(
    embeddings: Float32Array[],
    k: number,
    iterations: number
  ): Promise<{ clusters: number[][]; centroids: Float32Array[] }> {
    // Simplified implementation - in production, this would be fully GPU-accelerated
    const clusters: number[][] = Array(k).fill(null).map(() => []);
    const centroids: Float32Array[] = [];

    // Initialize centroids randomly
    for (let i = 0; i < k; i++) {
      centroids.push(new Float32Array(embeddings[Math.floor(Math.random() * embeddings.length)]));
    }

    // Perform clustering iterations
    for (let iter = 0; iter < iterations; iter++) {
      // Clear clusters
      clusters.forEach(cluster => cluster.length = 0);

      // Assign points to closest centroids
      for (let i = 0; i < embeddings.length; i++) {
        let closestCentroid = 0;
        let minDistance = Infinity;

        for (let j = 0; j < k; j++) {
          const distance = await this.computeCosineSimilarity(embeddings[i], centroids[j]);
          if (1 - distance < minDistance) {
            minDistance = 1 - distance;
            closestCentroid = j;
          }
        }

        clusters[closestCentroid].push(i);
      }

      // Update centroids
      for (let i = 0; i < k; i++) {
        if (clusters[i].length > 0) {
          const newCentroid = new Float32Array(embeddings[0].length);
          for (const pointIdx of clusters[i]) {
            for (let j = 0; j < newCentroid.length; j++) {
              newCentroid[j] += embeddings[pointIdx][j];
            }
          }
          for (let j = 0; j < newCentroid.length; j++) {
            newCentroid[j] /= clusters[i].length;
          }
          centroids[i] = newCentroid;
        }
      }
    }

    return { clusters, centroids };
  }

  private createBuffer(data: Float32Array | Uint32Array): GPUBuffer {
    const buffer = this.device!.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device!.queue.writeBuffer(buffer, 0, data.buffer);
    return buffer;
  }

  private createReadBuffer(size: number): GPUBuffer {
    return this.device!.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  }

  private async readBuffer(buffer: GPUBuffer): Promise<ArrayBuffer> {
    await buffer.mapAsync(GPUMapMode.READ);
    const data = buffer.getMappedRange().slice(0);
    buffer.unmap();
    return data;
  }

  private createTask(name: string, inputSize: number): GPUComputeTask {
    const task: GPUComputeTask = {
      id: `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      status: 'running',
      startTime: performance.now(),
      inputSize,
      outputSize: 0,
      workgroups: Math.ceil(inputSize / 256),
      shader: name
    };

    this.activeTasks.set(task.id, task);
    this.metrics.totalTasks++;
    webgpuMetrics.set({ ...this.metrics });

    return task;
  }

  private completeTask(task: GPUComputeTask, result: unknown): void {
    task.status = 'completed';
    task.endTime = performance.now();
    task.outputSize = result.results || result.chunks || result.similarity || 0;

    this.activeTasks.delete(task.id);
    this.taskHistory.push(task);

    // Update metrics
    this.metrics.completedTasks++;
    const executionTime = task.endTime! - task.startTime!;
    this.metrics.averageExecutionTime = 
      (this.metrics.averageExecutionTime * (this.metrics.completedTasks - 1) + executionTime) / 
      this.metrics.completedTasks;
    
    this.updateThroughput();
    webgpuMetrics.set({ ...this.metrics });
  }

  private failTask(task: GPUComputeTask, error: string): void {
    task.status = 'failed';
    task.endTime = performance.now();

    this.activeTasks.delete(task.id);
    this.taskHistory.push(task);

    this.metrics.failedTasks++;
    this.updateThroughput();
    webgpuMetrics.set({ ...this.metrics });

    console.error(`GPU task ${task.name} failed:`, error);
  }

  private updateThroughput(): void {
    const now = Date.now();
    const recentTasks = this.taskHistory.filter(t => 
      t.endTime && (now - t.endTime) < 60000 // Last minute
    );
    this.metrics.throughput = recentTasks.length / 60; // tasks per second
  }

  getCapabilities(): WebGPUCapabilities {
    return { ...this.capabilities };
  }

  getMetrics(): WebGPUMetrics {
    return { ...this.metrics };
  }

  getActiveTasks(): GPUComputeTask[] {
    return Array.from(this.activeTasks.values());
  }

  getTaskHistory(): GPUComputeTask[] {
    return [...this.taskHistory];
  }

  isSupported(): boolean {
    return this.capabilities.isSupported;
  }

  async cleanup(): Promise<void> {
    if (this.device) {
      // WebGPU devices don't have a destroy method - they're garbage collected
      this.device = null;
    }
    this.adapter = null;
    this.activeTasks.clear();
    console.log('🧹 WebGPU resources cleaned up');
  }
}

// Singleton instance
export const webgpuManager = new WebGPUManager();

// Svelte stores for reactive access
export const webgpuCapabilities = writable<WebGPUCapabilities>({
  device: null,
  adapter: null,
  features: new Set(),
  limits: {},
  isSupported: false,
  maxComputeWorkgroupsPerDimension: 0,
  maxComputeInvocationsPerWorkgroup: 0,
  maxStorageBufferBindingSize: 0
});

export const webgpuMetrics = writable<WebGPUMetrics>({
  totalTasks: 0,
  completedTasks: 0,
  failedTasks: 0,
  averageExecutionTime: 0,
  totalGPUMemoryUsed: 0,
  throughput: 0
});

export const webgpuTasks = writable<GPUComputeTask[]>([]);

// Derived stores
export const webgpuStatus = derived(
  [webgpuCapabilities, webgpuMetrics],
  ([$capabilities, $metrics]) => ({
    isReady: $capabilities.isSupported && $capabilities.device !== null,
    performance: {
      tasksPerSecond: $metrics.throughput,
      successRate: $metrics.totalTasks > 0 
        ? $metrics.completedTasks / $metrics.totalTasks 
        : 0,
      averageTime: $metrics.averageExecutionTime
    },
    features: Array.from($capabilities.features),
    memoryUsage: $metrics.totalGPUMemoryUsed
  })
);

// Auto-initialize on browser load
if (browser) {
  webgpuManager.initialize().catch(console.error);
}

// Helper functions for common GPU operations
export const webgpuHelpers = {
  // Vector operations
  async computeSimilarity(vectorA: Float32Array, vectorB: Float32Array): Promise<number> {
    return webgpuManager.computeCosineSimilarity(vectorA, vectorB);
  },

  async batchSimilarity(
    query: Float32Array, 
    documents: Float32Array[], 
    options?: { topK?: number; threshold?: number }
  ): Promise<Array<{ index: number; similarity: number }>> {
    return webgpuManager.batchVectorSimilarity(query, documents, options);
  },

  // Text processing
  async processText(text: string, chunkSize?: number, overlap?: number) {
    return webgpuManager.processTextChunks(text, chunkSize, overlap);
  },

  async analyzeSemantics(embeddings: Float32Array[], options?: { clusters?: number; iterations?: number }) {
    return webgpuManager.analyzeSemanticPatterns(embeddings, options);
  },

  // Utility functions
  isSupported(): boolean {
    return webgpuManager.isSupported();
  },

  getCapabilities(): WebGPUCapabilities {
    return webgpuManager.getCapabilities();
  },

  getMetrics(): WebGPUMetrics {
    return webgpuManager.getMetrics();
  }
};

export default webgpuManager;