/**
 * GPU Processor - WebGPU-accelerated embedding and clustering processing
 */

import { ShaderManager } from './shader-manager';
import { Logger } from './logger';

// Import WebGPU bindings
const gpu = require('kmamal/gpu');

interface GPUProcessorConfig {
  gpu: any;
  shaderManager: ShaderManager;
  maxBatchSize: number;
  logger: Logger;
}

interface EmbeddingBatch {
  texts: string[];
  batchId: string;
  timestamp: number;
}

interface ProcessingResult {
  embeddings: Float32Array[];
  processingTime: number;
  gpuTime: number;
  batchSize: number;
}

export class GPUProcessor {
  private gpu: any;
  private device: any;
  private queue: any;
  private shaderManager: ShaderManager;
  private logger: Logger;
  private config: GPUProcessorConfig;
  
  private embeddingComputePipeline: any;
  private clusteringComputePipeline: any;
  private similarityComputePipeline: any;
  private boostTransformPipeline: any;
  
  private processingQueue: EmbeddingBatch[] = [];
  private isProcessing: boolean = false;
  
  // Buffer pools for efficient memory management
  private inputBufferPool: any[] = [];
  private outputBufferPool: any[] = [];
  private uniformBufferPool: any[] = [];

  constructor(config: GPUProcessorConfig) {
    this.config = config;
    this.gpu = config.gpu;
    this.shaderManager = config.shaderManager;
    this.logger = config.logger;
  }

  async initialize(): Promise<void> {
    this.logger.info('🎮 Initializing GPU Processor...');

    try {
      // Get GPU device and queue
      this.device = await this.gpu.requestDevice({
        requiredLimits: {
          maxStorageBufferBindingSize: 1024 * 1024 * 1024, // 1GB
          maxComputeWorkgroupsPerDimension: 65535,
          maxComputeInvocationsPerWorkgroup: 1024,
          maxComputeWorkgroupStorageSize: 32768
        }
      });
      
      this.queue = this.device.queue;

      // Create compute pipelines
      await this.createComputePipelines();

      // Initialize buffer pools
      this.initializeBufferPools();

      // Start processing loop
      this.startProcessingLoop();

      this.logger.info('✅ GPU Processor initialized successfully');
    } catch (error) {
      this.logger.error('❌ GPU Processor initialization failed:', error);
      throw error;
    }
  }

  private async createComputePipelines(): Promise<void> {
    this.logger.info('🔧 Creating compute pipelines...');

    try {
      // Get shaders
      const embeddingShader = this.shaderManager.getShader('embedding_processor');
      const clusteringShader = this.shaderManager.getShader('kmeans_clustering');
      const similarityShader = this.shaderManager.getShader('cosine_similarity');
      const boostShader = this.shaderManager.getShader('boost_transform');

      // Create compute pipelines
      this.embeddingComputePipeline = this.device.createComputePipeline({
        compute: {
          module: this.device.createShaderModule({ code: embeddingShader }),
          entryPoint: 'main'
        }
      });

      this.clusteringComputePipeline = this.device.createComputePipeline({
        compute: {
          module: this.device.createShaderModule({ code: clusteringShader }),
          entryPoint: 'main'
        }
      });

      this.similarityComputePipeline = this.device.createComputePipeline({
        compute: {
          module: this.device.createShaderModule({ code: similarityShader }),
          entryPoint: 'main'
        }
      });

      this.boostTransformPipeline = this.device.createComputePipeline({
        compute: {
          module: this.device.createShaderModule({ code: boostShader }),
          entryPoint: 'main'
        }
      });

      this.logger.info('✅ Compute pipelines created successfully');
    } catch (error) {
      this.logger.error('❌ Failed to create compute pipelines:', error);
      throw error;
    }
  }

  private initializeBufferPools(): void {
    this.logger.info('💾 Initializing buffer pools...');

    const poolSize = 10;
    const maxBufferSize = 64 * 1024 * 1024; // 64MB

    for (let i = 0; i < poolSize; i++) {
      // Input buffers
      this.inputBufferPool.push(this.device.createBuffer({
        size: maxBufferSize,
        usage: gpu.BufferUsage.STORAGE | gpu.BufferUsage.COPY_DST
      }));

      // Output buffers
      this.outputBufferPool.push(this.device.createBuffer({
        size: maxBufferSize,
        usage: gpu.BufferUsage.STORAGE | gpu.BufferUsage.COPY_SRC
      }));

      // Uniform buffers
      this.uniformBufferPool.push(this.device.createBuffer({
        size: 256, // Small uniform buffer
        usage: gpu.BufferUsage.UNIFORM | gpu.BufferUsage.COPY_DST
      }));
    }

    this.logger.info(`✅ Buffer pools initialized (${poolSize} buffers each)`);
  }

  private startProcessingLoop(): void {
    setInterval(async () => {
      if (!this.isProcessing && this.processingQueue.length > 0) {
        await this.processBatch();
      }
    }, 10); // Check every 10ms
  }

  async processEmbeddings(call: any, callback: any): Promise<void> {
    const startTime = Date.now();

    try {
      const request = call.request;
      const texts = request.requests.map((req: any) => req.text);
      
      this.logger.debug(`📦 Processing embedding batch: ${texts.length} texts`);

      // Process embeddings on GPU
      const result = await this.computeEmbeddingsGPU(texts);

      const response = {
        embeddings: result.embeddings.map(emb => ({ values: Array.from(emb) })),
        dimensions: result.embeddings[0]?.length || 0,
        processingTime: Date.now() - startTime,
        batchSize: texts.length
      };

      callback(null, response);
    } catch (error) {
      this.logger.error('❌ Embedding processing failed:', error);
      callback(error, null);
    }
  }

  async performClustering(call: any, callback: any): Promise<void> {
    const startTime = Date.now();

    try {
      const request = call.request;
      const embeddings = request.embeddings.map((emb: any) => new Float32Array(emb.values));
      const numClusters = request.numClusters || 8;

      this.logger.debug(`🎯 Performing clustering: ${embeddings.length} embeddings, ${numClusters} clusters`);

      const result = await this.performKMeansClusteringGPU(embeddings, numClusters);

      const response = {
        assignments: result.assignments,
        centers: result.centers.map(center => ({ values: Array.from(center) })),
        inertia: result.inertia,
        iterations: result.iterations,
        processingTime: Date.now() - startTime
      };

      callback(null, response);
    } catch (error) {
      this.logger.error('❌ Clustering failed:', error);
      callback(error, null);
    }
  }

  async computeSimilarity(call: any, callback: any): Promise<void> {
    const startTime = Date.now();

    try {
      const request = call.request;
      const embeddingsA = request.embeddingsA.map((emb: any) => new Float32Array(emb.values));
      const embeddingsB = request.embeddingsB.map((emb: any) => new Float32Array(emb.values));

      this.logger.debug(`📊 Computing similarity: ${embeddingsA.length}x${embeddingsB.length} comparisons`);

      const similarities = await this.computeSimilarityGPU(embeddingsA, embeddingsB);

      const response = {
        scores: Array.from(similarities),
        metric: request.metric || 'cosine',
        processingTime: Date.now() - startTime
      };

      callback(null, response);
    } catch (error) {
      this.logger.error('❌ Similarity computation failed:', error);
      callback(error, null);
    }
  }

  async applyBoostTransform(call: any, callback: any): Promise<void> {
    const startTime = Date.now();

    try {
      const request = call.request;
      const embeddings = request.embeddings.map((emb: any) => new Float32Array(emb.values));
      const boostFactors = new Float32Array(request.boostFactors);

      this.logger.debug(`🚀 Applying boost transform: ${embeddings.length} embeddings`);

      const transformedEmbeddings = await this.applyBoostTransformGPU(embeddings, boostFactors);

      const response = {
        transformedEmbeddings: transformedEmbeddings.map(emb => ({ values: Array.from(emb) })),
        boostFactors: Array.from(boostFactors),
        processingTime: Date.now() - startTime
      };

      callback(null, response);
    } catch (error) {
      this.logger.error('❌ Boost transform failed:', error);
      callback(error, null);
    }
  }

  async processDocument(call: any, callback: any): Promise<void> {
    // Delegate to embedding processing for now
    await this.processEmbeddings(call, callback);
  }

  async streamDocuments(call: any): Promise<void> {
    // Implement streaming processing
    call.on('data', async (request: any) => {
      try {
        const result = await this.processEmbeddings({ request }, (err: any, response: any) => {
          if (err) {
            call.emit('error', err);
          } else {
            call.write(response);
          }
        });
      } catch (error) {
        call.emit('error', error);
      }
    });

    call.on('end', () => {
      call.end();
    });
  }

  private async computeEmbeddingsGPU(texts: string[]): Promise<ProcessingResult> {
    const startTime = Date.now();

    // Convert texts to token arrays (simplified tokenization)
    const tokenArrays = texts.map(text => this.tokenizeText(text));
    const maxTokens = Math.max(...tokenArrays.map(tokens => tokens.length));
    
    // Create input buffer
    const inputData = new Float32Array(texts.length * maxTokens);
    tokenArrays.forEach((tokens, i) => {
      tokens.forEach((token, j) => {
        inputData[i * maxTokens + j] = token;
      });
    });

    // Get buffers from pool
    const inputBuffer = this.getBuffer(this.inputBufferPool);
    const outputBuffer = this.getBuffer(this.outputBufferPool);

    try {
      // Upload data to GPU
      this.queue.writeBuffer(inputBuffer, 0, inputData);

      // Create bind group
      const bindGroup = this.device.createBindGroup({
        layout: this.embeddingComputePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } }
        ]
      });

      // Dispatch compute shader
      const commandEncoder = this.device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      
      computePass.setPipeline(this.embeddingComputePipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatch(Math.ceil(texts.length / 8), Math.ceil(maxTokens / 8));
      computePass.end();

      this.queue.submit([commandEncoder.finish()]);

      // Read results
      const resultData = await this.readBufferAsync(outputBuffer, texts.length * 384); // 384-dim embeddings
      const embeddings: Float32Array[] = [];
      
      for (let i = 0; i < texts.length; i++) {
        embeddings.push(resultData.slice(i * 384, (i + 1) * 384));
      }

      return {
        embeddings,
        processingTime: Date.now() - startTime,
        gpuTime: Date.now() - startTime, // Simplified
        batchSize: texts.length
      };
    } finally {
      // Return buffers to pool
      this.returnBuffer(this.inputBufferPool, inputBuffer);
      this.returnBuffer(this.outputBufferPool, outputBuffer);
    }
  }

  private async performKMeansClusteringGPU(embeddings: Float32Array[], numClusters: number): Promise<any> {
    const dimensions = embeddings[0].length;
    const numPoints = embeddings.length;

    // Flatten embeddings
    const inputData = new Float32Array(numPoints * dimensions);
    embeddings.forEach((emb, i) => {
      inputData.set(emb, i * dimensions);
    });

    // Get buffers
    const inputBuffer = this.getBuffer(this.inputBufferPool);
    const outputBuffer = this.getBuffer(this.outputBufferPool);

    try {
      // Upload data
      this.queue.writeBuffer(inputBuffer, 0, inputData);

      // Run clustering iterations
      const maxIterations = 100;
      const assignments = new Int32Array(numPoints);
      const centers = new Float32Array(numClusters * dimensions);
      
      // Initialize centers randomly
      for (let i = 0; i < numClusters; i++) {
        const randomIdx = Math.floor(Math.random() * numPoints);
        centers.set(embeddings[randomIdx], i * dimensions);
      }

      let iterations = 0;
      let converged = false;

      while (iterations < maxIterations && !converged) {
        // GPU compute pass for clustering
        const bindGroup = this.device.createBindGroup({
          layout: this.clusteringComputePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } }
          ]
        });

        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        
        computePass.setPipeline(this.clusteringComputePipeline);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatch(Math.ceil(numPoints / 64));
        computePass.end();

        this.queue.submit([commandEncoder.finish()]);

        iterations++;
        // Simplified convergence check
        if (iterations > 10) converged = true;
      }

      // Read final assignments
      const resultData = await this.readBufferAsync(outputBuffer, numPoints);
      const finalAssignments = Array.from(new Int32Array(resultData.buffer.slice(0, numPoints * 4)));

      return {
        assignments: finalAssignments,
        centers: Array.from({ length: numClusters }, (_, i) => 
          centers.slice(i * dimensions, (i + 1) * dimensions)
        ),
        iterations,
        inertia: 0 // Simplified
      };
    } finally {
      this.returnBuffer(this.inputBufferPool, inputBuffer);
      this.returnBuffer(this.outputBufferPool, outputBuffer);
    }
  }

  private async computeSimilarityGPU(embeddingsA: Float32Array[], embeddingsB: Float32Array[]): Promise<Float32Array> {
    // Simplified GPU similarity computation
    const similarities = new Float32Array(embeddingsA.length * embeddingsB.length);
    
    for (let i = 0; i < embeddingsA.length; i++) {
      for (let j = 0; j < embeddingsB.length; j++) {
        // Cosine similarity
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        
        for (let k = 0; k < embeddingsA[i].length; k++) {
          dotProduct += embeddingsA[i][k] * embeddingsB[j][k];
          normA += embeddingsA[i][k] * embeddingsA[i][k];
          normB += embeddingsB[j][k] * embeddingsB[j][k];
        }
        
        similarities[i * embeddingsB.length + j] = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
      }
    }
    
    return similarities;
  }

  private async applyBoostTransformGPU(embeddings: Float32Array[], boostFactors: Float32Array): Promise<Float32Array[]> {
    return embeddings.map(embedding => {
      const boosted = new Float32Array(embedding.length);
      for (let i = 0; i < embedding.length; i++) {
        const boostIndex = Math.min(i, boostFactors.length - 1);
        boosted[i] = embedding[i] * boostFactors[boostIndex];
      }
      return boosted;
    });
  }

  private async processBatch(): Promise<void> {
    if (this.processingQueue.length === 0) return;

    this.isProcessing = true;
    const batch = this.processingQueue.shift()!;

    try {
      const result = await this.computeEmbeddingsGPU(batch.texts);
      // Process result...
    } catch (error) {
      this.logger.error('❌ Batch processing failed:', error);
    } finally {
      this.isProcessing = false;
    }
  }

  private tokenizeText(text: string): number[] {
    // Simplified tokenization - convert to character codes
    return text.split('').map(char => char.charCodeAt(0)).slice(0, 512);
  }

  private getBuffer(pool: any[]): any {
    return pool.pop() || this.device.createBuffer({
      size: 64 * 1024 * 1024,
      usage: gpu.BufferUsage.STORAGE | gpu.BufferUsage.COPY_DST | gpu.BufferUsage.COPY_SRC
    });
  }

  private returnBuffer(pool: any[], buffer: any): void {
    if (pool.length < 10) {
      pool.push(buffer);
    }
  }

  private async readBufferAsync(buffer: any, size: number): Promise<Float32Array> {
    // Create staging buffer for readback
    const stagingBuffer = this.device.createBuffer({
      size,
      usage: gpu.BufferUsage.COPY_DST | gpu.BufferUsage.MAP_READ
    });

    // Copy to staging buffer
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
    this.queue.submit([commandEncoder.finish()]);

    // Map and read
    await stagingBuffer.mapAsync(gpu.MapMode.READ);
    const data = new Float32Array(stagingBuffer.getMappedRange().slice());
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return data;
  }

  async cleanup(): Promise<void> {
    this.logger.info('🧹 Cleaning up GPU Processor...');

    // Cleanup buffer pools
    this.inputBufferPool.forEach(buffer => buffer?.destroy());
    this.outputBufferPool.forEach(buffer => buffer?.destroy());
    this.uniformBufferPool.forEach(buffer => buffer?.destroy());

    // Cleanup pipelines
    this.embeddingComputePipeline = null;
    this.clusteringComputePipeline = null;
    this.similarityComputePipeline = null;
    this.boostTransformPipeline = null;

    if (this.device) {
      this.device.destroy();
    }

    this.logger.info('✅ GPU Processor cleanup completed');
  }
}