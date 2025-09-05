/**
 * 🖥️ WebGPU Worker for Browser-Side GPU Acceleration
 *
 * Handles:
 * - WebGPU compute shaders for tensor operations
 * - GPU memory management
 * - Parallel processing coordination
 * - RTX 3060 Ti optimization
 */

/// <reference lib="webworker" />

// Skip strict typing for WebGPU compatibility

export interface GPUWorkerMessage {
  type: 'process_gpu' | 'initialize_gpu' | 'cleanup_gpu';
  jobId: string;
  operation?: {
    type: string;
    input: any;
    metadata?: {
      legalWeight?: number;
      threshold?: number;
    };
  };
  config?: any;
}

class WebGPUWorker {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private initialized = false;
  private shaderCache = new Map<string, GPUShaderModule>();
  private pipelineCache = new Map<string, GPUComputePipeline>();

  constructor() {
    self.onmessage = this.handleMessage.bind(this);
  }

  private async handleMessage(event: MessageEvent<GPUWorkerMessage>) {
    const { type, jobId, operation, config } = event.data;

    try {
      switch (type) {
        case 'initialize_gpu':
          const success = await this.initializeGPU();
          self.postMessage({ jobId, result: { success }, error: null });
          break;

        case 'process_gpu':
          if (!this.initialized) {
            await this.initializeGPU();
          }
          const result = await this.processGPUOperation(operation);
          self.postMessage({ jobId, result, error: null });
          break;

        case 'cleanup_gpu':
          await this.cleanup();
          self.postMessage({ jobId, result: { cleaned: true }, error: null });
          break;

        default:
          throw new Error(`Unknown message type: ${type}`);
      }
    } catch (error: any) {
      self.postMessage({
        jobId,
        result: null,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }

  private async initializeGPU(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        throw new Error('WebGPU not supported');
      }

      // Request RTX 3060 Ti optimized adapter
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
        forceFallbackAdapter: false
      });

      if (!this.adapter) {
        throw new Error('No GPU adapter found');
      }

      // Request device with optimal settings for RTX 3060 Ti
      this.device = await this.adapter.requestDevice({
        requiredFeatures: [
          'timestamp-query',
          'texture-compression-bc',
          'float32-filterable'
        ] as GPUFeatureName[],
        requiredLimits: {
          maxBufferSize: 2147483648, // 2GB (within 8GB VRAM limit)
          maxStorageBufferBindingSize: 1073741824, // 1GB per binding
          maxComputeWorkgroupSizeX: 256,
          maxComputeWorkgroupSizeY: 256,
          maxComputeWorkgroupSizeZ: 64,
          maxComputeInvocationsPerWorkgroup: 512,
          maxComputeWorkgroupsPerDimension: 65535
        }
      });

      // Setup error handling
      this.device.lost.then((info) => {
        console.error(`WebGPU device lost: ${info.reason}`, info.message);
        this.initialized = false;
      });

      await this.loadOptimizedShaders();
      this.initialized = true;

      return true;
    } catch (error: any) {
      console.error('WebGPU initialization failed:', error);
      return false;
    }
  }

  private async loadOptimizedShaders(): Promise<any> {
    // Tensor processing shader optimized for legal documents
    const tensorShader = `
      struct TensorConfig {
        inputSize: u32,
        outputSize: u32,
        batchSize: u32,
        legalWeight: f32,
        similarity_threshold: f32,
        padding: vec3<f32>,
      }

      @group(0) @binding(0) var<storage, read> inputTensors: array<vec4<f32>>;
      @group(0) @binding(1) var<storage, read_write> outputTensors: array<vec4<f32>>;
      @group(0) @binding(2) var<uniform> config: TensorConfig;
      @group(0) @binding(3) var<storage, read> legalKeywords: array<u32>;

      // Optimized workgroup size for RTX 3060 Ti (4864 CUDA cores)
      @compute @workgroup_size(256, 1, 1)
      fn tensorProcess(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        if (index >= config.inputSize) {
          return;
        }

        let input = inputTensors[index];
        var output = vec4<f32>(0.0);

        // Legal document specific processing
        let legalScore = computeLegalScore(input, index);
        let semanticScore = computeSemanticScore(input);
        let temporalScore = computeTemporalScore(input, index);
        let contextScore = computeContextScore(input, index);

        output.x = input.x * legalScore * config.legalWeight;
        output.y = input.y * semanticScore;
        output.z = input.z * temporalScore;
        output.w = input.w * contextScore;

        // Apply similarity threshold
        if (length(output) > config.similarity_threshold) {
          outputTensors[index] = output;
        } else {
          outputTensors[index] = vec4<f32>(0.0);
        }
      }

      fn computeLegalScore(tensor: vec4<f32>, index: u32) -> f32 {
        var score = 1.0;

        // Check against legal keyword patterns
        let keywordHash = u32(tensor.x * 1000000.0) % arrayLength(&legalKeywords);
        let keyword = legalKeywords[keywordHash];

        // Boost score for legal terms
        if (keyword != 0u) {
          score *= 2.0;
        }

        // Legal document pattern recognition
        let pattern = (tensor.y + tensor.z) * 0.5;
        if (pattern > 0.7) {
          score *= 1.5; // Contract patterns
        } else if (pattern > 0.5) {
          score *= 1.3; // Regulatory patterns
        } else if (pattern > 0.3) {
          score *= 1.1; // General legal patterns
        }

        return clamp(score, 0.1, 3.0);
      }

      fn computeSemanticScore(tensor: vec4<f32>) -> f32 {
        // Semantic similarity computation
        let magnitude = length(tensor);
        let normalized = normalize(tensor);

        // Semantic patterns for legal documents
        let semanticWeight = dot(normalized, vec4<f32>(0.25, 0.25, 0.25, 0.25));

        return clamp(semanticWeight * magnitude, 0.0, 2.0);
      }

      fn computeTemporalScore(tensor: vec4<f32>, index: u32) -> f32 {
        // Temporal relevance based on document position
        let position = f32(index) / f32(config.inputSize);
        let decay = exp(-position * 0.1); // Exponential decay

        return tensor.z * decay;
      }

      fn computeContextScore(tensor: vec4<f32>, index: u32) -> f32 {
        // Context awareness for surrounding tokens
        let localContext = (tensor.x + tensor.y + tensor.z) / 3.0;
        let globalContext = f32(index) / f32(config.inputSize);

        return mix(localContext, globalContext, 0.3);
      }
    `;

    // Vector similarity shader
    const similarityShader = `
      struct SimilarityPair {
        vecA: vec4<f32>,
        vecB: vec4<f32>,
      }

      struct SimilarityResult {
        cosine: f32,
        euclidean: f32,
        manhattan: f32,
        jaccard: f32,
      }

      @group(0) @binding(0) var<storage, read> pairs: array<SimilarityPair>;
      @group(0) @binding(1) var<storage, read_write> results: array<SimilarityResult>;
      @group(0) @binding(2) var<uniform> numPairs: u32;

      @compute @workgroup_size(256, 1, 1)
      fn computeSimilarity(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        if (index >= numPairs) {
          return;
        }

        let pair = pairs[index];
        var result: SimilarityResult;

        // Cosine similarity
        let dotProd = dot(pair.vecA, pair.vecB);
        let normA = length(pair.vecA);
        let normB = length(pair.vecB);
        result.cosine = select(0.0, dotProd / (normA * normB), normA > 0.0 && normB > 0.0);

        // Euclidean distance
        let diff = pair.vecA - pair.vecB;
        result.euclidean = length(diff);

        // Manhattan distance
        let absDiff = abs(diff);
        result.manhattan = absDiff.x + absDiff.y + absDiff.z + absDiff.w;

        // Jaccard similarity (binary approximation)
        let intersection = dot(min(pair.vecA, pair.vecB), vec4<f32>(1.0));
        let union = dot(max(pair.vecA, pair.vecB), vec4<f32>(1.0));
        result.jaccard = select(0.0, intersection / union, union > 0.0);

        results[index] = result;
      }
    `;

    // K-means clustering shader
    const kmeansShader = `
      struct Point {
        position: vec4<f32>,
        clusterId: u32,
        confidence: f32,
        legalWeight: f32,
        reserved: f32,
      }

      struct Centroid {
        position: vec4<f32>,
        count: atomic<u32>,
      }

      @group(0) @binding(0) var<storage, read_write> points: array<Point>;
      @group(0) @binding(1) var<storage, read> centroids: array<Centroid>;
      @group(0) @binding(2) var<uniform> numPoints: u32;
      @group(0) @binding(3) var<uniform> numClusters: u32;

      @compute @workgroup_size(256, 1, 1)
      fn assignClusters(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        if (index >= numPoints) {
          return;
        }

        var point = points[index];
        var minDistance = 1000000.0;
        var bestCluster = 0u;
        var secondBestDistance = 1000000.0;

        // Find closest centroid with legal weighting
        for (var i = 0u; i < numClusters; i++) {
          let centroid = centroids[i].position;
          let distance = computeWeightedDistance(point.position, centroid, point.legalWeight);

          if (distance < minDistance) {
            secondBestDistance = minDistance;
            minDistance = distance;
            bestCluster = i;
          } else if (distance < secondBestDistance) {
            secondBestDistance = distance;
          }
        }

        // Calculate confidence based on distance ratios
        point.confidence = select(0.5, 1.0 - (minDistance / secondBestDistance), secondBestDistance > 0.0);
        point.clusterId = bestCluster;

        points[index] = point;
      }

      fn computeWeightedDistance(pointA: vec4<f32>, pointB: vec4<f32>, weight: f32) -> f32 {
        let diff = pointA - pointB;
        let weightedDiff = vec4<f32>(
          diff.x,
          diff.y,
          diff.z * weight, // Legal importance weighting
          diff.w
        );
        return dot(weightedDiff, weightedDiff);
      }
    `;

    // Create and cache shader modules
    if (this.device) {
      this.shaderCache.set('tensor', this.device.createShaderModule({ code: tensorShader }));
      this.shaderCache.set('similarity', this.device.createShaderModule({ code: similarityShader }));
      this.shaderCache.set('kmeans', this.device.createShaderModule({ code: kmeansShader }));
    }
  }

  private async processGPUOperation(operation: any): Promise<any> {
    if (!this.device) {
      throw new Error('GPU device not initialized');
    }

    const startTime = performance.now();

    switch (operation.type) {
      case 'embedding':
        return await this.processEmbedding(operation);
      case 'similarity':
        return await this.processSimilarity(operation);
      case 'clustering':
        return await this.processClustering(operation);
      case 'search':
        return await this.processSearch(operation);
      case 'transform':
        return await this.processTransform(operation);
      default:
        throw new Error(`Unsupported operation type: ${operation.type}`);
    }
  }

  private async processEmbedding(operation: any): Promise<Float32Array> {
    const pipeline = await this.getOrCreatePipeline('tensor');
    const input = new Float32Array(operation.input);

    // Create GPU buffers
    const inputBuffer = this.createBuffer(input, GPUBufferUsage.STORAGE);
    const outputBuffer = this.createBuffer(
      new Float32Array(input.length),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    const configData = new Float32Array([
      input.length / 4, // inputSize (assuming vec4 elements)
      input.length / 4, // outputSize
      1,                // batchSize
      operation.metadata?.legalWeight || 1.5, // legalWeight
      operation.metadata?.threshold || 0.5,   // similarity_threshold
      0, 0, 0           // padding
    ]);
    const configBuffer = this.createBuffer(configData, GPUBufferUsage.UNIFORM);

    // Legal keywords buffer
    const keywords = this.getLegalKeywords();
    const keywordBuffer = this.createBuffer(keywords, GPUBufferUsage.STORAGE);

    // Create bind group
    const bindGroup = this.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: configBuffer } },
        { binding: 3, resource: { buffer: keywordBuffer } }
      ]
    });

    // Execute compute pass
    const commandEncoder = this.device!.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil((input.length / 4) / 256));
    passEncoder.end();

    // Read results
    const outputByteLength = input.length * 4; // Float32Array bytes
    const readBuffer = this.createReadBuffer(outputByteLength);
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);

    this.device!.queue.submit([commandEncoder.finish()]);
    await this.device!.queue.onSubmittedWorkDone();

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange());
    const outputArray = new Float32Array(result);
    readBuffer.unmap();

    return outputArray;
  }

  private async processSimilarity(operation: any): Promise<any[]> {
    const pipeline = await this.getOrCreatePipeline('similarity');
    const { vectorsA, vectorsB } = operation.input;

    const numPairs = Math.min(vectorsA.length, vectorsB.length);
    const pairData = new Float32Array(numPairs * 8); // 2 vec4s per pair

    // Pack vector pairs
    for (let i = 0; i < numPairs; i++) {
      pairData.set(vectorsA[i], i * 8);
      pairData.set(vectorsB[i], i * 8 + 4);
    }

    const inputBuffer = this.createBuffer(pairData, GPUBufferUsage.STORAGE);
    const outputBuffer = this.createBuffer(
      new Float32Array(numPairs * 4), // 4 similarity metrics per pair
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const configBuffer = this.createBuffer(
      new Uint32Array([numPairs]),
      GPUBufferUsage.UNIFORM
    );

    const bindGroup = this.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: configBuffer } }
      ]
    });

    // Execute
    const commandEncoder = this.device!.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(numPairs / 256));
    passEncoder.end();

    // Read results
    const readBuffer = this.createReadBuffer(numPairs * 4 * 4);
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, numPairs * 4 * 4);

    this.device!.queue.submit([commandEncoder.finish()]);
    await this.device!.queue.onSubmittedWorkDone();

    await readBuffer.mapAsync(GPUMapMode.READ);
    const results = new Float32Array(readBuffer.getMappedRange());

    const similarities = [];
    for (let i = 0; i < numPairs; i++) {
      similarities.push({
        cosine: results[i * 4],
        euclidean: results[i * 4 + 1],
        manhattan: results[i * 4 + 2],
        jaccard: results[i * 4 + 3]
      });
    }

    readBuffer.unmap();
    return similarities;
  }

  private async processClustering(operation: any): Promise<any> {
    const pipeline = await this.getOrCreatePipeline('kmeans');
    // Implementation for K-means clustering
    return { clusters: [], centroids: [] };
  }

  private async processSearch(operation: any): Promise<any> {
    // Use tensor processing for search operations
    return await this.processEmbedding({ ...operation, type: 'embedding' });
  }

  private async processTransform(operation: any): Promise<any> {
    // Generic tensor transformation
    return await this.processEmbedding({ ...operation, type: 'embedding' });
  }

  private async getOrCreatePipeline(shaderName: string): Promise<GPUComputePipeline> {
    if (this.pipelineCache.has(shaderName)) {
      return this.pipelineCache.get(shaderName)!;
    }

    const shader = this.shaderCache.get(shaderName);
    if (!shader) {
      throw new Error(`Shader not found: ${shaderName}`);
    }

    const pipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shader,
        entryPoint: shaderName === 'tensor' ? 'tensorProcess' :
                   shaderName === 'similarity' ? 'computeSimilarity' : 'assignClusters'
      }
    });

    this.pipelineCache.set(shaderName, pipeline);
    return pipeline;
  }

  private createBuffer(data: ArrayBuffer | ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device!.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });

    if (data instanceof ArrayBuffer) {
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
    } else {
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    }

    buffer.unmap();
    return buffer;
  }

  private createReadBuffer(size: number): GPUBuffer {
    return this.device!.createBuffer({
      size: size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  }

  private getLegalKeywords(): Uint32Array {
    const keywords = [
      'contract', 'agreement', 'liability', 'breach', 'damages',
      'jurisdiction', 'statute', 'regulation', 'clause', 'provision',
      'warranty', 'indemnity', 'arbitration', 'precedent', 'citation'
    ];

    const hashes = keywords.map(keyword => {
      let hash = 0;
      for (let i = 0; i < keyword.length; i++) {
        hash = ((hash << 5) - hash + keyword.charCodeAt(i)) & 0xFFFFFFFF;
      }
      return hash >>> 0; // Ensure unsigned 32-bit
    });

    return new Uint32Array(hashes);
  }

  private async cleanup(): Promise<any> {
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this.adapter = null;
    this.initialized = false;
    this.shaderCache.clear();
    this.pipelineCache.clear();
  }
}

// Initialize worker
new WebGPUWorker();