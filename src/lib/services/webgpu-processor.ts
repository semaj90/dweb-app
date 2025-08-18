/**
 * WebGPU Processor Service
 * High-performance GPU computing for AI assistant
 * Implements: SOM training, embedding computations, vertex buffer encoding
 */

interface WebGPUProcessorConfig {
  preferredLimits?: {
    maxComputeWorkgroupSizeX?: number;
    maxComputeWorkgroupSizeY?: number;
    maxComputeWorkgroupsPerDimension?: number;
    maxStorageBufferBindingSize?: number;
  };
  enableProfiling?: boolean;
  fallbackToCPU?: boolean;
}

export class WebGPUProcessor {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private queue: GPUCommandQueue | null = null;
  private isInitialized = false;
  private config: WebGPUProcessorConfig;

  constructor(config: WebGPUProcessorConfig = {}) {
    this.config = {
      preferredLimits: {
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
        maxComputeWorkgroupsPerDimension: 65535,
        maxStorageBufferBindingSize: 134217728, // 128MB
      },
      enableProfiling: true,
      fallbackToCPU: true,
      ...config
    };
  }

  async initialize(): Promise<boolean> {
    if (this.isInitialized) return true;

    try {
      // Check WebGPU support
      if (!navigator.gpu) {
        throw new Error('WebGPU not supported');
      }

      // Request adapter
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });

      if (!this.adapter) {
        throw new Error('No WebGPU adapter found');
      }

      // Request device with required features
      this.device = await this.adapter.requestDevice({
        requiredLimits: this.config.preferredLimits,
        requiredFeatures: ['shader-f16'] as GPUFeatureName[]
      });

      this.queue = this.device.queue;
      this.isInitialized = true;

      console.log('✅ WebGPU processor initialized successfully');
      console.log('Adapter info:', this.adapter.info);
      console.log('Device limits:', this.device.limits);

      return true;
    } catch (error) {
      console.error('WebGPU initialization failed:', error);
      
      if (this.config.fallbackToCPU) {
        console.log('Falling back to CPU processing');
        return false; // Indicates CPU fallback
      }
      
      throw error;
    }
  }

  createBuffer(size: number, usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    if (!this.device) throw new Error('WebGPU not initialized');
    
    return this.device.createBuffer({
      size,
      usage
    });
  }

  createComputeShader(shaderCode: string): GPUShaderModule {
    if (!this.device) throw new Error('WebGPU not initialized');
    
    return this.device.createShaderModule({
      code: shaderCode
    });
  }

  /**
   * Train Self-Organizing Map on GPU
   * Implements Kohonen SOM algorithm with GPU acceleration
   */
  async trainSOM(
    embeddings: Float32Array, 
    gridWidth: number, 
    gridHeight: number, 
    iterations: number = 1000
  ): Promise<Float32Array> {
    if (!this.device || !this.queue) throw new Error('WebGPU not initialized');

    const embeddingDim = embeddings.length / (embeddings.length / 768); // Assume 768-dim embeddings
    const numDataPoints = embeddings.length / embeddingDim;
    const somSize = gridWidth * gridHeight * embeddingDim;

    // Create buffers
    const embeddingBuffer = this.createBuffer(embeddings.byteLength);
    const somBuffer = this.createBuffer(somSize * 4); // Float32
    const metadataBuffer = this.createBuffer(64); // For iteration counter, learning rate, etc.

    // Upload initial data
    this.queue.writeBuffer(embeddingBuffer, 0, embeddings);
    
    // Initialize SOM with random weights
    const initialSOM = new Float32Array(somSize);
    for (let i = 0; i < somSize; i++) {
      initialSOM[i] = Math.random() * 2 - 1; // Random values between -1 and 1
    }
    this.queue.writeBuffer(somBuffer, 0, initialSOM);

    // Create compute shader for SOM training
    const somShader = this.createComputeShader(`
      @group(0) @binding(0) var<storage, read> embeddings: array<f32>;
      @group(0) @binding(1) var<storage, read_write> som: array<f32>;
      @group(0) @binding(2) var<storage, read> metadata: array<f32>;
      
      const GRID_WIDTH: u32 = ${gridWidth}u;
      const GRID_HEIGHT: u32 = ${gridHeight}u;
      const EMBEDDING_DIM: u32 = ${embeddingDim}u;
      const NUM_DATA_POINTS: u32 = ${numDataPoints}u;
      
      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let x = global_id.x;
        let y = global_id.y;
        
        if (x >= GRID_WIDTH || y >= GRID_HEIGHT) {
          return;
        }
        
        let iteration = u32(metadata[0]);
        let learning_rate = metadata[1] * exp(-f32(iteration) / 1000.0);
        let neighborhood_radius = metadata[2] * exp(-f32(iteration) / 500.0);
        
        // Select random data point
        let data_idx = (iteration * 1664525u + 1013904223u) % NUM_DATA_POINTS;
        
        // Find Best Matching Unit (BMU)
        var best_distance = 1e10;
        var bmu_x = 0u;
        var bmu_y = 0u;
        
        for (var i = 0u; i < GRID_WIDTH; i++) {
          for (var j = 0u; j < GRID_HEIGHT; j++) {
            var distance = 0.0;
            
            for (var d = 0u; d < EMBEDDING_DIM; d++) {
              let som_idx = (i * GRID_HEIGHT + j) * EMBEDDING_DIM + d;
              let data_idx_full = data_idx * EMBEDDING_DIM + d;
              let diff = som[som_idx] - embeddings[data_idx_full];
              distance += diff * diff;
            }
            
            if (distance < best_distance) {
              best_distance = distance;
              bmu_x = i;
              bmu_y = j;
            }
          }
        }
        
        // Update weights based on distance to BMU
        let grid_distance = sqrt(f32((x - bmu_x) * (x - bmu_x) + (y - bmu_y) * (y - bmu_y)));
        let influence = exp(-(grid_distance * grid_distance) / (2.0 * neighborhood_radius * neighborhood_radius));
        
        if (influence > 0.01) {
          for (var d = 0u; d < EMBEDDING_DIM; d++) {
            let som_idx = (x * GRID_HEIGHT + y) * EMBEDDING_DIM + d;
            let data_idx_full = data_idx * EMBEDDING_DIM + d;
            let delta = learning_rate * influence * (embeddings[data_idx_full] - som[som_idx]);
            som[som_idx] += delta;
          }
        }
      }
    `);

    // Create compute pipeline
    const computePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: somShader,
        entryPoint: 'main'
      }
    });

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: embeddingBuffer } },
        { binding: 1, resource: { buffer: somBuffer } },
        { binding: 2, resource: { buffer: metadataBuffer } }
      ]
    });

    // Training loop
    for (let iteration = 0; iteration < iterations; iteration++) {
      // Update metadata
      const metadata = new Float32Array([
        iteration,
        0.1, // Initial learning rate
        Math.min(gridWidth, gridHeight) / 2 // Initial neighborhood radius
      ]);
      this.queue.writeBuffer(metadataBuffer, 0, metadata);

      // Dispatch compute shader
      const commandEncoder = this.device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(
        Math.ceil(gridWidth / 8),
        Math.ceil(gridHeight / 8)
      );
      
      computePass.end();
      this.queue.submit([commandEncoder.finish()]);

      // Log progress every 100 iterations
      if (iteration % 100 === 0) {
        console.log(`SOM training progress: ${iteration}/${iterations}`);
      }
    }

    // Read back results
    const resultBuffer = this.createBuffer(somSize * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(somBuffer, 0, resultBuffer, 0, somSize * 4);
    this.queue.submit([commandEncoder.finish()]);

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(resultBuffer.getMappedRange().slice(0));
    resultBuffer.unmap();

    // Cleanup
    embeddingBuffer.destroy();
    somBuffer.destroy();
    metadataBuffer.destroy();
    resultBuffer.destroy();

    console.log('✅ SOM training completed on GPU');
    return result;
  }

  /**
   * Compute document embeddings similarity matrix
   */
  async computeSimilarityMatrix(embeddings: Float32Array): Promise<Float32Array> {
    if (!this.device || !this.queue) throw new Error('WebGPU not initialized');

    const embeddingDim = 768; // Assume 768-dim embeddings
    const numDocs = embeddings.length / embeddingDim;
    const matrixSize = numDocs * numDocs;

    // Create buffers
    const embeddingBuffer = this.createBuffer(embeddings.byteLength);
    const similarityBuffer = this.createBuffer(matrixSize * 4);

    this.queue.writeBuffer(embeddingBuffer, 0, embeddings);

    // Create compute shader for similarity computation
    const similarityShader = this.createComputeShader(`
      @group(0) @binding(0) var<storage, read> embeddings: array<f32>;
      @group(0) @binding(1) var<storage, read_write> similarity: array<f32>;
      
      const EMBEDDING_DIM: u32 = ${embeddingDim}u;
      const NUM_DOCS: u32 = ${numDocs}u;
      
      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let i = global_id.x;
        let j = global_id.y;
        
        if (i >= NUM_DOCS || j >= NUM_DOCS) {
          return;
        }
        
        var dot_product = 0.0;
        var norm_i = 0.0;
        var norm_j = 0.0;
        
        for (var d = 0u; d < EMBEDDING_DIM; d++) {
          let val_i = embeddings[i * EMBEDDING_DIM + d];
          let val_j = embeddings[j * EMBEDDING_DIM + d];
          
          dot_product += val_i * val_j;
          norm_i += val_i * val_i;
          norm_j += val_j * val_j;
        }
        
        // Cosine similarity
        let cosine_sim = dot_product / (sqrt(norm_i) * sqrt(norm_j));
        similarity[i * NUM_DOCS + j] = cosine_sim;
      }
    `);

    const computePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: similarityShader,
        entryPoint: 'main'
      }
    });

    const bindGroup = this.device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: embeddingBuffer } },
        { binding: 1, resource: { buffer: similarityBuffer } }
      ]
    });

    // Execute computation
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(
      Math.ceil(numDocs / 16),
      Math.ceil(numDocs / 16)
    );
    
    computePass.end();
    this.queue.submit([commandEncoder.finish()]);

    // Read results
    const resultBuffer = this.createBuffer(matrixSize * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    
    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(similarityBuffer, 0, resultBuffer, 0, matrixSize * 4);
    this.queue.submit([copyEncoder.finish()]);

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(resultBuffer.getMappedRange().slice(0));
    resultBuffer.unmap();

    // Cleanup
    embeddingBuffer.destroy();
    similarityBuffer.destroy();
    resultBuffer.destroy();

    return result;
  }

  /**
   * Encode multi-dimensional data into image texture for efficient transfer
   */
  encodeDataToTexture(data: Float32Array, width: number, height: number): ImageData {
    const imageData = new ImageData(width, height);
    const pixels = imageData.data;

    for (let i = 0; i < data.length && i * 4 < pixels.length; i++) {
      const value = data[i];
      
      // Encode float32 into RGBA channels using bit manipulation
      const floatView = new Float32Array([value]);
      const intView = new Uint32Array(floatView.buffer);
      const bits = intView[0];

      pixels[i * 4] = (bits >>> 24) & 0xFF; // R
      pixels[i * 4 + 1] = (bits >>> 16) & 0xFF; // G
      pixels[i * 4 + 2] = (bits >>> 8) & 0xFF; // B
      pixels[i * 4 + 3] = bits & 0xFF; // A
    }

    return imageData;
  }

  /**
   * Decode image texture back to multi-dimensional data
   */
  decodeTextureToData(imageData: ImageData): Float32Array {
    const pixels = imageData.data;
    const result = new Float32Array(pixels.length / 4);

    for (let i = 0; i < result.length; i++) {
      const r = pixels[i * 4];
      const g = pixels[i * 4 + 1];
      const b = pixels[i * 4 + 2];
      const a = pixels[i * 4 + 3];

      // Reconstruct float32 from RGBA channels
      const bits = (r << 24) | (g << 16) | (b << 8) | a;
      const intView = new Uint32Array([bits]);
      const floatView = new Float32Array(intView.buffer);
      
      result[i] = floatView[0];
    }

    return result;
  }

  destroy() {
    this.device?.destroy();
    this.isInitialized = false;
  }
}

// Singleton instance
export const webGPUProcessor = new WebGPUProcessor({
  enableProfiling: true,
  fallbackToCPU: true
});