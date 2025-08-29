// gemma3-wasm-bridge.js
// WebAssembly bridge for Gemma 3 Legal Model with GPU acceleration
// Integrates with your existing SvelteKit frontend

import { LRUCache } from 'lru-cache';
import { MessagePack } from '@msgpack/msgpack';

/**
 * Gemma3 WebAssembly Bridge
 * Provides browser-based inference with GPU acceleration via WebGPU
 */
export class Gemma3WASMBridge {
  constructor(config = {}) {
    this.config = {
      modelPath: config.modelPath || '/models/gemma3-legal-q4.wasm',
      weightsPath: config.weightsPath || '/models/gemma3-legal-weights.bin',
      vocabPath: config.vocabPath || '/models/gemma3-vocab.json',
      maxContextLength: config.maxContextLength || 4096,
      maxGenerationLength: config.maxGenerationLength || 2000,
      temperature: config.temperature || 0.1,
      topK: config.topK || 40,
      topP: config.topP || 0.9,
      repeatPenalty: config.repeatPenalty || 1.1,
      cacheSize: config.cacheSize || 100,
      useWebGPU: config.useWebGPU !== false,
      useSimd: config.useSimd !== false,
      useThreads: config.useThreads !== false,
      numThreads: config.numThreads || navigator.hardwareConcurrency || 4
    };

    this.wasmModule = null;
    this.model = null;
    this.tokenizer = null;
    this.vocab = null;
    this.webgpuDevice = null;
    this.initialized = false;

    // Performance monitoring
    this.metrics = {
      tokensProcessed: 0,
      inferenceTime: 0,
      cacheHits: 0,
      cacheMisses: 0
    };

    // Initialize cache
    this.cache = new LRUCache({
      max: this.config.cacheSize,
      ttl: 1000 * 60 * 60, // 1 hour TTL
      updateAgeOnGet: true
    });

    // Worker pool for parallel processing
    this.workerPool = [];
    this.taskQueue = [];
    this.activeWorkers = 0;
  }

  /**
   * Initialize the WASM model and WebGPU
   */
  async initialize() {
    try {
      console.log('Initializing Gemma3 WASM bridge...');
      
      // Check for WebGPU support
      if (this.config.useWebGPU && 'gpu' in navigator) {
        await this.initializeWebGPU();
      } else {
        console.warn('WebGPU not available, falling back to CPU');
        this.config.useWebGPU = false;
      }

      // Load WASM module with SIMD support
      await this.loadWASMModule();

      // Load model weights
      await this.loadModelWeights();

      // Load vocabulary
      await this.loadVocabulary();

      // Initialize tokenizer
      this.initializeTokenizer();

      // Initialize worker pool
      await this.initializeWorkerPool();

      this.initialized = true;
      console.log('Gemma3 WASM bridge initialized successfully');
      
      return true;
    } catch (error) {
      console.error('Failed to initialize Gemma3 WASM:', error);
      throw error;
    }
  }

  /**
   * Initialize WebGPU for GPU acceleration
   */
  async initializeWebGPU() {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });

    if (!adapter) {
      throw new Error('WebGPU adapter not available');
    }

    this.webgpuDevice = await adapter.requestDevice({
      requiredFeatures: ['shader-f16'],
      requiredLimits: {
        maxBufferSize: 2 * 1024 * 1024 * 1024, // 2GB
        maxStorageBufferBindingSize: 1 * 1024 * 1024 * 1024, // 1GB
        maxComputeWorkgroupStorageSize: 32768,
        maxComputeInvocationsPerWorkgroup: 1024
      }
    });

    console.log('WebGPU initialized:', {
      vendor: adapter.vendor,
      architecture: adapter.architecture,
      device: adapter.device,
      description: adapter.description
    });
  }

  /**
   * Load WASM module with feature detection
   */
  async loadWASMModule() {
    const wasmFeatures = {
      simd: this.config.useSimd && WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11])),
      threads: this.config.useThreads && typeof SharedArrayBuffer !== 'undefined',
      bulkMemory: true,
      multiValue: true
    };

    console.log('WASM features:', wasmFeatures);

    const response = await fetch(this.config.modelPath);
    const wasmBytes = await response.arrayBuffer();

    const importObject = {
      env: {
        memory: new WebAssembly.Memory({
          initial: 256,
          maximum: 16384,
          shared: wasmFeatures.threads
        }),
        table: new WebAssembly.Table({
          initial: 0,
          element: 'anyfunc'
        }),
        __wbindgen_throw: (ptr, len) => {
          throw new Error(this.getStringFromWasm(ptr, len));
        },
        // GPU acceleration functions
        gpu_matmul_f16: this.gpuMatMulF16.bind(this),
        gpu_attention: this.gpuAttention.bind(this),
        gpu_layer_norm: this.gpuLayerNorm.bind(this),
        gpu_gelu: this.gpuGelu.bind(this),
        // Threading functions
        spawn_worker: this.spawnWorker.bind(this),
        join_worker: this.joinWorker.bind(this),
        // Memory functions
        malloc: (size) => this.wasmModule.exports.malloc(size),
        free: (ptr) => this.wasmModule.exports.free(ptr)
      },
      wasi_snapshot_preview1: {
        proc_exit: () => {},
        fd_write: () => 0,
        fd_read: () => 0,
        fd_close: () => 0,
        fd_seek: () => 0,
        environ_sizes_get: () => 0,
        environ_get: () => 0,
        clock_time_get: () => Date.now(),
        random_get: (ptr, len) => {
          const view = new Uint8Array(this.wasmModule.exports.memory.buffer, ptr, len);
          crypto.getRandomValues(view);
          return 0;
        }
      }
    };

    const wasmModule = await WebAssembly.instantiate(wasmBytes, importObject);
    this.wasmModule = wasmModule.instance;

    // Initialize model in WASM
    this.wasmModule.exports.init_model();
  }

  /**
   * Load model weights
   */
  async loadModelWeights() {
    const response = await fetch(this.config.weightsPath);
    const weightsBuffer = await response.arrayBuffer();
    
    // Allocate memory in WASM for weights
    const weightsPtr = this.wasmModule.exports.malloc(weightsBuffer.byteLength);
    const weightsView = new Uint8Array(
      this.wasmModule.exports.memory.buffer,
      weightsPtr,
      weightsBuffer.byteLength
    );
    
    // Copy weights to WASM memory
    weightsView.set(new Uint8Array(weightsBuffer));
    
    // Load weights into model
    this.wasmModule.exports.load_weights(weightsPtr, weightsBuffer.byteLength);
    
    console.log(`Loaded ${(weightsBuffer.byteLength / 1024 / 1024).toFixed(2)}MB of model weights`);
  }

  /**
   * Load vocabulary for tokenization
   */
  async loadVocabulary() {
    const response = await fetch(this.config.vocabPath);
    this.vocab = await response.json();
    console.log(`Loaded vocabulary with ${Object.keys(this.vocab).length} tokens`);
  }

  /**
   * Initialize tokenizer
   */
  initializeTokenizer() {
    this.tokenizer = {
      encode: (text) => this.tokenize(text),
      decode: (tokens) => this.detokenize(tokens)
    };
  }

  /**
   * Process legal text with the model
   */
  async processLegalText(text, options = {}) {
    if (!this.initialized) {
      throw new Error('Model not initialized');
    }

    const startTime = performance.now();

    // Check cache
    const cacheKey = this.getCacheKey(text, options);
    const cached = this.cache.get(cacheKey);
    if (cached) {
      this.metrics.cacheHits++;
      return cached;
    }
    this.metrics.cacheMisses++;

    try {
      // Prepare input
      const prompt = this.prepareLegalPrompt(text, options);
      const tokens = await this.tokenizer.encode(prompt);

      // Check token limit
      if (tokens.length > this.config.maxContextLength) {
        console.warn('Input exceeds context length, truncating');
        tokens.length = this.config.maxContextLength - 512;
      }

      // Run inference
      const outputTokens = await this.generate(tokens, {
        maxLength: options.maxLength || this.config.maxGenerationLength,
        temperature: options.temperature || this.config.temperature,
        topK: options.topK || this.config.topK,
        topP: options.topP || this.config.topP,
        repeatPenalty: options.repeatPenalty || this.config.repeatPenalty
      });

      // Decode output
      const output = await this.tokenizer.decode(outputTokens);

      // Parse legal entities and citations
      const analysis = this.parseLegalOutput(output);

      const result = {
        text: output,
        analysis,
        tokens: outputTokens.length,
        processingTime: performance.now() - startTime
      };

      // Cache result
      this.cache.set(cacheKey, result);

      // Update metrics
      this.metrics.tokensProcessed += outputTokens.length;
      this.metrics.inferenceTime += result.processingTime;

      return result;

    } catch (error) {
      console.error('Error processing legal text:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings for vector search
   */
  async generateEmbeddings(text) {
    if (!this.initialized) {
      throw new Error('Model not initialized');
    }

    const tokens = await this.tokenizer.encode(text);
    
    // Allocate memory for tokens
    const tokensPtr = this.wasmModule.exports.malloc(tokens.length * 4);
    const tokensView = new Int32Array(
      this.wasmModule.exports.memory.buffer,
      tokensPtr,
      tokens.length
    );
    tokensView.set(tokens);

    // Generate embeddings
    const embeddingsPtr = this.wasmModule.exports.generate_embeddings(
      tokensPtr,
      tokens.length
    );

    // Read embeddings from WASM memory
    const embeddingSize = 768; // nomic-embed-text compatible
    const embeddings = new Float32Array(
      this.wasmModule.exports.memory.buffer,
      embeddingsPtr,
      embeddingSize
    );

    // Clean up
    this.wasmModule.exports.free(tokensPtr);
    this.wasmModule.exports.free(embeddingsPtr);

    return Array.from(embeddings);
  }

  /**
   * Initialize worker pool for parallel processing
   */
  async initializeWorkerPool() {
    const workerCount = Math.min(4, this.config.numThreads);
    
    for (let i = 0; i < workerCount; i++) {
      const worker = new Worker('/workers/gemma3-worker.js', {
        type: 'module'
      });

      worker.postMessage({
        type: 'init',
        config: this.config
      });

      await new Promise((resolve) => {
        worker.onmessage = (e) => {
          if (e.data.type === 'ready') {
            resolve();
          }
        };
      });

      this.workerPool.push(worker);
    }

    console.log(`Initialized ${workerCount} workers`);
  }

  /**
   * GPU-accelerated matrix multiplication
   */
  async gpuMatMulF16(aPtr, bPtr, cPtr, m, n, k) {
    if (!this.webgpuDevice) {
      return false; // Fall back to CPU
    }

    // Create GPU buffers
    const aBuffer = this.createGPUBuffer(aPtr, m * k * 2); // f16
    const bBuffer = this.createGPUBuffer(bPtr, k * n * 2);
    const cBuffer = this.createGPUBuffer(cPtr, m * n * 2, 'storage');

    // Create compute pipeline
    const computePipeline = this.webgpuDevice.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.webgpuDevice.createShaderModule({
          code: `
            @group(0) @binding(0) var<storage, read> a: array<f16>;
            @group(0) @binding(1) var<storage, read> b: array<f16>;
            @group(0) @binding(2) var<storage, read_write> c: array<f16>;

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
              let row = global_id.x;
              let col = global_id.y;
              
              if (row >= ${m}u || col >= ${n}u) {
                return;
              }

              var sum = f16(0.0);
              for (var i = 0u; i < ${k}u; i++) {
                sum += a[row * ${k}u + i] * b[i * ${n}u + col];
              }
              
              c[row * ${n}u + col] = sum;
            }
          `
        }),
        entryPoint: 'main'
      }
    });

    // Execute compute pass
    const commandEncoder = this.webgpuDevice.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, this.webgpuDevice.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: cBuffer } }
      ]
    }));
    
    passEncoder.dispatchWorkgroups(Math.ceil(m / 16), Math.ceil(n / 16));
    passEncoder.end();
    
    this.webgpuDevice.queue.submit([commandEncoder.finish()]);
    
    // Copy result back to WASM memory
    await this.copyGPUBufferToWASM(cBuffer, cPtr, m * n * 2);
    
    return true;
  }

  /**
   * Prepare legal prompt with system instructions
   */
  prepareLegalPrompt(text, options) {
    const systemPrompt = options.systemPrompt || `You are a legal AI assistant trained on case law, statutes, and legal documents.
Provide accurate, detailed legal analysis while noting this is not legal advice.
Focus on: jurisdiction, applicable laws, precedents, legal reasoning, and potential outcomes.`;

    return `${systemPrompt}\n\nUser Query: ${text}\n\nLegal Analysis:`;
  }

  /**
   * Parse legal output for entities and citations
   */
  parseLegalOutput(output) {
    const analysis = {
      entities: [],
      citations: [],
      statutes: [],
      concepts: [],
      jurisdiction: null,
      confidence: 0
    };

    // Extract case citations (e.g., "123 F.3d 456")
    const citationRegex = /\b\d+\s+[A-Z]+\.?\d*[a-z]*\s+\d+/g;
    analysis.citations = output.match(citationRegex) || [];

    // Extract statutes (e.g., "18 U.S.C. § 1001")
    const statuteRegex = /\b\d+\s+[A-Z.]+\s*§+\s*\d+/g;
    analysis.statutes = output.match(statuteRegex) || [];

    // Extract legal concepts
    const legalConcepts = [
      'breach of contract', 'negligence', 'liability', 'damages',
      'jurisdiction', 'standing', 'precedent', 'stare decisis',
      'mens rea', 'actus reus', 'due process', 'equal protection'
    ];
    
    analysis.concepts = legalConcepts.filter(concept => 
      output.toLowerCase().includes(concept)
    );

    // Estimate confidence based on citations and reasoning
    analysis.confidence = Math.min(
      0.5 + (analysis.citations.length * 0.1) + (analysis.concepts.length * 0.05),
      0.95
    );

    return analysis;
  }

  /**
   * Helper functions
   */
  tokenize(text) {
    // Simplified tokenization - in production, use proper BPE tokenizer
    const tokens = [];
    const words = text.toLowerCase().split(/\s+/);
    
    for (const word of words) {
      if (this.vocab[word]) {
        tokens.push(this.vocab[word]);
      } else {
        // Handle unknown words
        tokens.push(this.vocab['<unk>'] || 0);
      }
    }
    
    return tokens;
  }

  detokenize(tokens) {
    const reverseVocab = Object.fromEntries(
      Object.entries(this.vocab).map(([k, v]) => [v, k])
    );
    
    return tokens.map(t => reverseVocab[t] || '<unk>').join(' ');
  }

  getCacheKey(text, options) {
    const data = { text, ...options };
    return JSON.stringify(data);
  }

  createGPUBuffer(ptr, size, usage = 'read-only storage') {
    const buffer = this.webgpuDevice.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true
    });

    const arrayBuffer = buffer.getMappedRange();
    const view = new Uint8Array(arrayBuffer);
    const wasmView = new Uint8Array(
      this.wasmModule.exports.memory.buffer,
      ptr,
      size
    );
    view.set(wasmView);
    buffer.unmap();

    return buffer;
  }

  async copyGPUBufferToWASM(buffer, ptr, size) {
    const staging = this.webgpuDevice.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const commandEncoder = this.webgpuDevice.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.webgpuDevice.queue.submit([commandEncoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(staging.getMappedRange());
    
    const wasmView = new Uint8Array(
      this.wasmModule.exports.memory.buffer,
      ptr,
      size
    );
    wasmView.set(data);
    
    staging.unmap();
    staging.destroy();
  }

  /**
   * Clean up resources
   */
  dispose() {
    // Clean up workers
    for (const worker of this.workerPool) {
      worker.terminate();
    }
    this.workerPool = [];

    // Clean up WASM
    if (this.wasmModule) {
      this.wasmModule.exports.cleanup();
    }

    // Clean up WebGPU
    if (this.webgpuDevice) {
      this.webgpuDevice.destroy();
    }

    // Clear cache
    this.cache.clear();

    this.initialized = false;
  }
}

// Export for use in SvelteKit
export default Gemma3WASMBridge;
