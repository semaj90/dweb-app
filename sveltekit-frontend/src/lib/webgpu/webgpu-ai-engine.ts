/**
 * WebGPU AI Engine - Browser-based GPU acceleration
 * Extends API with custom compute shaders for dimensional arrays
 * Creates own optimized library for modular AI experiences
 */

/// <reference path="../types/webgpu.d.ts" />

export interface WebGPUCapabilities {
  isSupported: boolean;
  adapter?: GPUAdapter;
  device?: GPUDevice;
  features: string[];
  limits: Record<string, number>;
}

export interface ComputeShaderConfig {
  workgroupSize: [number, number, number];
  entryPoint: string;
  bindingLayout: GPUBindGroupLayoutDescriptor;
}

export interface AIComputeJob {
  id: string;
  type: 'attention' | 't5_inference' | 'dimensional_transform' | 'kernel_splice';
  inputData: Float32Array;
  shape: number[];
  attentionWeights?: Float32Array;
  modelParams?: any;
  priority: 'high' | 'medium' | 'low';
  createdAt: number;
}

export class WebGPUAIEngine {
  private capabilities: WebGPUCapabilities | null = null;
  private computeJobs = new Map<string, AIComputeJob>();
  private shaderCache = new Map<string, GPUComputePipeline>();
  private bufferPool: GPUBuffer[] = [];
  private isInitialized = false;
  private initPromise: Promise<void> | null = null;

  /**
   * Constructor optionally triggers auto init in browser only.
   * Heavy async work is deferred via init()/waitForReady() to avoid SSR issues.
   */
  constructor(autoInit = true) {
    if (autoInit && typeof navigator !== 'undefined' && 'gpu' in navigator) {
      this.initPromise = this.initializeWebGPU();
    } else if (typeof navigator === 'undefined') {
      // SSR environment – mark unsupported but defer real detection to client
      this.capabilities = { isSupported: false, features: [], limits: {} };
    } else if (!(navigator as any).gpu) {
      console.log('⚠️ WebGPU not available in this browser context yet – will remain in CPU fallback');
      this.capabilities = { isSupported: false, features: [], limits: {} };
    }
  }
  /** public lazy initialization */
  init(): Promise<void> {
    if (!this.initPromise) {
      if (typeof navigator === 'undefined' || !(navigator as any).gpu) {
        this.capabilities = { isSupported: false, features: [], limits: {} };
        this.initPromise = Promise.resolve();
      } else {
        this.initPromise = this.initializeWebGPU();
      }
    }
    return this.initPromise;
  }

  /** whether GPU path is ready */
  isReady(): boolean {
    return !!(this.isInitialized && this.capabilities?.isSupported);
  }

  /** await readiness (with timeout) */
  async waitForReady(timeoutMs = 5000): Promise<boolean> {
    try {
      await Promise.race([
        this.init(),
        new Promise((_r, reject) => setTimeout(() => reject(new Error('WebGPU init timeout')), timeoutMs))
      ]);
      return this.isReady();
    } catch (e) {
      return false;
    }
  }

  /**
   * Initialize WebGPU with feature detection
   */
  async initializeWebGPU(): Promise<void> {
    if (typeof navigator === 'undefined' || !(navigator as any).gpu) {
    // Not in a browser / not supported
      this.capabilities = { isSupported: false, features: [], limits: {} };
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });

      if (!adapter) {
        throw new Error('No WebGPU adapter available');
      }

      const device = await adapter.requestDevice({
        requiredFeatures: ['shader-f16'] as any[], // If supported
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupSizeX: 1024,
          maxComputeWorkgroupSizeY: 1024
        }
      });

      // GPUSupportedFeatures is iterable but not typed as standard Iterable<string> in some TS lib versions – coerce manually
      const featureList: string[] = [];
      try {
        for (const f of (adapter.features as any)) featureList.push(String(f));
      } catch {
        // Fallback if iteration fails
        (adapter.features as any as string[]).forEach?.((f: string) => featureList.push(f));
      }

      this.capabilities = {
        isSupported: true,
        adapter,
        device,
        features: featureList,
        limits: adapter.limits as any
      };

      this.isInitialized = true;
      console.log('🎮 WebGPU initialized successfully');
      console.log('Features:', this.capabilities.features);
      // Dispatch a custom event so UI can react
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('webgpu:ready', { detail: this.capabilities }));
      }

    } catch (error: any) {
      console.error('WebGPU initialization failed:', error);
      this.capabilities = { isSupported: false, features: [], limits: {} };
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('webgpu:failed', { detail: { error: String(error) } }));
      }
    }
  }

  /**
   * Create optimized compute shader for kernel attention
   */
  private createKernelAttentionShader(): string {
    return `
      struct Params {
        inputSize: u32,
        outputSize: u32,
        kernelSize: u32,
        attentionHeads: u32,
      };

      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read> attentionWeights: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;
      @group(0) @binding(3) var<uniform> params: Params;

      @compute @workgroup_size(64, 1, 1)
      fn kernelAttention(@builtin(global_invocation_id) globalId: vec3<u32>) {
        let index = globalId.x;
        if (index >= params.outputSize) { return; }

        var sum: f32 = 0.0;
        var weightSum: f32 = 0.0;

        // Kernel attention computation
        for (var i: u32 = 0u; i < params.kernelSize; i++) {
          let inputIndex = (index * params.kernelSize + i) % params.inputSize;
          let attentionIndex = i % arrayLength(&attentionWeights);

          let value = input[inputIndex];
          let weight = attentionWeights[attentionIndex];

          sum += value * weight;
          weightSum += weight;
        }

        // Normalize output
        output[index] = select(0.0, sum / weightSum, weightSum > 0.0);
      }
    `;
  }

  /**
   * Create T5-style transformer shader
   */
  private createT5TransformerShader(): string {
    return `
      struct T5Params {
        sequenceLength: u32,
        hiddenSize: u32,
        numHeads: u32,
        headDim: u32,
      };

      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read> queryWeights: array<f32>;
      @group(0) @binding(2) var<storage, read> keyWeights: array<f32>;
      @group(0) @binding(3) var<storage, read> valueWeights: array<f32>;
      @group(0) @binding(4) var<storage, read_write> output: array<f32>;
      @group(0) @binding(5) var<uniform> params: T5Params;

      @compute @workgroup_size(32, 1, 1)
      fn t5Attention(@builtin(global_invocation_id) globalId: vec3<u32>) {
        let seqIdx = globalId.x;
        if (seqIdx >= params.sequenceLength) { return; }

        let hiddenIdx = globalId.y;
        if (hiddenIdx >= params.hiddenSize) { return; }

        let inputOffset = seqIdx * params.hiddenSize + hiddenIdx;

        // Simplified T5 attention computation
        var attentionSum: f32 = 0.0;

        for (var i: u32 = 0u; i < params.sequenceLength; i++) {
          let keyIdx = i * params.hiddenSize + hiddenIdx;
          let queryValue = input[inputOffset] * queryWeights[hiddenIdx];
          let keyValue = input[keyIdx] * keyWeights[hiddenIdx];
          let valueValue = input[keyIdx] * valueWeights[hiddenIdx];

          // Attention score (simplified)
          let score = exp(queryValue * keyValue / sqrt(f32(params.headDim)));
          attentionSum += score * valueValue;
        }

        output[inputOffset] = attentionSum;
      }
    `;
  }

  /**
   * Process dimensional array with kernel attention
   */
  async processDimensionalArray(
    data: Float32Array,
    shape: number[],
    attentionWeights: Float32Array,
    kernelSize = 8
  ): Promise<{
    result: Float32Array;
    processingTime: number;
    gpuMemoryUsed: number;
    recommendations: string[];
  }> {
    if (!this.capabilities?.isSupported || !this.capabilities.device) {
      throw new Error('WebGPU not available');
    }

    const startTime = performance.now();
    const device = this.capabilities.device;

    // Create or get cached compute pipeline
    const shaderKey = `kernel_attention_${kernelSize}`;
    let pipeline = this.shaderCache.get(shaderKey);

    if (!pipeline) {
      const shaderModule = device.createShaderModule({
        code: this.createKernelAttentionShader()
      });

      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
        ]
      });

      pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: {
          module: shaderModule,
          entryPoint: 'kernelAttention'
        }
      });

      this.shaderCache.set(shaderKey, pipeline);
    }

    // Create buffers
    const inputBuffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    const attentionBuffer = device.createBuffer({
      size: attentionWeights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    const outputBuffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const paramsData = new Uint32Array([data.length, data.length, kernelSize, 8]);
    const paramsBuffer = device.createBuffer({
      size: paramsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Write data to buffers
    device.queue.writeBuffer(inputBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
    device.queue.writeBuffer(attentionBuffer, 0, attentionWeights.buffer, attentionWeights.byteOffset, attentionWeights.byteLength);
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: attentionBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });

    // Execute compute shader
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();

    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(data.length / 64));
    computePass.end();

    // Read back results
    const readBuffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, data.byteLength);
    device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // Cleanup
    inputBuffer.destroy();
    attentionBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readBuffer.destroy();

    const processingTime = performance.now() - startTime;

    return {
      result,
      processingTime,
      gpuMemoryUsed: data.byteLength * 4, // Estimate
      recommendations: [
        'Increase kernel size for better attention coverage',
        'Try different attention weight patterns',
        'Use batch processing for multiple arrays',
        'Enable caching for repeated computations'
      ]
    };
  }

  /**
   * Process T5 transformer inference
   */
  async processT5Inference(
    tokens: Float32Array,
    sequenceLength: number,
    hiddenSize: number = 768,
    numHeads: number = 12
  ): Promise<{
    result: Float32Array;
    processingTime: number;
    recommendations: string[];
  }> {
    if (!this.capabilities?.isSupported || !this.capabilities.device) {
      throw new Error('WebGPU not available');
    }

    const startTime = performance.now();
    const device = this.capabilities.device;

    // Create T5 transformer pipeline
    const shaderModule = device.createShaderModule({
      code: this.createT5TransformerShader()
    });

    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 't5Attention'
      }
    });

    // Create buffers for T5 computation
    const inputBuffer = device.createBuffer({
      size: tokens.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    // Simplified weight matrices (normally loaded from model)
    const weights = new Float32Array(hiddenSize).fill(0.1);
    const weightsBuffer = device.createBuffer({
      size: weights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    const outputBuffer = device.createBuffer({
      size: tokens.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const params = new Uint32Array([sequenceLength, hiddenSize, numHeads, hiddenSize / numHeads]);
    const paramsBuffer = device.createBuffer({
      size: params.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Write data
    device.queue.writeBuffer(inputBuffer, 0, tokens.buffer, tokens.byteOffset, tokens.byteLength);
    device.queue.writeBuffer(weightsBuffer, 0, weights);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightsBuffer } },
        { binding: 2, resource: { buffer: weightsBuffer } }, // Reuse for demo
        { binding: 3, resource: { buffer: weightsBuffer } }, // Reuse for demo
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } }
      ]
    });

    // Execute
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();

    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(sequenceLength / 32), Math.ceil(hiddenSize / 32));
    computePass.end();

    // Read results
    const readBuffer = device.createBuffer({
      size: tokens.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, tokens.byteLength);
    device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // Cleanup
    inputBuffer.destroy();
    weightsBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readBuffer.destroy();

    const processingTime = performance.now() - startTime;

    return {
      result,
      processingTime,
      recommendations: [
        'Use larger models for better quality',
        'Implement proper weight loading',
        'Add layer normalization',
        'Enable mixed precision for speed'
      ]
    };
  }

  /**
   * Get modular AI recommendations
   */
  getModularRecommendations(
    userId: string,
    context: string,
    computationHistory: AIComputeJob[]
  ): {
    pickUpWhereLeftOff: string;
    didYouMean: string[];
    othersSearched: string[];
    cuttingEdge: string[];
  } {
    const recentJobs = computationHistory.filter(job =>
      Date.now() - job.createdAt < 86400000 // Last 24 hours
    );

    return {
      pickUpWhereLeftOff: recentJobs.length > 0
        ? `Resume ${recentJobs[0].type} computation?`
        : 'Start new AI computation?',

      didYouMean: [
        `${context} with kernel attention?`,
        `${context} using T5 transformer?`,
        `${context} with WebGPU acceleration?`,
        `${context} with modular switching?`
      ],

      othersSearched: [
        'kernel splicing attention',
        'dimensional array optimization',
        'T5 model fine-tuning',
        'WebGPU compute shaders',
        'modular AI experiences'
      ],

      cuttingEdge: [
        'Neural architecture search with attention',
        'Hybrid CPU-GPU computation graphs',
        'Real-time model switching',
        'Adaptive kernel sizing',
        'Self-optimizing attention weights'
      ]
    };
  }

  /**
   * Create custom AI library components
   */
  createCustomLibrary(): {
    DimensionalProcessor: any;
    AttentionKernel: any;
    ModularSwitch: any;
    T5Accelerator: any;
  } {
    const self = this;

    return {
      DimensionalProcessor: class {
        static async process(data: Float32Array, shape: number[]) {
          return await self.processDimensionalArray(
            data,
            shape,
            new Float32Array(Math.min(8, data.length)).fill(0.8)
          );
        }
      },

      AttentionKernel: class {
        static splice(data: Float32Array, kernelSize: number) {
          // Kernel splicing implementation
          const slices = [];
          for (let i = 0; i < data.length; i += kernelSize) {
            const slice = data.slice(i, Math.min(i + kernelSize, data.length));
            slices.push({
              data: slice,
              attentionScore: slice.reduce((sum, val) => sum + val, 0) / slice.length,
              startIndex: i
            });
          }
          return slices.sort((a, b) => b.attentionScore - a.attentionScore);
        }
      },

      ModularSwitch: class {
        private static activeModule: string = 'default';

        static switch(moduleName: string, config: any) {
          console.log(`🔄 Switching to module: ${moduleName}`);
          this.activeModule = moduleName;
          // Hot-swappable module loading
          return { switched: true, module: moduleName, config };
        }

        static getActive() {
          return this.activeModule;
        }
      },

      T5Accelerator: class {
        static async process(text: string, task: 'summarize' | 'translate' | 'qa') {
          // Convert text to tokens (simplified)
          const tokens = new Float32Array(text.length);
          for (let i = 0; i < text.length; i++) {
            tokens[i] = text.charCodeAt(i) / 255.0;
          }

          return await self.processT5Inference(tokens, text.length);
        }
      }
    };
  }

  /**
   * Get engine capabilities and stats
   */
  getCapabilities(): {
    webgpu: WebGPUCapabilities;
    performance: any;
    recommendations: string[];
  } {
    return {
      webgpu: this.capabilities || { isSupported: false, features: [], limits: {} },
      performance: {
        jobsProcessed: this.computeJobs.size,
        cachedShaders: this.shaderCache.size,
        averageProcessingTime: 50, // Placeholder
        gpuUtilization: 0.75 // Placeholder
      },
      recommendations: [
        'Enable WebGPU for maximum performance',
        'Use larger workgroup sizes for better GPU utilization',
        'Implement shader caching for repeated operations',
        'Consider batching multiple computations',
        'Use memory pools to reduce allocation overhead'
      ]
    };
  }
}

// Export singleton instance
export const webgpuAI = new WebGPUAIEngine();