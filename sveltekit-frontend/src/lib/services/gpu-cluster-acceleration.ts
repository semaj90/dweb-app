/**
 * GPU Cluster Acceleration System
 * Multi-cluster GPU context switching with WebGL/WebGPU shader caching
 * Enables GPU utilization across Node.js worker processes
 */

import { EventEmitter } from 'node:events';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { createCanvas, Canvas } from 'canvas';
import { JSDOM } from 'jsdom';
import cluster from 'node:cluster';
import { writable, type Writable } from 'svelte/store';

// GPU context and shader interfaces
export interface GPUContext {
  id: string;
  workerId: number;
  contextType: 'webgl' | 'webgl2' | 'webgpu';
  canvas: Canvas | OffscreenCanvas;
  gl: WebGLRenderingContext | WebGL2RenderingContext | null;
  device?: GPUDevice; // WebGPU device
  queue?: GPUQueue;   // WebGPU command queue
  isActive: boolean;
  lastUsed: number;
  memoryUsage: number;
  shaderCount: number;
}

export interface CachedShader {
  id: string;
  name: string;
  vertexSource: string;
  fragmentSource: string;
  compiledProgram?: WebGLProgram;
  webgpuModule?: GPUShaderModule;
  uniforms: Map<string, WebGLUniformLocation>;
  attributes: Map<string, number>;
  compilationTime: number;
  accessCount: number;
  lastAccessed: number;
  contextId: string;
  memorySize: number;
}

export interface GPUClusterMetrics {
  totalContexts: number;
  activeContexts: number;
  totalShaders: number;
  cacheHitRate: number;
  compilationTime: number;
  memoryUsage: {
    total: number;
    perContext: number;
    shaderCache: number;
  };
  performance: {
    frameRate: number;
    renderTime: number;
    contextSwitches: number;
  };
}

export interface GPUWorkload {
  id: string;
  type: 'vector-processing' | 'matrix-operations' | 'shader-compilation' | 'attention-weights';
  priority: 'low' | 'medium' | 'high' | 'critical';
  data: Float32Array | Uint32Array;
  shaderProgram: string;
  uniforms?: Record<string, any>;
  expectedDuration: number;
  callback: (result: any) => void;
}

/**
 * GPU Cluster Manager
 * Manages GPU contexts across multiple Node.js cluster workers
 */
export class GPUClusterManager extends EventEmitter {
  private contexts = new Map<string, GPUContext>();
  private shaderCache = new Map<string, CachedShader>();
  private gpuWorkers = new Map<string, Worker>();
  private workloadQueue: GPUWorkload[] = [];
  
  // Performance tracking
  private metrics: Writable<GPUClusterMetrics>;
  private contextSwitchCount = 0;
  private totalCompilationTime = 0;
  private cacheHits = 0;
  private cacheMisses = 0;
  
  // Configuration
  private config = {
    maxContextsPerWorker: 2,
    shaderCacheSize: 1000,
    contextTimeout: 300000, // 5 minutes
    enableWebGPU: true,
    enableOptimizations: true,
    memoryLimit: 512 * 1024 * 1024, // 512MB
  };

  constructor() {
    super();
    
    this.metrics = writable(this.getInitialMetrics());
    this.initializeGPUWorkers();
    this.setupClusterIntegration();
    this.startPerformanceMonitoring();
  }

  /**
   * Initialize GPU worker threads for each cluster worker
   */
  private async initializeGPUWorkers(): Promise<void> {
    console.log('🎮 Initializing GPU workers for cluster...');
    
    // Create GPU workers for each cluster worker
    const workerCount = cluster.isPrimary ? 
      Object.keys(cluster.workers || {}).length : 1;
    
    for (let i = 0; i < workerCount; i++) {
      await this.createGPUWorker(i);
    }
    
    // Initialize WebGPU if available
    if (this.config.enableWebGPU) {
      await this.initializeWebGPU();
    }
    
    console.log(`✅ GPU cluster manager initialized with ${this.contexts.size} contexts`);
  }

  /**
   * Create a GPU worker thread
   */
  private async createGPUWorker(workerId: number): Promise<void> {
    const worker = new Worker(__filename, {
      workerData: {
        type: 'gpu-worker',
        workerId,
        config: this.config
      }
    });

    // Handle worker messages
    worker.on('message', (message) => {
      this.handleWorkerMessage(workerId.toString(), message);
    });

    worker.on('error', (error) => {
      console.error(`GPU worker ${workerId} error:`, error);
      this.restartGPUWorker(workerId);
    });

    this.gpuWorkers.set(workerId.toString(), worker);

    // Create GPU contexts for this worker
    await this.createGPUContexts(workerId);
  }

  /**
   * Create GPU contexts (WebGL/WebGPU) for a worker
   */
  private async createGPUContexts(workerId: number): Promise<void> {
    const contextTypes: Array<'webgl' | 'webgl2' | 'webgpu'> = ['webgl2', 'webgl'];
    
    if (this.config.enableWebGPU) {
      contextTypes.unshift('webgpu');
    }

    for (let i = 0; i < this.config.maxContextsPerWorker; i++) {
      const contextType = contextTypes[i % contextTypes.length];
      const contextId = `${workerId}-${contextType}-${i}`;
      
      try {
        const context = await this.createGPUContext(contextId, workerId, contextType);
        this.contexts.set(contextId, context);
        
        console.log(`📱 Created ${contextType} context: ${contextId}`);
        
      } catch (error) {
        console.error(`Failed to create ${contextType} context for worker ${workerId}:`, error);
      }
    }
  }

  /**
   * Create individual GPU context
   */
  private async createGPUContext(
    contextId: string, 
    workerId: number, 
    contextType: 'webgl' | 'webgl2' | 'webgpu'
  ): Promise<GPUContext> {
    
    let canvas: Canvas;
    let gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
    let device: GPUDevice | undefined;
    let queue: GPUQueue | undefined;

    if (contextType === 'webgpu') {
      // WebGPU context (Node.js with @webgpu/dawn or similar)
      try {
        // Note: This requires WebGPU implementation for Node.js
        // For production, you'd use libraries like @webgpu/dawn
        const adapter = await (globalThis as any).navigator?.gpu?.requestAdapter?.();
        if (adapter) {
          device = await adapter.requestDevice();
          queue = device.queue;
        }
      } catch (error) {
        console.warn('WebGPU not available, falling back to WebGL');
        contextType = 'webgl2';
      }
    }

    if (!device) {
      // Create WebGL context using node-canvas
      canvas = createCanvas(1, 1); // Offscreen canvas
      
      try {
        if (contextType === 'webgl2') {
          gl = canvas.getContext('webgl2') as WebGL2RenderingContext;
        } else {
          gl = canvas.getContext('webgl') as WebGLRenderingContext;
        }
        
        if (!gl) {
          throw new Error(`Failed to create ${contextType} context`);
        }
        
        // Enable extensions
        gl.getExtension('OES_texture_float');
        gl.getExtension('OES_texture_float_linear');
        gl.getExtension('WEBGL_color_buffer_float');
        
      } catch (error) {
        throw new Error(`WebGL context creation failed: ${error}`);
      }
    }

    return {
      id: contextId,
      workerId,
      contextType,
      canvas: canvas!,
      gl,
      device,
      queue,
      isActive: false,
      lastUsed: Date.now(),
      memoryUsage: 0,
      shaderCount: 0
    };
  }

  /**
   * Initialize WebGPU support
   */
  private async initializeWebGPU(): Promise<void> {
    try {
      // Check if WebGPU is available in Node.js environment
      if (typeof (globalThis as any).navigator?.gpu === 'undefined') {
        console.log('⚠️ WebGPU not available in this Node.js environment');
        return;
      }

      const adapter = await (globalThis as any).navigator.gpu.requestAdapter();
      if (!adapter) {
        console.log('⚠️ No WebGPU adapter available');
        return;
      }

      console.log('🚀 WebGPU adapter found:', adapter);
      
      // Get adapter info
      const info = await adapter.requestAdapterInfo();
      console.log('GPU Info:', {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description
      });

    } catch (error) {
      console.warn('WebGPU initialization failed:', error);
    }
  }

  /**
   * Compile and cache shader program
   */
  public async compileShader(
    name: string,
    vertexSource: string,
    fragmentSource: string,
    contextId?: string
  ): Promise<CachedShader> {
    
    const shaderId = this.generateShaderId(name, vertexSource, fragmentSource);
    
    // Check cache first
    const cached = this.shaderCache.get(shaderId);
    if (cached) {
      cached.accessCount++;
      cached.lastAccessed = Date.now();
      this.cacheHits++;
      return cached;
    }

    this.cacheMisses++;
    
    // Select optimal context for compilation
    const context = contextId ? 
      this.contexts.get(contextId) : 
      this.selectOptimalContext();

    if (!context) {
      throw new Error('No available GPU context for shader compilation');
    }

    const startTime = Date.now();
    
    try {
      let compiledProgram: WebGLProgram | undefined;
      let webgpuModule: GPUShaderModule | undefined;
      
      if (context.contextType === 'webgpu' && context.device) {
        // WebGPU shader compilation
        webgpuModule = context.device.createShaderModule({
          code: this.convertToWGSL(vertexSource, fragmentSource)
        });
        
      } else if (context.gl) {
        // WebGL shader compilation
        compiledProgram = this.compileWebGLShader(
          context.gl, 
          vertexSource, 
          fragmentSource
        );
      }

      const compilationTime = Date.now() - startTime;
      this.totalCompilationTime += compilationTime;

      // Extract uniforms and attributes
      const uniforms = new Map<string, WebGLUniformLocation>();
      const attributes = new Map<string, number>();
      
      if (compiledProgram && context.gl) {
        this.extractShaderMetadata(context.gl, compiledProgram, uniforms, attributes);
      }

      const cachedShader: CachedShader = {
        id: shaderId,
        name,
        vertexSource,
        fragmentSource,
        compiledProgram,
        webgpuModule,
        uniforms,
        attributes,
        compilationTime,
        accessCount: 1,
        lastAccessed: Date.now(),
        contextId: context.id,
        memorySize: this.calculateShaderMemorySize(vertexSource, fragmentSource)
      };

      // Cache the shader
      this.shaderCache.set(shaderId, cachedShader);
      context.shaderCount++;
      
      // Cleanup old shaders if cache is full
      if (this.shaderCache.size > this.config.shaderCacheSize) {
        this.cleanupShaderCache();
      }

      console.log(`✨ Compiled shader '${name}' in ${compilationTime}ms`);
      
      return cachedShader;
      
    } catch (error) {
      console.error(`Shader compilation failed for '${name}':`, error);
      throw error;
    }
  }

  /**
   * Compile WebGL shader program
   */
  private compileWebGLShader(
    gl: WebGLRenderingContext | WebGL2RenderingContext,
    vertexSource: string,
    fragmentSource: string
  ): WebGLProgram {
    
    const vertexShader = this.compileWebGLShaderStage(gl, gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.compileWebGLShaderStage(gl, gl.FRAGMENT_SHADER, fragmentSource);
    
    const program = gl.createProgram();
    if (!program) {
      throw new Error('Failed to create shader program');
    }
    
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const error = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Shader program linking failed: ${error}`);
    }
    
    // Clean up individual shaders
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    
    return program;
  }

  /**
   * Compile individual WebGL shader stage
   */
  private compileWebGLShaderStage(
    gl: WebGLRenderingContext | WebGL2RenderingContext,
    type: number,
    source: string
  ): WebGLShader {
    
    const shader = gl.createShader(type);
    if (!shader) {
      throw new Error('Failed to create shader');
    }
    
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const error = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compilation failed: ${error}`);
    }
    
    return shader;
  }

  /**
   * Convert GLSL to WGSL for WebGPU (simplified)
   */
  private convertToWGSL(vertexSource: string, fragmentSource: string): string {
    // This is a simplified conversion - in practice you'd use a proper GLSL->WGSL transpiler
    return `
      // Vertex stage
      @vertex
      fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
        return vec4<f32>(position, 1.0);
      }
      
      // Fragment stage  
      @fragment
      fn fs_main() -> @location(0) vec4<f32> {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
      }
    `;
  }

  /**
   * Execute GPU workload across cluster
   */
  public async executeWorkload(workload: GPUWorkload): Promise<any> {
    const context = this.selectOptimalContext(workload.type);
    
    if (!context) {
      throw new Error('No available GPU context for workload');
    }

    this.switchContext(context.id);
    
    try {
      let result: any;
      
      switch (workload.type) {
        case 'vector-processing':
          result = await this.processVectors(context, workload);
          break;
          
        case 'matrix-operations':
          result = await this.processMatrices(context, workload);
          break;
          
        case 'attention-weights':
          result = await this.processAttentionWeights(context, workload);
          break;
          
        case 'shader-compilation':
          result = await this.compileShaderWorkload(context, workload);
          break;
          
        default:
          throw new Error(`Unknown workload type: ${workload.type}`);
      }
      
      this.emit('workload-completed', { workloadId: workload.id, result, context: context.id });
      return result;
      
    } catch (error) {
      this.emit('workload-failed', { workloadId: workload.id, error, context: context.id });
      throw error;
    }
  }

  /**
   * Process vector operations on GPU
   */
  private async processVectors(context: GPUContext, workload: GPUWorkload): Promise<Float32Array> {
    if (context.contextType === 'webgpu' && context.device) {
      return this.processVectorsWebGPU(context, workload);
    } else if (context.gl) {
      return this.processVectorsWebGL(context, workload);
    }
    
    throw new Error('No suitable GPU context for vector processing');
  }

  /**
   * Process vectors using WebGPU compute shaders
   */
  private async processVectorsWebGPU(context: GPUContext, workload: GPUWorkload): Promise<Float32Array> {
    const device = context.device!;
    const queue = context.queue!;
    
    // Create compute shader for vector operations
    const computeShader = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          if (index >= arrayLength(&data)) {
            return;
          }
          
          // Example: vector normalization
          data[index] = data[index] * 2.0;
        }
      `
    });

    // Create buffer for input data
    const inputBuffer = device.createBuffer({
      size: workload.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Upload data
    queue.writeBuffer(inputBuffer, 0, workload.data);

    // Create compute pipeline
    const computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: computeShader,
        entryPoint: 'main',
      },
    });

    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [{
        binding: 0,
        resource: { buffer: inputBuffer },
      }],
    });

    // Execute compute shader
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(workload.data.length / 64));
    passEncoder.end();

    // Create output buffer
    const outputBuffer = device.createBuffer({
      size: workload.data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    commandEncoder.copyBufferToBuffer(inputBuffer, 0, outputBuffer, 0, workload.data.byteLength);
    queue.submit([commandEncoder.finish()]);

    // Read results
    await outputBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(outputBuffer.getMappedRange());
    outputBuffer.unmap();

    return result;
  }

  /**
   * Process vectors using WebGL compute (via transform feedback)
   */
  private async processVectorsWebGL(context: GPUContext, workload: GPUWorkload): Promise<Float32Array> {
    const gl = context.gl as WebGL2RenderingContext;
    
    // Use transform feedback for compute-like operations
    const vertexShader = `#version 300 es
      in float inputValue;
      out float outputValue;
      
      void main() {
        outputValue = inputValue * 2.0; // Example operation
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
      }
    `;

    const fragmentShader = `#version 300 es
      precision mediump float;
      out vec4 fragColor;
      void main() {
        fragColor = vec4(1.0);
      }
    `;

    const program = this.compileWebGLShader(gl, vertexShader, fragmentShader);
    
    // Create vertex buffer
    const inputBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, inputBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, workload.data, gl.STATIC_DRAW);

    // Setup transform feedback
    const outputBuffer = gl.createBuffer();
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, outputBuffer);
    gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, workload.data.byteLength, gl.DYNAMIC_READ);

    gl.useProgram(program);
    
    // Bind input
    const inputLocation = gl.getAttribLocation(program, 'inputValue');
    gl.enableVertexAttribArray(inputLocation);
    gl.bindBuffer(gl.ARRAY_BUFFER, inputBuffer);
    gl.vertexAttribPointer(inputLocation, 1, gl.FLOAT, false, 0, 0);

    // Setup transform feedback
    const transformFeedback = gl.createTransformFeedback();
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, transformFeedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, outputBuffer);

    // Execute
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, workload.data.length);
    gl.endTransformFeedback();

    // Read results
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, outputBuffer);
    const result = new Float32Array(workload.data.length);
    // Note: In Node.js, you'd need to use appropriate buffer reading methods
    
    return result;
  }

  /**
   * Process attention weights for transformer models
   */
  private async processAttentionWeights(context: GPUContext, workload: GPUWorkload): Promise<Float32Array> {
    // Specialized attention computation on GPU
    const data = workload.data as Float32Array;
    const seqLen = Math.sqrt(data.length);
    
    if (context.contextType === 'webgpu' && context.device) {
      // WebGPU attention computation
      const computeShader = context.device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read> input: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output: array<f32>;
          
          @compute @workgroup_size(8, 8)
          fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x;
            let j = global_id.y;
            let seq_len = u32(${seqLen});
            
            if (i >= seq_len || j >= seq_len) {
              return;
            }
            
            let idx = i * seq_len + j;
            // Softmax attention computation
            output[idx] = exp(input[idx]) / 1.0; // Simplified
          }
        `
      });
      
      // Execute attention computation
      // ... (implementation similar to vector processing)
    }
    
    return data; // Placeholder
  }

  /**
   * Select optimal GPU context for workload
   */
  private selectOptimalContext(workloadType?: string): GPUContext | null {
    const availableContexts = Array.from(this.contexts.values())
      .filter(ctx => !ctx.isActive)
      .sort((a, b) => {
        // Prefer WebGPU for compute workloads
        if (workloadType?.includes('vector') || workloadType?.includes('matrix')) {
          if (a.contextType === 'webgpu' && b.contextType !== 'webgpu') return -1;
          if (b.contextType === 'webgpu' && a.contextType !== 'webgpu') return 1;
        }
        
        // Sort by memory usage and last used time
        return (a.memoryUsage - b.memoryUsage) || (a.lastUsed - b.lastUsed);
      });

    return availableContexts[0] || null;
  }

  /**
   * Switch to specific GPU context
   */
  private switchContext(contextId: string): void {
    // Deactivate current contexts
    this.contexts.forEach(ctx => ctx.isActive = false);
    
    // Activate target context
    const context = this.contexts.get(contextId);
    if (context) {
      context.isActive = true;
      context.lastUsed = Date.now();
      this.contextSwitchCount++;
    }
  }

  /**
   * Setup integration with Node.js cluster
   */
  private setupClusterIntegration(): void {
    if (cluster.isPrimary) {
      // Primary process: coordinate GPU work across workers
      cluster.on('message', (worker, message) => {
        if (message.type === 'gpu-workload') {
          this.distributeWorkload(message.workload);
        }
      });
    } else {
      // Worker process: handle GPU work locally
      process.on('message', (message) => {
        if (message.type === 'gpu-workload') {
          this.executeWorkload(message.workload)
            .then(result => {
              process.send?.({ type: 'gpu-result', result, workloadId: message.workload.id });
            })
            .catch(error => {
              process.send?.({ type: 'gpu-error', error: error.message, workloadId: message.workload.id });
            });
        }
      });
    }
  }

  /**
   * Distribute GPU workload across cluster workers
   */
  private distributeWorkload(workload: GPUWorkload): void {
    // Select worker based on GPU context availability
    const optimalWorker = this.selectOptimalWorkerForGPU();
    
    if (optimalWorker && cluster.workers && cluster.workers[optimalWorker]) {
      cluster.workers[optimalWorker].send({
        type: 'gpu-workload',
        workload
      });
    } else {
      // Execute locally if no workers available
      this.executeWorkload(workload);
    }
  }

  /**
   * Select optimal cluster worker for GPU workload
   */
  private selectOptimalWorkerForGPU(): number | null {
    if (!cluster.workers) return null;
    
    // Simple round-robin for now - could be more sophisticated
    const workerIds = Object.keys(cluster.workers).map(Number);
    const selectedWorker = workerIds[this.contextSwitchCount % workerIds.length];
    
    return selectedWorker;
  }

  /**
   * Utility methods
   */
  private generateShaderId(name: string, vertex: string, fragment: string): string {
    const combined = `${name}:${vertex}:${fragment}`;
    let hash = 0;
    for (let i = 0; i < combined.length; i++) {
      const char = combined.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return `shader_${Math.abs(hash).toString(36)}`;
  }

  private calculateShaderMemorySize(vertex: string, fragment: string): number {
    return (vertex.length + fragment.length) * 2; // Rough estimate
  }

  private extractShaderMetadata(
    gl: WebGLRenderingContext | WebGL2RenderingContext,
    program: WebGLProgram,
    uniforms: Map<string, WebGLUniformLocation>,
    attributes: Map<string, number>
  ): void {
    
    // Extract uniforms
    const uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uniformCount; i++) {
      const uniform = gl.getActiveUniform(program, i);
      if (uniform) {
        const location = gl.getUniformLocation(program, uniform.name);
        if (location) {
          uniforms.set(uniform.name, location);
        }
      }
    }

    // Extract attributes
    const attributeCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
    for (let i = 0; i < attributeCount; i++) {
      const attribute = gl.getActiveAttrib(program, i);
      if (attribute) {
        const location = gl.getAttribLocation(program, attribute.name);
        attributes.set(attribute.name, location);
      }
    }
  }

  private cleanupShaderCache(): void {
    // Remove least recently used shaders
    const shaders = Array.from(this.shaderCache.values())
      .sort((a, b) => a.lastAccessed - b.lastAccessed);
    
    const toRemove = shaders.slice(0, Math.floor(this.config.shaderCacheSize * 0.1));
    
    toRemove.forEach(shader => {
      this.shaderCache.delete(shader.id);
      
      // Clean up GPU resources
      const context = this.contexts.get(shader.contextId);
      if (context?.gl && shader.compiledProgram) {
        context.gl.deleteProgram(shader.compiledProgram);
        context.shaderCount--;
      }
    });
  }

  private startPerformanceMonitoring(): void {
    setInterval(() => {
      this.updateMetrics();
    }, 1000);
  }

  private updateMetrics(): void {
    const totalMemory = Array.from(this.contexts.values())
      .reduce((sum, ctx) => sum + ctx.memoryUsage, 0);
    
    const activeContexts = Array.from(this.contexts.values())
      .filter(ctx => ctx.isActive).length;
    
    const cacheHitRate = this.cacheHits + this.cacheMisses > 0 ? 
      this.cacheHits / (this.cacheHits + this.cacheMisses) : 0;

    const metrics: GPUClusterMetrics = {
      totalContexts: this.contexts.size,
      activeContexts,
      totalShaders: this.shaderCache.size,
      cacheHitRate,
      compilationTime: this.totalCompilationTime,
      memoryUsage: {
        total: totalMemory,
        perContext: this.contexts.size > 0 ? totalMemory / this.contexts.size : 0,
        shaderCache: Array.from(this.shaderCache.values())
          .reduce((sum, shader) => sum + shader.memorySize, 0)
      },
      performance: {
        frameRate: 60, // Would be measured
        renderTime: 16.67, // Would be measured
        contextSwitches: this.contextSwitchCount
      }
    };

    this.metrics.set(metrics);
  }

  private getInitialMetrics(): GPUClusterMetrics {
    return {
      totalContexts: 0,
      activeContexts: 0,
      totalShaders: 0,
      cacheHitRate: 0,
      compilationTime: 0,
      memoryUsage: { total: 0, perContext: 0, shaderCache: 0 },
      performance: { frameRate: 0, renderTime: 0, contextSwitches: 0 }
    };
  }

  // Public getters
  public getMetrics() { return this.metrics; }
  public getContexts() { return Array.from(this.contexts.values()); }
  public getShaderCache() { return Array.from(this.shaderCache.values()); }
  
  // Cleanup
  public async destroy(): Promise<void> {
    // Clean up GPU contexts
    this.contexts.forEach(context => {
      if (context.gl) {
        // Clean up WebGL context
        const loseContext = context.gl.getExtension('WEBGL_lose_context');
        loseContext?.loseContext();
      }
    });

    // Terminate GPU workers
    this.gpuWorkers.forEach(worker => worker.terminate());
    
    this.contexts.clear();
    this.shaderCache.clear();
    this.gpuWorkers.clear();
  }

  // Placeholder methods for different workload types
  private async processMatrices(context: GPUContext, workload: GPUWorkload): Promise<any> {
    // Matrix operations implementation
    return workload.data;
  }

  private async compileShaderWorkload(context: GPUContext, workload: GPUWorkload): Promise<any> {
    // Shader compilation workload
    return { compiled: true };
  }

  private restartGPUWorker(workerId: number): void {
    // Restart failed GPU worker
    console.log(`🔄 Restarting GPU worker ${workerId}`);
    // Implementation would recreate the worker
  }
}

// GPU Worker Thread Implementation
if (!isMainThread && workerData?.type === 'gpu-worker') {
  console.log(`🎮 GPU worker ${workerData.workerId} started`);
  
  // Initialize GPU context in worker thread
  // Handle GPU workloads sent from main thread
  
  parentPort?.on('message', async (message) => {
    if (message.type === 'gpu-workload') {
      try {
        // Process GPU workload in worker thread
        const result = await processGPUWorkloadInWorker(message.workload);
        parentPort?.postMessage({ type: 'result', result, workloadId: message.workload.id });
      } catch (error) {
        parentPort?.postMessage({ 
          type: 'error', 
          error: error instanceof Error ? error.message : 'Unknown error', 
          workloadId: message.workload.id 
        });
      }
    }
  });
}

async function processGPUWorkloadInWorker(workload: GPUWorkload): Promise<any> {
  // GPU workload processing in worker thread
  // This would use the same GPU context creation and processing logic
  return { processed: true, data: workload.data };
}

/**
 * Factory function for creating GPU cluster manager
 */
export function createGPUClusterManager(): GPUClusterManager {
  return new GPUClusterManager();
}

/**
 * Utility function to check GPU capabilities
 */
export async function checkGPUCapabilities(): Promise<{
  webgl: boolean;
  webgl2: boolean;
  webgpu: boolean;
  extensions: string[];
}> {
  
  const capabilities = {
    webgl: false,
    webgl2: false,
    webgpu: false,
    extensions: [] as string[]
  };

  try {
    // Check WebGL support in Node.js
    const canvas = createCanvas(1, 1);
    
    const webglContext = canvas.getContext('webgl');
    if (webglContext) {
      capabilities.webgl = true;
      capabilities.extensions = webglContext.getSupportedExtensions() || [];
    }

    const webgl2Context = canvas.getContext('webgl2');
    if (webgl2Context) {
      capabilities.webgl2 = true;
    }

    // Check WebGPU support
    if (typeof (globalThis as any).navigator?.gpu !== 'undefined') {
      const adapter = await (globalThis as any).navigator.gpu.requestAdapter();
      capabilities.webgpu = !!adapter;
    }

  } catch (error) {
    console.warn('GPU capability check failed:', error);
  }

  return capabilities;
}