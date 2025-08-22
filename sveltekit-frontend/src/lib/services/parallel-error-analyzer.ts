// ======================================================================
// PARALLEL ERROR ANALYZER WITH GPU ACCELERATION
// High-performance error analysis using WebGPU and Worker threads
// ======================================================================

import { gpuLokiErrorAPI } from './gpu-loki-error-orchestrator.js';
import { browser } from '$app/environment';

interface GPUComputePipeline {
  device: GPUDevice;
  computeShader: GPUShaderModule;
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
}

interface ErrorAnalysisWorker {
  worker: Worker;
  id: string;
  busy: boolean;
  processedCount: number;
}

interface ParallelAnalysisConfig {
  maxWorkers: number;
  batchSize: number;
  useGPU: boolean;
  workerScript: string;
}

class ParallelErrorAnalyzer {
  private gpuPipeline: GPUComputePipeline | null = null;
  private workers: ErrorAnalysisWorker[] = [];
  private config: ParallelAnalysisConfig = {
    maxWorkers: navigator.hardwareConcurrency || 4,
    batchSize: 50,
    useGPU: true,
    workerScript: '/workers/error-analysis-worker.js'
  };

  async initialize() {
    console.log('🚀 Initializing Parallel Error Analyzer...');
    
    if (browser) {
      await this.initializeGPU();
      await this.initializeWorkers();
    }
    
    console.log(`✅ Parallel analyzer ready with ${this.workers.length} workers`);
  }

  private async initializeGPU() {
    if (!this.config.useGPU || !navigator.gpu) {
      console.log('⚠️ GPU not available, using CPU-only processing');
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter?.requestDevice();
      
      if (!device) throw new Error('GPU device not available');

      // Create compute shader for error pattern analysis
      const computeShaderSource = `
        struct ErrorData {
          code: u32,
          line: u32, 
          confidence: f32,
          category: u32,
        };

        struct AnalysisResult {
          priority: f32,
          fixable: u32,
          complexity: f32,
          suggestions: u32,
        };

        @group(0) @binding(0) var<storage, read> errorData: array<ErrorData>;
        @group(0) @binding(1) var<storage, read_write> results: array<AnalysisResult>;

        // Error pattern recognition weights
        const SYNTAX_WEIGHT: f32 = 0.9;
        const TYPE_WEIGHT: f32 = 0.7; 
        const IMPORT_WEIGHT: f32 = 0.8;
        const SEMANTIC_WEIGHT: f32 = 0.5;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          if (index >= arrayLength(&errorData)) { return; }

          let error = errorData[index];
          var result: AnalysisResult;

          // Calculate priority based on error characteristics
          result.priority = error.confidence;
          
          // Apply category weights
          switch (error.category) {
            case 0u: { // syntax
              result.priority *= SYNTAX_WEIGHT;
              result.fixable = 1u;
              result.complexity = 1.0;
            }
            case 1u: { // type
              result.priority *= TYPE_WEIGHT;
              result.fixable = select(0u, 1u, error.confidence > 0.7);
              result.complexity = 2.0;
            }
            case 2u: { // import
              result.priority *= IMPORT_WEIGHT;
              result.fixable = 1u;
              result.complexity = 1.5;
            }
            default: { // semantic
              result.priority *= SEMANTIC_WEIGHT;
              result.fixable = 0u;
              result.complexity = 3.0;
            }
          }

          // Common fixable error codes
          let fixable_codes = array<u32, 6>(1434u, 2304u, 2307u, 2457u, 1005u, 1128u);
          for (var i = 0u; i < 6u; i++) {
            if (error.code == fixable_codes[i]) {
              result.fixable = 1u;
              result.suggestions = min(error.code / 100u, 5u);
              break;
            }
          }

          results[index] = result;
        }
      `;

      const shaderModule = device.createShaderModule({
        code: computeShaderSource
      });

      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' }
          },
          {
            binding: 1, 
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
          }
        ]
      });

      this.gpuPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout]
        }),
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });

      console.log('✅ GPU compute pipeline initialized');
    } catch (error) {
      console.warn('⚠️ GPU initialization failed:', error);
      this.config.useGPU = false;
    }
  }

  private async initializeWorkers() {
    // Create worker threads for CPU processing
    for (let i = 0; i < this.config.maxWorkers; i++) {
      try {
        const worker = new Worker(
          new URL('../workers/error-analysis-worker.ts', import.meta.url),
          { type: 'module' }
        );

        const errorWorker: ErrorAnalysisWorker = {
          worker,
          id: `worker_${i}`,
          busy: false,
          processedCount: 0
        };

        // Setup worker message handling
        worker.onmessage = (event) => {
          errorWorker.busy = false;
          errorWorker.processedCount++;
          
          const { type, result, error } = event.data;
          if (type === 'analysis_complete') {
            this.handleWorkerResult(errorWorker.id, result);
          } else if (type === 'error') {
            console.error(`Worker ${errorWorker.id} error:`, error);
          }
        };

        this.workers.push(errorWorker);
      } catch (error) {
        console.warn(`Failed to create worker ${i}:`, error);
      }
    }
  }

  async analyzeErrorsParallel(errors: any[]): Promise<unknown[]> {
    if (!errors.length) return [];

    console.log(`⚡ Analyzing ${errors.length} errors in parallel...`);
    const startTime = performance.now();

    let results: any[] = [];

    // Use GPU if available and batch is large enough
    if (this.gpuPipeline && errors.length >= 100) {
      results = await this.analyzeWithGPU(errors);
    } else {
      results = await this.analyzeWithWorkers(errors);
    }

    const processingTime = performance.now() - startTime;
    console.log(`✅ Parallel analysis completed in ${processingTime.toFixed(2)}ms`);

    return results;
  }

  private async analyzeWithGPU(errors: any[]): Promise<unknown[]> {
    if (!this.gpuPipeline) return [];

    console.log(`🚀 GPU analysis of ${errors.length} errors`);

    // Prepare data for GPU
    const errorData = new Uint32Array(errors.length * 4); // 4 values per error
    const resultData = new Float32Array(errors.length * 4); // 4 results per error

    for (let i = 0; i < errors.length; i++) {
      const error = errors[i];
      const offset = i * 4;
      
      errorData[offset] = parseInt(error.code.replace('TS', '')) || 0;
      errorData[offset + 1] = error.line || 0;
      errorData[offset + 2] = Math.floor((error.confidence || 0.5) * 100);
      errorData[offset + 3] = this.getCategoryCode(error.category);
    }

    // Create GPU buffers
    const device = (this.gpuPipeline as any).device;
    
    const errorBuffer = device.createBuffer({
      size: errorData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    const resultBuffer = device.createBuffer({
      size: resultData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const stagingBuffer = device.createBuffer({
      size: resultData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // Upload error data
    device.queue.writeBuffer(errorBuffer, 0, errorData);

    // Create bind group
    const bindGroup = device.createBindGroup({
      layout: (this.gpuPipeline as any).getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: errorBuffer } },
        { binding: 1, resource: { buffer: resultBuffer } }
      ]
    });

    // Dispatch compute shader
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    
    computePass.setPipeline(this.gpuPipeline);
    computePass.setBindGroup(0, bindGroup);
    
    const workgroupCount = Math.ceil(errors.length / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();

    // Copy results back
    commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, resultData.byteLength);
    
    device.queue.submit([commandEncoder.finish()]);

    // Read results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const results = new Float32Array(stagingBuffer.getMappedRange());
    
    // Convert GPU results back to analysis format
    const analysisResults = [];
    for (let i = 0; i < errors.length; i++) {
      const offset = i * 4;
      analysisResults.push({
        errorId: errors[i].id,
        priority: results[offset],
        fixable: results[offset + 1] > 0,
        complexity: results[offset + 2],
        suggestionCount: Math.floor(results[offset + 3]),
        gpuProcessed: true
      });
    }

    stagingBuffer.unmap();
    
    // Cleanup GPU resources
    errorBuffer.destroy();
    resultBuffer.destroy();
    stagingBuffer.destroy();

    return analysisResults;
  }

  private getCategoryCode(category: string): number {
    switch (category) {
      case 'syntax': return 0;
      case 'type': return 1;
      case 'import': return 2;
      case 'semantic': return 3;
      default: return 3;
    }
  }

  private async analyzeWithWorkers(errors: any[]): Promise<unknown[]> {
    console.log(`👥 Worker analysis of ${errors.length} errors`);

    const batches = this.createBatches(errors, this.config.batchSize);
    const results: any[] = [];
    
    const processBatch = async (batch: any[]): Promise<unknown[]> => {
      return new Promise((resolve) => {
        const availableWorker = this.getAvailableWorker();
        
        if (!availableWorker) {
          // Fallback to synchronous processing
          resolve(this.analyzeErrorsSynchronous(batch));
          return;
        }

        availableWorker.busy = true;
        
        const timeout = setTimeout(() => {
          availableWorker.busy = false;
          resolve(this.analyzeErrorsSynchronous(batch));
        }, 5000);

        const handleMessage = (event: MessageEvent) => {
          if (event.data.type === 'analysis_complete') {
            clearTimeout(timeout);
            availableWorker.worker.removeEventListener('message', handleMessage);
            availableWorker.busy = false;
            resolve(event.data.results);
          }
        };

        availableWorker.worker.addEventListener('message', handleMessage);
        availableWorker.worker.postMessage({
          type: 'analyze_errors',
          errors: batch
        });
      });
    };

    // Process batches in parallel
    const batchPromises = batches.map(batch => processBatch(batch));
    const batchResults = await Promise.all(batchPromises);
    
    // Flatten results
    for (const batchResult of batchResults) {
      results.push(...batchResult);
    }

    return results;
  }

  private createBatches(errors: any[], batchSize: number): unknown[][] {
    const batches = [];
    for (let i = 0; i < errors.length; i += batchSize) {
      batches.push(errors.slice(i, i + batchSize));
    }
    return batches;
  }

  private getAvailableWorker(): ErrorAnalysisWorker | null {
    return this.workers.find(worker => !worker.busy) || null;
  }

  private handleWorkerResult(workerId: string, result: any) {
    // Handle worker completion
    console.log(`✅ Worker ${workerId} completed analysis`);
  }

  private analyzeErrorsSynchronous(errors: any[]): unknown[] {
    // Fallback synchronous analysis
    return errors.map(error => ({
      errorId: error.id,
      priority: error.confidence || 0.5,
      fixable: this.isErrorFixable(error.code),
      complexity: this.estimateComplexity(error),
      suggestionCount: this.getSuggestionCount(error.code),
      gpuProcessed: false
    }));
  }

  private isErrorFixable(code: string): boolean {
    const fixableCodes = ['1434', '2304', '2307', '2457', '1005', '1128'];
    return fixableCodes.includes(code.replace('TS', ''));
  }

  private estimateComplexity(error: any): number {
    let complexity = 1;
    
    if (error.category === 'semantic') complexity += 2;
    if (error.category === 'type') complexity += 1;
    if (error.confidence > 0.8) complexity -= 0.5;
    
    return Math.max(1, complexity);
  }

  private getSuggestionCount(code: string): number {
    const codeNum = parseInt(code.replace('TS', ''));
    return Math.min(Math.floor(codeNum / 1000) + 1, 5);
  }

  getStats() {
    return {
      workers: this.workers.length,
      busyWorkers: this.workers.filter(w => w.busy).length,
      gpuEnabled: !!this.gpuPipeline,
      totalProcessed: this.workers.reduce((sum, w) => sum + w.processedCount, 0)
    };
  }

  destroy() {
    // Clean up workers
    for (const worker of this.workers) {
      worker.worker.terminate();
    }
    this.workers = [];
  }
}

export const parallelErrorAnalyzer = new ParallelErrorAnalyzer();

// ======================================================================
// INTEGRATION API
// ======================================================================

export const parallelAnalysisAPI = {
  async initialize() {
    await parallelErrorAnalyzer.initialize();
  },

  async analyzeErrors(errors: any[]) {
    return await parallelErrorAnalyzer.analyzeErrorsParallel(errors);
  },

  getStats() {
    return parallelErrorAnalyzer.getStats();
  },

  destroy() {
    parallelErrorAnalyzer.destroy();
  }
};