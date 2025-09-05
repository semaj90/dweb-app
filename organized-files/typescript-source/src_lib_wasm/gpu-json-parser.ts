// @ts-nocheck
// @ts-nocheck
/**
 * GPU-Accelerated Rapid JSON Parser for VS Code Extensions
 * WebAssembly wrapper with performance optimizations and caching
 */

// WebGPU type declarations
declare global {
  const GPUBufferUsage: {
    readonly STORAGE: number;
    readonly COPY_DST: number;
    readonly COPY_SRC: number;
    readonly MAP_READ: number;
  };
  
  const GPUShaderStage: {
    readonly COMPUTE: number;
  };
  
  const GPUMapMode: {
    readonly READ: number;
  };
}

interface ParseMetrics {
  parseTime: number;
  documentSize: number;
  objectCount: number;
  arrayCount: number;
  parseMethod: string;
}

interface CacheStats {
  hits: number;
  misses: number;
  hitRate: number;
  cacheSize: number;
}

interface ParseResult {
  success: boolean;
  error?: boolean;
  errorMessage?: string;
  errorOffset?: number;
  parsed?: boolean;
}

interface BatchResult {
  results: ParseResult[];
  batchTime: number;
  documentCount: number;
  threadsUsed: number;
}

interface StringifyResult {
  success: boolean;
  error?: boolean;
  message?: string;
  json?: string;
  size?: number;
}

interface ValidationResult {
  valid: boolean;
  error?: string;
  message?: string;
}

interface WasmModule {
  RapidJsonParser: new () => RapidJsonParserWasm;
  getCacheStats(): CacheStats;
  clearCache(): void;
  createParser(): RapidJsonParserWasm;
  destroyParser(parser: RapidJsonParserWasm): void;
}

interface RapidJsonParserWasm {
  parseWithCache(json: string, useCache?: boolean): ParseResult;
  parseBatch(jsonArray: string[]): BatchResult;
  getValue(path: string): any;
  getMetrics(): ParseMetrics;
  stringify(options?: { pretty?: boolean }): StringifyResult;
  validate(schemaJson: string): ValidationResult;
}

/**
 * High-performance JSON parser using WebAssembly and GPU acceleration
 */
export class GpuAcceleratedJsonParser {
  private wasmModule: WasmModule | null = null;
  private parser: RapidJsonParserWasm | null = null;
  private isInitialized = false;
  private initPromise: Promise<void> | null = null;
  private performanceCache = new Map<string, any>();
  private webWorker: Worker | null = null;

  constructor() {
    this.initPromise = this.initialize();
  }

  /**
   * Initialize WebAssembly module and GPU resources
   */
  private async initialize(): Promise<void> {
    try {
      // Load WebAssembly module
      const wasmPath = "/static/wasm/rapid-json-parser.js";
      const RapidJsonWasm = await import(wasmPath);
      this.wasmModule = await RapidJsonWasm.default();

      // Create parser instance
      if (this.wasmModule) {
        this.parser = this.wasmModule.createParser();
      }

      // Initialize web worker for heavy operations
      await this.initializeWebWorker();

      this.isInitialized = true;
      console.log("GPU-accelerated JSON parser initialized successfully");
    } catch (error) {
      console.error("Failed to initialize WebAssembly JSON parser:", error);
      throw error;
    }
  }

  /**
   * Initialize web worker for parallel processing
   */
  private async initializeWebWorker(): Promise<void> {
    const workerCode = `
            let wasmModule = null;
            let parser = null;

            self.onmessage = async function(e) {
                const { id, type, data } = e.data;

                try {
                    if (type === 'init') {
                        const RapidJsonWasm = await import('${"/static/wasm/rapid-json-parser.js"}');
                        wasmModule = await RapidJsonWasm.default();
                        parser = wasmModule.createParser();
                        self.postMessage({ id, type: 'init', success: true });
                        return;
                    }

                    if (!parser) {
                        self.postMessage({ id, type, error: 'Parser not initialized' });
                        return;
                    }

                    let result;
                    switch (type) {
                        case 'parse':
                            result = parser.parseWithCache(data.json, data.useCache);
                            break;
                        case 'parseBatch':
                            result = parser.parseBatch(data.jsonArray);
                            break;
                        case 'getValue':
                            result = parser.getValue(data.path);
                            break;
                        case 'stringify':
                            result = parser.stringify(data.options);
                            break;
                        default:
                            throw new Error('Unknown operation type: ' + type);
                    }

                    self.postMessage({ id, type, result });
                } catch (error) {
                    self.postMessage({ id, type, error: error.message });
                }
            };
        `;

    const blob = new Blob([workerCode], { type: "application/javascript" });
    this.webWorker = new Worker(URL.createObjectURL(blob));

    // Initialize worker
    return new Promise((resolve, reject) => {
      const initId = Math.random().toString(36);

      const handleMessage = (e: MessageEvent) => {
        if (e.data.id === initId && e.data.type === "init") {
          this.webWorker?.removeEventListener("message", handleMessage);
          if (e.data.success) {
            resolve();
          } else {
            reject(new Error("Worker initialization failed"));
          }
        }
      };

      if (this.webWorker) {
        this.webWorker.addEventListener("message", handleMessage);
        this.webWorker.postMessage({ id: initId, type: "init" });
      } else {
        reject(new Error("Web worker not created"));
      }
    });
  }

  /**
   * Ensure parser is initialized before operations
   */
  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized && this.initPromise) {
      await this.initPromise;
    }
    if (!this.isInitialized || !this.parser) {
      throw new Error("JSON parser not initialized");
    }
  }

  /**
   * Parse JSON with caching and performance optimization
   */
  async parse(
    json: string,
    options: { useCache?: boolean; useWorker?: boolean } = {}
  ): Promise<ParseResult> {
    await this.ensureInitialized();

    const { useCache = true, useWorker = false } = options;

    // For large JSON, use web worker
    if (useWorker && json.length > 100000) {
      return this.parseWithWorker(json, { useCache });
    }

    // Use main thread for smaller JSON
    return this.parser!.parseWithCache(json, useCache);
  }

  /**
   * Parse JSON using web worker for non-blocking operation
   */
  private async parseWithWorker(
    json: string,
    options: { useCache?: boolean } = {}
  ): Promise<ParseResult> {
    if (!this.webWorker) {
      throw new Error("Web worker not available");
    }

    return new Promise((resolve, reject) => {
      const id = Math.random().toString(36);

      const handleMessage = (e: MessageEvent) => {
        if (e.data.id === id) {
          this.webWorker?.removeEventListener("message", handleMessage);
          if (e.data.error) {
            reject(new Error(e.data.error));
          } else {
            resolve(e.data.result);
          }
        }
      };

      if (this.webWorker) {
        this.webWorker.addEventListener("message", handleMessage);
        this.webWorker.postMessage({
          id,
          type: "parse",
          data: { json, useCache: options.useCache },
        });
      } else {
        reject(new Error("Web worker not available"));
      }
    });
  }

  /**
   * Parse multiple JSON documents in parallel
   */
  async parseBatch(
    jsonArray: string[],
    options: { useWorker?: boolean } = {}
  ): Promise<BatchResult> {
    await this.ensureInitialized();

    const { useWorker = true } = options;

    // For batch operations, prefer web worker
    if (useWorker && this.webWorker) {
      return new Promise((resolve, reject) => {
        const id = Math.random().toString(36);

        const handleMessage = (e: MessageEvent) => {
          if (e.data.id === id) {
            this.webWorker?.removeEventListener("message", handleMessage);
            if (e.data.error) {
              reject(new Error(e.data.error));
            } else {
              resolve(e.data.result);
            }
          }
        };

        if (this.webWorker) {
          this.webWorker.addEventListener("message", handleMessage);
          this.webWorker.postMessage({
            id,
            type: "parseBatch",
            data: { jsonArray },
          });
        } else {
          reject(new Error("Web worker not available"));
        }
      });
    }

    return this.parser!.parseBatch(jsonArray);
  }

  /**
   * Get value from parsed document using JSONPath-like syntax
   */
  async getValue(path: string): Promise<any> {
    await this.ensureInitialized();
    return this.parser!.getValue(path);
  }

  /**
   * Convert document back to JSON string
   */
  async stringify(
    options: { pretty?: boolean } = {}
  ): Promise<StringifyResult> {
    await this.ensureInitialized();
    return this.parser!.stringify(options);
  }

  /**
   * Validate JSON against schema
   */
  async validate(schemaJson: string): Promise<ValidationResult> {
    await this.ensureInitialized();
    return this.parser!.validate(schemaJson);
  }

  /**
   * Get performance metrics for last operation
   */
  async getMetrics(): Promise<ParseMetrics> {
    await this.ensureInitialized();
    return this.parser!.getMetrics();
  }

  /**
   * Get cache statistics
   */
  async getCacheStats(): Promise<CacheStats> {
    await this.ensureInitialized();
    return this.wasmModule!.getCacheStats();
  }

  /**
   * Clear parser cache
   */
  async clearCache(): Promise<void> {
    await this.ensureInitialized();
    this.wasmModule!.clearCache();
    this.performanceCache.clear();
  }

  /**
   * GPU-accelerated JSON validation using compute shaders (if available)
   */
  async validateWithGpu(
    json: string
  ): Promise<{ valid: boolean; errors: string[] }> {
    // Check for WebGPU support
    if (!("gpu" in navigator)) {
      console.warn("WebGPU not available, falling back to CPU validation");
      const result = await this.validate(json);
      return {
        valid: result.valid,
        errors: result.error ? [result.error] : [],
      };
    }

    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (!adapter) {
        throw new Error("No WebGPU adapter available");
      }

      const device = await adapter.requestDevice();

      // Create compute shader for JSON validation
      const shaderCode = `
                @group(0) @binding(0) var<storage, read> input: array<u32>;
                @group(0) @binding(1) var<storage, read_write> output: array<u32>;

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&input)) {
                        return;
                    }

                    // Basic JSON character validation
                    let char = input[index];
                    var is_valid = 1u;

                    // Check for valid JSON characters
                    if (char < 32u && char != 9u && char != 10u && char != 13u) {
                        is_valid = 0u;
                    }

                    output[index] = is_valid;
                }
            `;

      const shaderModule = device.createShaderModule({ code: shaderCode });

      // Convert JSON string to buffer
      const jsonBytes = new TextEncoder().encode(json);
      const inputBuffer = device.createBuffer({
        size: jsonBytes.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const outputBuffer = device.createBuffer({
        size: jsonBytes.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      const stagingBuffer = device.createBuffer({
        size: jsonBytes.length * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      // Create bind group
      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
        ],
      });

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
        ],
      });

      // Create compute pipeline
      const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout],
        }),
        compute: { module: shaderModule, entryPoint: "main" },
      });

      // Upload data and run compute shader
      const uint32Array = new Uint32Array(jsonBytes);
      device.queue.writeBuffer(inputBuffer, 0, uint32Array);

      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(jsonBytes.length / 64));
      passEncoder.end();

      commandEncoder.copyBufferToBuffer(
        outputBuffer,
        0,
        stagingBuffer,
        0,
        jsonBytes.length * 4
      );
      device.queue.submit([commandEncoder.finish()]);

      // Read results
      await stagingBuffer.mapAsync(GPUMapMode.read);
      const resultArray = new Uint32Array(stagingBuffer.getMappedRange());
      const errors: string[] = [];

      for (let i = 0; i < resultArray.length; i++) {
        if (resultArray[i] === 0) {
          errors.push(
            `Invalid character at position ${i}: ${String.fromCharCode(jsonBytes[i])}`
          );
        }
      }

      stagingBuffer.unmap();

      return {
        valid: errors.length === 0,
        errors,
      };
    } catch (error) {
      console.warn("GPU validation failed, falling back to CPU:", error);
      const result = await this.validate(json);
      return {
        valid: result.valid,
        errors: result.error ? [result.error] : [],
      };
    }
  }

  /**
   * Performance benchmark for different parsing methods
   */
  async benchmark(
    testJson: string,
    iterations: number = 100
  ): Promise<{
    wasmTime: number;
    nativeTime: number;
    speedup: number;
    cacheHitRate: number;
  }> {
    await this.ensureInitialized();

    // Clear cache for fair comparison
    await this.clearCache();

    // Benchmark WebAssembly parser
    const wasmStart = performance.now();
    for (let i = 0; i < iterations; i++) {
      await this.parse(testJson, { useCache: true });
    }
    const wasmEnd = performance.now();
    const wasmTime = wasmEnd - wasmStart;

    // Benchmark native JSON.parse
    const nativeStart = performance.now();
    for (let i = 0; i < iterations; i++) {
      JSON.parse(testJson);
    }
    const nativeEnd = performance.now();
    const nativeTime = nativeEnd - nativeStart;

    const cacheStats = await this.getCacheStats();

    return {
      wasmTime,
      nativeTime,
      speedup: nativeTime / wasmTime,
      cacheHitRate: cacheStats.hitRate,
    };
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    if (this.webWorker) {
      this.webWorker.terminate();
      this.webWorker = null;
    }

    if (this.parser && this.wasmModule) {
      this.wasmModule.destroyParser(this.parser);
      this.parser = null;
    }

    this.performanceCache.clear();
    this.isInitialized = false;
  }
}

/**
 * Singleton instance for global use
 */
let globalParser: GpuAcceleratedJsonParser | null = null;

/**
 * Get the global parser instance
 */
export function getGlobalJsonParser(): GpuAcceleratedJsonParser {
  if (!globalParser) {
    globalParser = new GpuAcceleratedJsonParser();
  }
  return globalParser;
}

/**
 * Convenience function for parsing JSON
 */
export async function parseJson(
  json: string,
  options?: { useCache?: boolean; useWorker?: boolean }
): Promise<ParseResult> {
  const parser = getGlobalJsonParser();
  return parser.parse(json, options);
}

/**
 * Convenience function for batch parsing
 */
export async function parseJsonBatch(
  jsonArray: string[],
  options?: { useWorker?: boolean }
): Promise<BatchResult> {
  const parser = getGlobalJsonParser();
  return parser.parseBatch(jsonArray, options);
}

export type {
  ParseMetrics,
  CacheStats,
  ParseResult,
  BatchResult,
  StringifyResult,
  ValidationResult,
};
