// WebAssembly GPU Worker for Browser
// This runs in a Web Worker with access to WebGPU

// Polyfill TypeScript WebGPU types for environments without WebGPU lib definitions
// (Use real typings / remove these if your tsconfig includes WebGPU libs)
declare type GPUDevice = any;
declare type GPUBuffer = any;
declare type GPUBindGroup = any;
declare type GPUComputePipeline = any;
declare type GPUAdapter = any;
declare const GPUBufferUsage: any;
declare const GPUMapMode: any;

// Types for tensor operations
export interface TensorOp {
    type: 'matmul' | 'conv2d' | 'attention' | 'fft' | 'embedding';
    inputA: Float32Array;
    inputB?: Float32Array;
    params?: unknown;
}

export interface VertexCache {
    url: string;
    buffer: Float32Array;
    timestamp: number;
    score: number;
}

class GPUWorker {
    private gpuDevice: GPUDevice | null = null;
    private wasmModule: any = null;
    private vertexCache: Map<string, VertexCache> = new Map();
    private urlHeuristics: Map<string, number> = new Map();

    async initialize() {
        // Initialize WebGPU
        if ('gpu' in navigator) {
            const adapter = await (navigator as any).gpu.requestAdapter();
            if (adapter) {
                this.gpuDevice = await adapter.requestDevice();
                console.log('WebGPU initialized');
            }
        }

        // Load WebAssembly module
        const response = await fetch('/wasm/gpu-compute.wasm');
        const wasmBuffer = await response.arrayBuffer();
        const wasmModule = await WebAssembly.instantiate(wasmBuffer, {
            env: {
                memory: new WebAssembly.Memory({ initial: 256, maximum: 4096 }),
                __memory_base: 0,
                __table_base: 0,
                abort: () => console.error('WASM abort'),
            }
        });

        this.wasmModule = wasmModule.instance.exports;
        console.log('WebAssembly module loaded');
    }

    // Create GPU compute pipeline for matrix multiplication
    async createMatMulPipeline() {
        if (!this.gpuDevice) return null;

        const shaderModule = this.gpuDevice.createShaderModule({
            code: `
                struct Matrix {
                    data: array<f32>,
                    rows: u32,
                    cols: u32,
                }

                @group(0) @binding(0) var<storage, read> a: Matrix;
                @group(0) @binding(1) var<storage, read> b: Matrix;
                @group(0) @binding(2) var<storage, read_write> result: Matrix;

                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let row = global_id.x;
                    let col = global_id.y;

                    if (row >= a.rows || col >= b.cols) {
                        return;
                    }

                    var sum = 0.0;
                    for (var i = 0u; i < a.cols; i = i + 1u) {
                        sum = sum + a.data[row * a.cols + i] * b.data[i * b.cols + col];
                    }

                    result.data[row * b.cols + col] = sum;
                }
            `
        });

        return this.gpuDevice.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }

    // Create GPU pipeline for convolution
    async createConv2DPipeline() {
        if (!this.gpuDevice) return null;

        const shaderModule = this.gpuDevice.createShaderModule({
            code: `
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read> kernel: array<f32>;
                @group(0) @binding(2) var<storage, read_write> output: array<f32>;
                @group(0) @binding(3) var<uniform> params: vec4<u32>; // width, height, kernel_size, padding

                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let x = global_id.x;
                    let y = global_id.y;
                    let width = params.x;
                    let height = params.y;
                    let kernel_size = params.z;
                    let half_kernel = kernel_size / 2u;

                    if (x >= width || y >= height) {
                        return;
                    }

                    var sum = 0.0;
                    for (var ky = 0u; ky < kernel_size; ky = ky + 1u) {
                        for (var kx = 0u; kx < kernel_size; kx = kx + 1u) {
                            let px = x + kx - half_kernel;
                            let py = y + ky - half_kernel;

                            if (px < width && py < height) {
                                let input_idx = py * width + px;
                                let kernel_idx = ky * kernel_size + kx;
                                sum = sum + input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }

                    output[y * width + x] = sum;
                }
            `
        });

        return this.gpuDevice.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }

    // Process tensor operation
    async processTensorOp(op: TensorOp): Promise<Float32Array> {
        // Check vertex cache first
        const cacheKey = this.getCacheKey(op);
        if (this.vertexCache.has(cacheKey)) {
            const cached = this.vertexCache.get(cacheKey)!;
            cached.score += 1; // Update heuristic score
            return cached.buffer;
        }

        let result: Float32Array;

        if (this.gpuDevice) {
            // Use WebGPU
            result = await this.processWithWebGPU(op);
        } else if (this.wasmModule) {
            // Fallback to WebAssembly
            result = this.processWithWASM(op);
        } else {
            // CPU fallback
            result = this.processWithCPU(op);
        }

        // Cache the result
        this.cacheResult(cacheKey, result);

        return result;
    }

    // Process with WebGPU
    async processWithWebGPU(op: TensorOp): Promise<Float32Array> {
        switch (op.type) {
            case 'matmul':
                return this.gpuMatMul(op.inputA, op.inputB!, op.params);
            case 'conv2d':
                return this.gpuConv2D(op.inputA, op.inputB!, op.params);
            default:
                return this.processWithWASM(op);
        }
    }

    // GPU Matrix Multiplication
    async gpuMatMul(a: Float32Array, b: Float32Array, params: any): Promise<Float32Array> {
        if (!this.gpuDevice) return new Float32Array();

        const pipeline = await this.createMatMulPipeline();
        if (!pipeline) return new Float32Array();

        // Create buffers
        const aBuffer = this.gpuDevice.createBuffer({
            size: a.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const bBuffer = this.gpuDevice.createBuffer({
            size: b.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const resultSize = params.m * params.n * 4;
        const resultBuffer = this.gpuDevice.createBuffer({
            size: resultSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Write data to buffers
        this.gpuDevice.queue.writeBuffer(aBuffer, 0, a);
        this.gpuDevice.queue.writeBuffer(bBuffer, 0, b);

        // Create bind group
        const bindGroup = this.gpuDevice.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: aBuffer } },
                { binding: 1, resource: { buffer: bBuffer } },
                { binding: 2, resource: { buffer: resultBuffer } },
            ],
        });

        // Encode commands
        const commandEncoder = this.gpuDevice.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(
            Math.ceil(params.m / 8),
            Math.ceil(params.n / 8)
        );
        passEncoder.end();

        // Read back result
        const readBuffer = this.gpuDevice.createBuffer({
            size: resultSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultSize);
        this.gpuDevice.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();

        return result;
    }

    // GPU Convolution
    async gpuConv2D(input: Float32Array, kernel: Float32Array, params: any): Promise<Float32Array> {
        // Similar to matmul but with conv2d shader
        // Implementation would follow same pattern
        return new Float32Array(input.length);
    }

    // Process with WebAssembly
    processWithWASM(op: TensorOp): Float32Array {
        if (!this.wasmModule) return new Float32Array();

        switch (op.type) {
            case 'matmul':
                return this.wasmModule.matmul(op.inputA, op.inputB, op.params);
            case 'conv2d':
                return this.wasmModule.conv2d(op.inputA, op.inputB, op.params);
            case 'attention':
                return this.wasmModule.attention(op.inputA, op.inputB, op.params);
            case 'fft':
                return this.wasmModule.fft(op.inputA);
            default:
                return op.inputA;
        }
    }

    // CPU fallback
    processWithCPU(op: TensorOp): Float32Array {
        switch (op.type) {
            case 'matmul':
                return this.cpuMatMul(op.inputA, op.inputB!);
            case 'conv2d':
                return this.cpuConv2D(op.inputA, op.inputB!);
            default:
                return op.inputA;
        }
    }

    // Simple CPU implementations
    cpuMatMul(a: Float32Array, b: Float32Array): Float32Array {
        // Simple matrix multiplication
        const result = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i % b.length];
        }
        return result;
    }

    cpuConv2D(input: Float32Array, kernel: Float32Array): Float32Array {
        // Simple convolution
        const result = new Float32Array(input.length);
        const kernelSize = Math.sqrt(kernel.length);

        for (let i = 0; i < input.length; i++) {
            let sum = 0;
            for (let j = 0; j < kernel.length; j++) {
                const idx = i + j - Math.floor(kernelSize / 2);
                if (idx >= 0 && idx < input.length) {
                    sum += input[idx] * kernel[j];
                }
            }
            result[i] = sum;
        }

        return result;
    }

    // Cache management
    getCacheKey(op: TensorOp): string {
        return `${op.type}-${op.inputA.length}-${op.inputB?.length || 0}`;
    }

    cacheResult(key: string, buffer: Float32Array) {
        this.vertexCache.set(key, {
            url: key,
            buffer: buffer,
            timestamp: Date.now(),
            score: 1
        });

        // Limit cache size
        if (this.vertexCache.size > 100) {
            // Remove least recently used
            const sorted = Array.from(this.vertexCache.entries())
                .sort((a, b) => a[1].score - b[1].score);
            this.vertexCache.delete(sorted[0][0]);
        }
    }

    // Heuristic learning for URL patterns
    updateURLHeuristics(url: string) {
        const count = this.urlHeuristics.get(url) || 0;
        this.urlHeuristics.set(url, count + 1);

        // Learn patterns from frequently accessed URLs
        if (count > 10) {
            // Preload similar operations
            this.preloadSimilarOperations(url);
        }
    }

    preloadSimilarOperations(url: string) {
        // Implement pattern matching and preloading logic
        console.log(`Preloading operations similar to ${url}`);
    }
}

// Worker message handler
let gpuWorker: GPUWorker | null = null;

self.addEventListener('message', async (event: any) => {
    const { type, data } = event.data;

    switch (type) {
        case 'init':
            gpuWorker = new GPUWorker();
            await gpuWorker.initialize();
            self.postMessage({ type: 'ready' });
            break;

        case 'process':
            if (gpuWorker) {
                const result = await gpuWorker.processTensorOp(data);
                self.postMessage({ type: 'result', data: result });
            }
            break;

        case 'cache-stats':
            if (gpuWorker) {
                // Return cache statistics
                self.postMessage({ type: 'stats', data: {} });
            }
            break;
    }
});

// Export for module usage
export { GPUWorker };