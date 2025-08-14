/**
 * WebGPU Client-Side Acceleration for Legal AI Processing
 * Provides GPU-accelerated vector similarity, matrix operations, and embedding computations
 */

export interface WebGPUCapabilities {
    available: boolean;
    adapter?: GPUAdapter;
    device?: GPUDevice;
    limits: {
        maxBufferSize: number;
        maxComputeWorkgroupSizeX: number;
        maxComputeWorkgroupSizeY: number;
        maxComputeWorkgroupSizeZ: number;
    };
    features: string[];
}

export interface VectorOperation {
    type: 'similarity' | 'clustering' | 'embedding' | 'matrix_multiply';
    inputA: Float32Array;
    inputB?: Float32Array;
    dimensions: number;
    options?: {
        similarity_threshold?: number;
        cluster_count?: number;
        iterations?: number;
    };
}

export interface ComputeResult {
    type: string;
    result: Float32Array;
    executionTime: number;
    workgroupsDispatched: number;
    bufferSize: number;
}

class WebGPUAccelerator {
    private capabilities: WebGPUCapabilities | null = null;
    private device: GPUDevice | null = null;
    private computeShaders: Map<string, GPUShaderModule> = new Map();
    private bufferPool: Map<string, GPUBuffer> = new Map();
    private initialized = false;

    // Compute shaders for legal AI operations
    private readonly shaderSources = {
        vectorSimilarity: `
            @group(0) @binding(0) var<storage, read> vectorA: array<f32>;
            @group(0) @binding(1) var<storage, read> vectorB: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            @group(0) @binding(3) var<uniform> params: vec4<u32>; // dimensions, threshold, reserved, reserved

            @compute @workgroup_size(64, 1, 1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let index = id.x;
                let dimensions = params.x;
                
                if (index >= dimensions) {
                    return;
                }
                
                // Compute dot product for cosine similarity
                var dotProduct = 0.0;
                var normA = 0.0;
                var normB = 0.0;
                
                for (var i = 0u; i < dimensions; i++) {
                    let a = vectorA[i];
                    let b = vectorB[i];
                    dotProduct += a * b;
                    normA += a * a;
                    normB += b * b;
                }
                
                // Cosine similarity
                let similarity = dotProduct / (sqrt(normA) * sqrt(normB));
                
                // Store result for this workgroup
                if (index == 0u) {
                    result[0] = similarity;
                }
            }
        `,

        kMeansClustering: `
            @group(0) @binding(0) var<storage, read> dataPoints: array<f32>;
            @group(0) @binding(1) var<storage, read_write> centroids: array<f32>;
            @group(0) @binding(2) var<storage, read_write> assignments: array<u32>;
            @group(0) @binding(3) var<uniform> params: vec4<u32>; // numPoints, dimensions, numClusters, iteration

            @compute @workgroup_size(256, 1, 1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let pointIndex = id.x;
                let numPoints = params.x;
                let dimensions = params.y;
                let numClusters = params.z;
                
                if (pointIndex >= numPoints) {
                    return;
                }
                
                var minDistance = 1e10;
                var bestCluster = 0u;
                
                // Find closest centroid
                for (var cluster = 0u; cluster < numClusters; cluster++) {
                    var distance = 0.0;
                    
                    for (var dim = 0u; dim < dimensions; dim++) {
                        let pointValue = dataPoints[pointIndex * dimensions + dim];
                        let centroidValue = centroids[cluster * dimensions + dim];
                        let diff = pointValue - centroidValue;
                        distance += diff * diff;
                    }
                    
                    if (distance < minDistance) {
                        minDistance = distance;
                        bestCluster = cluster;
                    }
                }
                
                assignments[pointIndex] = bestCluster;
            }
        `,

        matrixMultiply: `
            @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
            @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            @group(0) @binding(3) var<uniform> dimensions: vec4<u32>; // M, N, K, reserved

            @compute @workgroup_size(16, 16, 1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let M = dimensions.x;
                let N = dimensions.y;
                let K = dimensions.z;
                
                let row = id.x;
                let col = id.y;
                
                if (row >= M || col >= N) {
                    return;
                }
                
                var sum = 0.0;
                for (var k = 0u; k < K; k++) {
                    sum += matrixA[row * K + k] * matrixB[k * N + col];
                }
                
                result[row * N + col] = sum;
            }
        `,

        embeddingTransform: `
            @group(0) @binding(0) var<storage, read> embeddings: array<f32>;
            @group(0) @binding(1) var<storage, read> weights: array<f32>;
            @group(0) @binding(2) var<storage, read> bias: array<f32>;
            @group(0) @binding(3) var<storage, read_write> output: array<f32>;
            @group(0) @binding(4) var<uniform> params: vec4<u32>; // inputDim, outputDim, activation, reserved

            @compute @workgroup_size(256, 1, 1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let outputIndex = id.x;
                let inputDim = params.x;
                let outputDim = params.y;
                let activation = params.z;
                
                if (outputIndex >= outputDim) {
                    return;
                }
                
                var sum = bias[outputIndex];
                
                for (var i = 0u; i < inputDim; i++) {
                    sum += embeddings[i] * weights[i * outputDim + outputIndex];
                }
                
                // Apply activation function
                if (activation == 1u) { // ReLU
                    sum = max(0.0, sum);
                } else if (activation == 2u) { // Sigmoid
                    sum = 1.0 / (1.0 + exp(-sum));
                } else if (activation == 3u) { // Tanh
                    sum = tanh(sum);
                }
                
                output[outputIndex] = sum;
            }
        `
    };

    /**
     * Initialize WebGPU and check capabilities
     */
    async initialize(): Promise<WebGPUCapabilities> {
        if (this.initialized) {
            return this.capabilities!;
        }

        try {
            // Check WebGPU availability
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported in this browser');
            }

            // Request adapter
            const adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });

            if (!adapter) {
                throw new Error('No WebGPU adapter available');
            }

            // Request device
            const device = await adapter.requestDevice({
                requiredFeatures: [],
                requiredLimits: {
                    maxBufferSize: adapter.limits.maxBufferSize,
                    maxComputeWorkgroupSizeX: Math.min(1024, adapter.limits.maxComputeWorkgroupSizeX),
                }
            });

            this.device = device;

            // Get capabilities
            this.capabilities = {
                available: true,
                adapter,
                device,
                limits: {
                    maxBufferSize: adapter.limits.maxBufferSize,
                    maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
                    maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
                    maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
                },
                features: Array.from(adapter.features)
            };

            // Initialize compute shaders
            await this.initializeShaders();
            
            this.initialized = true;

            console.log('WebGPU initialized successfully:', this.capabilities);
            return this.capabilities;

        } catch (error) {
            console.warn('WebGPU initialization failed:', error);
            this.capabilities = {
                available: false,
                limits: {
                    maxBufferSize: 0,
                    maxComputeWorkgroupSizeX: 0,
                    maxComputeWorkgroupSizeY: 0,
                    maxComputeWorkgroupSizeZ: 0,
                },
                features: []
            };
            return this.capabilities;
        }
    }

    /**
     * Initialize compute shaders
     */
    private async initializeShaders(): Promise<void> {
        if (!this.device) return;

        for (const [name, source] of Object.entries(this.shaderSources)) {
            try {
                const shaderModule = this.device.createShaderModule({
                    label: `${name}_shader`,
                    code: source
                });

                this.computeShaders.set(name, shaderModule);
                console.log(`Initialized shader: ${name}`);
            } catch (error) {
                console.error(`Failed to create shader ${name}:`, error);
            }
        }
    }

    /**
     * Compute vector similarity using WebGPU
     */
    async computeVectorSimilarity(vectorA: Float32Array, vectorB: Float32Array): Promise<number> {
        if (!this.device || !this.initialized) {
            throw new Error('WebGPU not initialized');
        }

        if (vectorA.length !== vectorB.length) {
            throw new Error('Vectors must have the same dimensions');
        }

        const startTime = performance.now();
        const dimensions = vectorA.length;

        // Create buffers
        const bufferA = this.device.createBuffer({
            size: vectorA.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const bufferB = this.device.createBuffer({
            size: vectorB.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const resultBuffer = this.device.createBuffer({
            size: 4, // Single f32 result
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const paramsBuffer = this.device.createBuffer({
            size: 16, // vec4<u32>
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const readBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Write data to buffers
        this.device.queue.writeBuffer(bufferA, 0, vectorA);
        this.device.queue.writeBuffer(bufferB, 0, vectorB);
        this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([dimensions, 0, 0, 0]));

        // Create bind group
        const shader = this.computeShaders.get('vectorSimilarity')!;
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: resultBuffer } },
                { binding: 3, resource: { buffer: paramsBuffer } },
            ],
        });

        // Create compute pipeline
        const computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
                module: shader,
                entryPoint: 'main',
            },
        });

        // Execute computation
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 64));
        passEncoder.end();

        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, 4);
        this.device.queue.submit([commandEncoder.finish()]);

        // Read result
        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange());
        const similarity = result[0];
        readBuffer.unmap();

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        resultBuffer.destroy();
        paramsBuffer.destroy();
        readBuffer.destroy();

        const executionTime = performance.now() - startTime;
        console.log(`WebGPU vector similarity computed in ${executionTime.toFixed(2)}ms`);

        return similarity;
    }

    /**
     * Perform K-means clustering using WebGPU
     */
    async performKMeansClustering(dataPoints: Float32Array, dimensions: number, numClusters: number, iterations: number = 10): Promise<{ centroids: Float32Array; assignments: Uint32Array }> {
        if (!this.device || !this.initialized) {
            throw new Error('WebGPU not initialized');
        }

        const startTime = performance.now();
        const numPoints = dataPoints.length / dimensions;

        // Initialize centroids randomly
        const centroids = new Float32Array(numClusters * dimensions);
        for (let i = 0; i < centroids.length; i++) {
            centroids[i] = Math.random() * 2 - 1; // Random values between -1 and 1
        }

        const assignments = new Uint32Array(numPoints);

        // Create buffers
        const dataBuffer = this.device.createBuffer({
            size: dataPoints.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const centroidsBuffer = this.device.createBuffer({
            size: centroids.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        const assignmentsBuffer = this.device.createBuffer({
            size: assignments.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const paramsBuffer = this.device.createBuffer({
            size: 16, // vec4<u32>
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Write initial data
        this.device.queue.writeBuffer(dataBuffer, 0, dataPoints);
        this.device.queue.writeBuffer(centroidsBuffer, 0, centroids);

        // Create compute pipeline
        const shader = this.computeShaders.get('kMeansClustering')!;
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        const computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
                module: shader,
                entryPoint: 'main',
            },
        });

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: dataBuffer } },
                { binding: 1, resource: { buffer: centroidsBuffer } },
                { binding: 2, resource: { buffer: assignmentsBuffer } },
                { binding: 3, resource: { buffer: paramsBuffer } },
            ],
        });

        // K-means iterations
        for (let iter = 0; iter < iterations; iter++) {
            // Update parameters
            this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numPoints, dimensions, numClusters, iter]));

            // Execute clustering assignment
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(numPoints / 256));
            passEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            // TODO: Update centroids based on assignments (would need additional compute pass)
        }

        // Read results
        const assignmentsReadBuffer = this.device.createBuffer({
            size: assignments.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const centroidsReadBuffer = this.device.createBuffer({
            size: centroids.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(assignmentsBuffer, 0, assignmentsReadBuffer, 0, assignments.byteLength);
        commandEncoder.copyBufferToBuffer(centroidsBuffer, 0, centroidsReadBuffer, 0, centroids.byteLength);
        this.device.queue.submit([commandEncoder.finish()]);

        await assignmentsReadBuffer.mapAsync(GPUMapMode.READ);
        await centroidsReadBuffer.mapAsync(GPUMapMode.READ);

        const finalAssignments = new Uint32Array(assignmentsReadBuffer.getMappedRange());
        const finalCentroids = new Float32Array(centroidsReadBuffer.getMappedRange());

        assignmentsReadBuffer.unmap();
        centroidsReadBuffer.unmap();

        // Cleanup
        dataBuffer.destroy();
        centroidsBuffer.destroy();
        assignmentsBuffer.destroy();
        paramsBuffer.destroy();
        assignmentsReadBuffer.destroy();
        centroidsReadBuffer.destroy();

        const executionTime = performance.now() - startTime;
        console.log(`WebGPU K-means clustering completed in ${executionTime.toFixed(2)}ms`);

        return {
            centroids: finalCentroids,
            assignments: new Uint32Array(finalAssignments)
        };
    }

    /**
     * Accelerated matrix multiplication
     */
    async matrixMultiply(matrixA: Float32Array, matrixB: Float32Array, M: number, N: number, K: number): Promise<Float32Array> {
        if (!this.device || !this.initialized) {
            throw new Error('WebGPU not initialized');
        }

        const startTime = performance.now();
        const resultSize = M * N;

        // Create buffers
        const bufferA = this.device.createBuffer({
            size: matrixA.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const bufferB = this.device.createBuffer({
            size: matrixB.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const resultBuffer = this.device.createBuffer({
            size: resultSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const dimensionsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Write data
        this.device.queue.writeBuffer(bufferA, 0, matrixA);
        this.device.queue.writeBuffer(bufferB, 0, matrixB);
        this.device.queue.writeBuffer(dimensionsBuffer, 0, new Uint32Array([M, N, K, 0]));

        // Create compute pipeline
        const shader = this.computeShaders.get('matrixMultiply')!;
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: resultBuffer } },
                { binding: 3, resource: { buffer: dimensionsBuffer } },
            ],
        });

        const computePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
                module: shader,
                entryPoint: 'main',
            },
        });

        // Execute
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(M / 16), Math.ceil(N / 16));
        passEncoder.end();

        // Copy result
        const readBuffer = this.device.createBuffer({
            size: resultSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, resultSize * 4);
        this.device.queue.submit([commandEncoder.finish()]);

        // Read result
        await readBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readBuffer.getMappedRange());
        const finalResult = new Float32Array(result);
        readBuffer.unmap();

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        resultBuffer.destroy();
        dimensionsBuffer.destroy();
        readBuffer.destroy();

        const executionTime = performance.now() - startTime;
        console.log(`WebGPU matrix multiplication (${M}x${K} * ${K}x${N}) completed in ${executionTime.toFixed(2)}ms`);

        return finalResult;
    }

    /**
     * Get performance metrics
     */
    getPerformanceMetrics(): any {
        return {
            initialized: this.initialized,
            capabilities: this.capabilities,
            shadersLoaded: this.computeShaders.size,
            buffersActive: this.bufferPool.size
        };
    }

    /**
     * Cleanup resources
     */
    dispose(): void {
        // Clear buffer pool
        for (const buffer of this.bufferPool.values()) {
            buffer.destroy();
        }
        this.bufferPool.clear();

        // Clear compute shaders
        this.computeShaders.clear();

        // Reset state
        this.device = null;
        this.capabilities = null;
        this.initialized = false;
    }
}

// Export singleton instance
export const webGPUAccelerator = new WebGPUAccelerator();