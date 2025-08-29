/**
 * WebGPU Tensor Acceleration System
 * Client-side GPU acceleration for legal AI operations
 * Complements server-side CUDA acceleration
 */

export interface WebGPUTensorConfig {
	deviceType: 'discrete' | 'integrated' | 'auto';
	powerPreference: 'high-performance' | 'low-power';
	enableDebug: boolean;
	maxBufferSize: number;
	shaderCacheEnabled: boolean;
}

export interface TensorOperation {
	id: string;
	type: 'vectorSimilarity' | 'embedding' | 'reduction' | 'transform';
	inputShapes: number[][];
	outputShape: number[];
	parameters: Record<string, unknown>;
}

export interface GPUMetrics {
	memoryUsage: number;
	computeUtilization: number;
	operationsPerSecond: number;
	averageLatency: number;
	totalOperations: number;
	errorCount: number;
	lastError?: string;
}

export class WebGPUTensorAccelerator {
	private device: GPUDevice | null = null;
	private adapter: GPUAdapter | null = null;
	private queue: GPUQueue | null = null;
	private config: WebGPUTensorConfig;
	private shaderCache = new Map<string, GPUShaderModule>();
	private bufferPool: GPUBuffer[] = [];
	private metrics: GPUMetrics = {
		memoryUsage: 0,
		computeUtilization: 0,
		operationsPerSecond: 0,
		averageLatency: 0,
		totalOperations: 0,
		errorCount: 0
	};
	private isInitialized = false;
	private operationQueue: TensorOperation[] = [];
	private processingQueue = false;

	constructor(config: Partial<WebGPUTensorConfig> = {}) {
		this.config = {
			deviceType: 'auto',
			powerPreference: 'high-performance',
			enableDebug: false,
			maxBufferSize: 256 * 1024 * 1024, // 256MB
			shaderCacheEnabled: true,
			...config
		};
	}

	async initialize(): Promise<boolean> {
		try {
			console.log('🚀 Initializing WebGPU Tensor Accelerator...');

			if (!navigator.gpu) {
				throw new Error('WebGPU not supported in this browser');
			}

			// Request GPU adapter
			this.adapter = await navigator.gpu.requestAdapter({
				powerPreference: this.config.powerPreference,
				forceFallbackAdapter: false
			});

			if (!this.adapter) {
				throw new Error('Failed to get WebGPU adapter');
			}

			console.log('📊 WebGPU Adapter Info:', {
				vendor: this.adapter.info?.vendor,
				architecture: this.adapter.info?.architecture,
				device: this.adapter.info?.device,
				limits: this.adapter.limits
			});

			// Request GPU device
			const requiredFeatures: GPUFeatureName[] = [];
			const availableFeatures = Array.from(this.adapter.features);
			
			// Enable useful features if available
			if (availableFeatures.includes('timestamp-query' as GPUFeatureName)) {
				requiredFeatures.push('timestamp-query' as GPUFeatureName);
			}

			this.device = await this.adapter.requestDevice({
				requiredFeatures,
				requiredLimits: {
					maxBufferSize: this.config.maxBufferSize,
					maxStorageBufferBindingSize: this.config.maxBufferSize,
					maxComputeWorkgroupStorageSize: 32768
				}
			});

			this.queue = this.device.queue;

			// Set up error handling
			this.device.addEventListener('uncapturederror', (event: any) => {
				this.metrics.errorCount++;
				this.metrics.lastError = event.error.message;
				console.error('WebGPU Error:', event.error);
			});

			// Initialize shader cache
			await this.initializeShaders();

			this.isInitialized = true;
			console.log('✅ WebGPU Tensor Accelerator initialized successfully');

			// Start metrics collection
			this.startMetricsCollection();

			return true;
		} catch (error: any) {
			console.error('💥 WebGPU initialization failed:', error);
			this.metrics.errorCount++;
			this.metrics.lastError = error.message;
			return false;
		}
	}

	private async initializeShaders(): Promise<void> {
		if (!this.device) throw new Error('Device not initialized');

		const shaders = {
			vectorSimilarity: `
				@group(0) @binding(0) var<storage, read> vectorA: array<f32>;
				@group(0) @binding(1) var<storage, read> vectorB: array<f32>;
				@group(0) @binding(2) var<storage, read_write> result: array<f32>;
				@group(0) @binding(3) var<uniform> params: vec4<f32>; // [size, 0, 0, 0]

				@compute @workgroup_size(256)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let index = global_id.x;
					let size = u32(params.x);
					
					if (index >= size) {
						return;
					}

					let a = vectorA[index];
					let b = vectorB[index];
					
					// Compute dot product component
					result[index] = a * b;
				}
			`,
			
			cosineReduction: `
				@group(0) @binding(0) var<storage, read> dotProducts: array<f32>;
				@group(0) @binding(1) var<storage, read> normA: array<f32>;
				@group(0) @binding(2) var<storage, read> normB: array<f32>;
				@group(0) @binding(3) var<storage, read_write> result: array<f32>;
				@group(0) @binding(4) var<uniform> params: vec4<f32>; // [size, 0, 0, 0]

				var<workgroup> temp: array<f32, 256>;

				@compute @workgroup_size(256)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
						@builtin(local_invocation_id) local_id: vec3<u32>,
						@builtin(workgroup_id) workgroup_id: vec3<u32>) {
					let index = global_id.x;
					let local_index = local_id.x;
					let size = u32(params.x);
					
					// Load data into shared memory
					if (index < size) {
						temp[local_index] = dotProducts[index];
					} else {
						temp[local_index] = 0.0;
					}
					
					workgroupBarrier();
					
					// Reduction within workgroup
					var stride = 128u;
					while (stride > 0u) {
						if (local_index < stride && index + stride < size) {
							temp[local_index] += temp[local_index + stride];
						}
						workgroupBarrier();
						stride = stride / 2u;
					}
					
					// Write result
					if (local_index == 0u) {
						let dot_sum = temp[0];
						let norm_a = normA[0];
						let norm_b = normB[0];
						let cosine = dot_sum / (norm_a * norm_b);
						result[workgroup_id.x] = cosine;
					}
				}
			`,

			embedding: `
				@group(0) @binding(0) var<storage, read> input: array<u32>;
				@group(0) @binding(1) var<storage, read> weights: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				@group(0) @binding(3) var<uniform> params: vec4<f32>; // [input_size, embedding_dim, vocab_size, 0]

				@compute @workgroup_size(256)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let output_index = global_id.x;
					let input_size = u32(params.x);
					let embedding_dim = u32(params.y);
					let vocab_size = u32(params.z);
					
					if (output_index >= input_size * embedding_dim) {
						return;
					}
					
					let token_idx = output_index / embedding_dim;
					let dim_idx = output_index % embedding_dim;
					let token_id = input[token_idx];
					
					if (token_id < vocab_size) {
						let weight_idx = token_id * embedding_dim + dim_idx;
						output[output_index] = weights[weight_idx];
					} else {
						output[output_index] = 0.0;
					}
				}
			`,

			matrixMultiply: `
				@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
				@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
				@group(0) @binding(2) var<storage, read_write> result: array<f32>;
				@group(0) @binding(3) var<uniform> params: vec4<f32>; // [M, N, K, 0]

				var<workgroup> tileA: array<array<f32, 16>, 16>;
				var<workgroup> tileB: array<array<f32, 16>, 16>;

				@compute @workgroup_size(16, 16)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
						@builtin(local_invocation_id) local_id: vec3<u32>) {
					let M = u32(params.x);
					let N = u32(params.y);
					let K = u32(params.z);
					
					let row = global_id.y;
					let col = global_id.x;
					let localRow = local_id.y;
					let localCol = local_id.x;
					
					var sum = 0.0;
					
					let numTiles = (K + 15u) / 16u;
					
					for (var tile = 0u; tile < numTiles; tile++) {
						let tileCol = tile * 16u + localCol;
						let tileRow = tile * 16u + localRow;
						
						// Load tile A
						if (row < M && tileCol < K) {
							tileA[localRow][localCol] = matrixA[row * K + tileCol];
						} else {
							tileA[localRow][localCol] = 0.0;
						}
						
						// Load tile B
						if (tileRow < K && col < N) {
							tileB[localRow][localCol] = matrixB[tileRow * N + col];
						} else {
							tileB[localRow][localCol] = 0.0;
						}
						
						workgroupBarrier();
						
						// Compute partial sum
						for (var k = 0u; k < 16u; k++) {
							sum += tileA[localRow][k] * tileB[k][localCol];
						}
						
						workgroupBarrier();
					}
					
					// Write result
					if (row < M && col < N) {
						result[row * N + col] = sum;
					}
				}
			`
		};

		for (const [name, source] of Object.entries(shaders)) {
			const shader = this.device.createShaderModule({
				label: `${name}Shader`,
				code: source
			});
			
			if (this.config.shaderCacheEnabled) {
				this.shaderCache.set(name, shader);
			}
		}
	}

	async calculateVectorSimilarity(vectorA: Float32Array, vectorB: Float32Array): Promise<number> {
		if (!this.isInitialized || !this.device) {
			throw new Error('WebGPU not initialized');
		}

		const start = performance.now();
		
		try {
			const size = Math.min(vectorA.length, vectorB.length);
			
			// Create buffers
			const bufferA = this.createBuffer(vectorA, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
			const bufferB = this.createBuffer(vectorB, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
			const resultBuffer = this.device.createBuffer({
				size: size * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
			const paramsBuffer = this.device.createBuffer({
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
			
			// Upload parameters
			this.queue!.writeBuffer(paramsBuffer, 0, new Float32Array([size, 0, 0, 0]));
			
			// Create bind group
			const shader = this.shaderCache.get('vectorSimilarity')!;
			const computePipeline = this.device.createComputePipeline({
				layout: 'auto',
				compute: {
					module: shader,
					entryPoint: 'main'
				}
			});
			
			const bindGroup = this.device.createBindGroup({
				layout: computePipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: bufferA } },
					{ binding: 1, resource: { buffer: bufferB } },
					{ binding: 2, resource: { buffer: resultBuffer } },
					{ binding: 3, resource: { buffer: paramsBuffer } }
				]
			});
			
			// Execute computation
			const commandEncoder = this.device.createCommandEncoder();
			const computePass = commandEncoder.beginComputePass();
			computePass.setPipeline(computePipeline);
			computePass.setBindGroup(0, bindGroup);
			computePass.dispatchWorkgroups(Math.ceil(size / 256));
			computePass.end();
			
			// Copy result to staging buffer
			const stagingBuffer = this.device.createBuffer({
				size: size * 4,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
			});
			
			commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, size * 4);
			this.queue!.submit([commandEncoder.finish()]);
			
			// Read result
			await stagingBuffer.mapAsync(GPUMapMode.READ);
			const resultArray = new Float32Array(stagingBuffer.getMappedRange());
			
			// Calculate cosine similarity
			let dotProduct = 0;
			let normA = 0;
			let normB = 0;
			
			for (let i = 0; i < size; i++) {
				dotProduct += resultArray[i];
				normA += vectorA[i] * vectorA[i];
				normB += vectorB[i] * vectorB[i];
			}
			
			const cosineSimilarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
			
			// Cleanup
			stagingBuffer.unmap();
			bufferA.destroy();
			bufferB.destroy();
			resultBuffer.destroy();
			stagingBuffer.destroy();
			paramsBuffer.destroy();
			
			// Update metrics
			const duration = performance.now() - start;
			this.updateMetrics(duration);
			
			return cosineSimilarity;
			
		} catch (error: any) {
			this.metrics.errorCount++;
			this.metrics.lastError = error.message;
			throw error;
		}
	}

	async batchProcessEmbeddings(queries: string[]): Promise<Float32Array[]> {
		const embeddings: Float32Array[] = [];
		
		for (const query of queries) {
			// Simple tokenization and embedding (in production, use proper tokenizer)
			const tokens = this.simpleTokenize(query);
			const embedding = await this.generateEmbedding(tokens);
			embeddings.push(embedding);
		}
		
		return embeddings;
	}

	private simpleTokenize(text: string): Uint32Array {
		// Simplified tokenization - replace with proper tokenizer
		const words = text.toLowerCase().split(/\W+/).filter(w => w.length > 0);
		const tokens = words.map(word => this.getTokenId(word));
		return new Uint32Array(tokens);
	}

	private getTokenId(word: string): number {
		// Simple hash-based token ID (replace with proper vocabulary)
		let hash = 0;
		for (let i = 0; i < word.length; i++) {
			hash = ((hash << 5) - hash + word.charCodeAt(i)) & 0xffffffff;
		}
		return Math.abs(hash) % 50000; // Assume vocab size of 50k
	}

	private async generateEmbedding(tokens: Uint32Array): Promise<Float32Array> {
		if (!this.device) throw new Error('Device not initialized');

		const embeddingDim = 384; // nomic-embed-text dimension
		const vocabSize = 50000;
		
		// Create random weights (in production, load pre-trained weights)
		const weights = new Float32Array(vocabSize * embeddingDim);
		for (let i = 0; i < weights.length; i++) {
			weights[i] = (Math.random() - 0.5) * 0.1;
		}
		
		// Create buffers
		const tokensBuffer = this.createBuffer(tokens, GPUBufferUsage.STORAGE);
		const weightsBuffer = this.createBuffer(weights, GPUBufferUsage.STORAGE);
		const outputBuffer = this.device.createBuffer({
			size: tokens.length * embeddingDim * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
		});
		const paramsBuffer = this.device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		
		// Upload parameters
		this.queue!.writeBuffer(paramsBuffer, 0, new Float32Array([tokens.length, embeddingDim, vocabSize, 0]));
		
		// Execute embedding computation
		const shader = this.shaderCache.get('embedding')!;
		const computePipeline = this.device.createComputePipeline({
			layout: 'auto',
			compute: {
				module: shader,
				entryPoint: 'main'
			}
		});
		
		const bindGroup = this.device.createBindGroup({
			layout: computePipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: tokensBuffer } },
				{ binding: 1, resource: { buffer: weightsBuffer } },
				{ binding: 2, resource: { buffer: outputBuffer } },
				{ binding: 3, resource: { buffer: paramsBuffer } }
			]
		});
		
		const commandEncoder = this.device.createCommandEncoder();
		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(computePipeline);
		computePass.setBindGroup(0, bindGroup);
		computePass.dispatchWorkgroups(Math.ceil((tokens.length * embeddingDim) / 256));
		computePass.end();
		
		// Copy and read result
		const stagingBuffer = this.device.createBuffer({
			size: tokens.length * embeddingDim * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		
		commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, tokens.length * embeddingDim * 4);
		this.queue!.submit([commandEncoder.finish()]);
		
		await stagingBuffer.mapAsync(GPUMapMode.READ);
		const result = new Float32Array(stagingBuffer.getMappedRange());
		
		// Average token embeddings
		const finalEmbedding = new Float32Array(embeddingDim);
		for (let i = 0; i < embeddingDim; i++) {
			let sum = 0;
			for (let j = 0; j < tokens.length; j++) {
				sum += result[j * embeddingDim + i];
			}
			finalEmbedding[i] = sum / tokens.length;
		}
		
		// Cleanup
		stagingBuffer.unmap();
		tokensBuffer.destroy();
		weightsBuffer.destroy();
		outputBuffer.destroy();
		stagingBuffer.destroy();
		paramsBuffer.destroy();
		
		return finalEmbedding;
	}

	private createBuffer(data: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
		const buffer = this.device!.createBuffer({
			size: data.byteLength,
			usage
		});
		this.queue!.writeBuffer(buffer, 0, data);
		return buffer;
	}

	private updateMetrics(duration: number): void {
		this.metrics.totalOperations++;
		this.metrics.averageLatency = (this.metrics.averageLatency * (this.metrics.totalOperations - 1) + duration) / this.metrics.totalOperations;
		this.metrics.operationsPerSecond = 1000 / this.metrics.averageLatency;
	}

	private startMetricsCollection(): void {
		setInterval(() => {
			if (this.adapter?.info) {
				// Update memory usage if available
				this.metrics.memoryUsage = (this.adapter as any).memoryUsage || 0;
			}
		}, 1000);
	}

	getMetrics(): GPUMetrics {
		return { ...this.metrics };
	}

	getCapabilities(): Record<string, unknown> {
		return {
			isSupported: !!navigator.gpu,
			isInitialized: this.isInitialized,
			adapter: this.adapter?.info || null,
			limits: this.adapter?.limits || null,
			features: this.adapter ? Array.from(this.adapter.features) : [],
			shaderCacheSize: this.shaderCache.size
		};
	}

	async cleanup(): Promise<void> {
		if (this.device) {
			this.device.destroy();
			this.device = null;
		}
		
		this.bufferPool.forEach(buffer => buffer.destroy());
		this.bufferPool = [];
		this.shaderCache.clear();
		this.isInitialized = false;
		
		console.log('🧹 WebGPU Tensor Accelerator cleaned up');
	}
}

// Singleton instance
let tensorAccelerator: WebGPUTensorAccelerator | null = null;

export async function initializeWebGPU(): Promise<WebGPUTensorAccelerator | null> {
	if (!tensorAccelerator) {
		tensorAccelerator = new WebGPUTensorAccelerator({
			powerPreference: 'high-performance',
			enableDebug: true,
			shaderCacheEnabled: true
		});
		
		const success = await tensorAccelerator.initialize();
		if (!success) {
			tensorAccelerator = null;
		}
	}
	
	return tensorAccelerator;
}

export function getWebGPUAccelerator(): WebGPUTensorAccelerator | null {
	return tensorAccelerator;
}