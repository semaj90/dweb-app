// WebGPU Polyfill with WebGL fallback for vector operations
// Provides GPU acceleration for legal AI vector processing with fallback support

import type { WebGPUDevice, WebGPUComputeShader, WebGPUVectorOperation } from '$lib/types/vector-jobs';

export class WebGPUPolyfill {
	private device: GPUDevice | null = null;
	private adapter: GPUAdapter | null = null;
	private queue: GPUCommandQueue | null = null;
	private isWebGPUAvailable = false;
	private webglFallback: WebGL2RenderingContext | null = null;
	private canvas: HTMLCanvasElement | null = null;

	// Shader cache
	private shaderCache = new Map<string, WebGPUComputeShader>();
	
	// Performance tracking
	private performanceStats = {
		operationsCompleted: 0,
		totalProcessingTime: 0,
		averageProcessingTime: 0,
		webgpuOpsCount: 0,
		webglOpsCount: 0
	};

	async initialize(): Promise<boolean> {
		// Try WebGPU first
		if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
			try {
				this.adapter = await (navigator as any).gpu.requestAdapter({
					powerPreference: 'high-performance'
				});

				if (this.adapter) {
					this.device = await this.adapter.requestDevice({
						requiredFeatures: [],
						requiredLimits: {
							maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize,
							maxComputeWorkgroupStorageSize: this.adapter.limits.maxComputeWorkgroupStorageSize,
							maxComputeInvocationsPerWorkgroup: this.adapter.limits.maxComputeInvocationsPerWorkgroup
						}
					});

					this.queue = this.device.queue;
					this.isWebGPUAvailable = true;
					
					console.log('🔥 WebGPU initialized successfully');
					console.log('GPU:', this.adapter.info);
					
					return true;
				}
			} catch (error: any) {
				console.warn('WebGPU initialization failed, falling back to WebGL:', error);
			}
		}

		// Fallback to WebGL2
		return this.initializeWebGLFallback();
	}

	private initializeWebGLFallback(): boolean {
		try {
			if (typeof document === 'undefined') {
				return false; // Server-side, no WebGL available
			}

			this.canvas = document.createElement('canvas');
			this.webglFallback = this.canvas.getContext('webgl2', {
				powerPreference: 'high-performance'
			});

			if (!this.webglFallback) {
				console.error('WebGL2 not available');
				return false;
			}

			// Check for required extensions
			const requiredExtensions = [
				'EXT_color_buffer_float',
				'OES_texture_float_linear',
				'WEBGL_debug_renderer_info'
			];

			for (const ext of requiredExtensions) {
				if (!this.webglFallback.getExtension(ext)) {
					console.warn(`WebGL extension ${ext} not available`);
				}
			}

			console.log('✅ WebGL2 fallback initialized');
			console.log('Renderer:', this.webglFallback.getParameter(this.webglFallback.RENDERER));
			
			return true;
		} catch (error: any) {
			console.error('WebGL initialization failed:', error);
			return false;
		}
	}

	getDeviceInfo(): WebGPUDevice {
		if (this.isWebGPUAvailable && this.device && this.adapter) {
			return {
				device: this.device,
				queue: this.queue!,
				adapter: this.adapter,
				features: Array.from(this.device.features),
				limits: Object.fromEntries(
					Object.entries(this.device.limits).map(([key, value]) => [key, Number(value)])
				),
				isAvailable: true
			};
		}

		return {
			device: null as any,
			queue: null as any,
			adapter: null as any,
			features: [],
			limits: {},
			isAvailable: false
		};
	}

	// Vector embedding computation using WebGPU compute shaders
	async computeEmbedding(inputVector: number[], dimensions: number = 384): Promise<number[]> {
		const startTime = performance.now();

		try {
			let result: number[];

			if (this.isWebGPUAvailable) {
				result = await this.computeEmbeddingWebGPU(inputVector, dimensions);
				this.performanceStats.webgpuOpsCount++;
			} else if (this.webglFallback) {
				result = await this.computeEmbeddingWebGL(inputVector, dimensions);
				this.performanceStats.webglOpsCount++;
			} else {
				// CPU fallback
				result = this.computeEmbeddingCPU(inputVector, dimensions);
			}

			const processingTime = performance.now() - startTime;
			this.updatePerformanceStats(processingTime);

			return result;
		} catch (error: any) {
			console.error('Embedding computation failed:', error);
			// Always fall back to CPU computation
			return this.computeEmbeddingCPU(inputVector, dimensions);
		}
	}

	private async computeEmbeddingWebGPU(inputVector: number[], dimensions: number): Promise<number[]> {
		if (!this.device) throw new Error('WebGPU device not available');

		// Get or create compute shader
		const shaderKey = `embedding_${dimensions}`;
		let shader = this.shaderCache.get(shaderKey);

		if (!shader) {
			shader = await this.createEmbeddingShader(dimensions);
			this.shaderCache.set(shaderKey, shader);
		}

		// Create buffers
		const inputBuffer = this.device.createBuffer({
			size: inputVector.length * 4, // 4 bytes per float32
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		const outputBuffer = this.device.createBuffer({
			size: dimensions * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		const resultBuffer = this.device.createBuffer({
			size: dimensions * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});

		// Write input data
		this.queue!.writeBuffer(inputBuffer, 0, new Float32Array(inputVector));

		// Create bind group
		const bindGroup = this.device.createBindGroup({
			layout: shader.bindGroupLayout,
			entries: [
				{ binding: 0, resource: { buffer: inputBuffer } },
				{ binding: 1, resource: { buffer: outputBuffer } }
			],
		});

		// Dispatch compute shader
		const commandEncoder = this.device.createCommandEncoder();
		const computePass = commandEncoder.beginComputePass();
		
		computePass.setPipeline(shader.pipeline);
		computePass.setBindGroup(0, bindGroup);
		computePass.dispatchWorkgroups(Math.ceil(dimensions / 256));
		computePass.end();

		// Copy result
		commandEncoder.copyBufferToBuffer(outputBuffer, 0, resultBuffer, 0, dimensions * 4);
		this.queue!.submit([commandEncoder.finish()]);

		// Read result
		await resultBuffer.mapAsync(GPUMapMode.READ);
		const result = new Float32Array(resultBuffer.getMappedRange());
		const output = Array.from(result);
		
		resultBuffer.unmap();

		// Cleanup buffers
		inputBuffer.destroy();
		outputBuffer.destroy();
		resultBuffer.destroy();

		return output;
	}

	private async createEmbeddingShader(dimensions: number): Promise<WebGPUComputeShader> {
		if (!this.device) throw new Error('WebGPU device not available');

		const shaderCode = `
			struct VectorData {
				values: array<f32>
			};

			@group(0) @binding(0) var<storage, read> input_vector: VectorData;
			@group(0) @binding(1) var<storage, read_write> output_vector: VectorData;

			@compute @workgroup_size(256)
			fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
				let index = global_id.x;
				if (index >= ${dimensions}u) {
					return;
				}

				// Simple embedding transformation: normalize and apply non-linear activation
				let input_size = arrayLength(&input_vector.values);
				var sum: f32 = 0.0;
				
				// Compute weighted sum with positional encoding
				for (var i: u32 = 0; i < input_size; i++) {
					let weight = sin(f32(index * i) * 0.001 + f32(index) * 0.1);
					sum += input_vector.values[i] * weight;
				}
				
				// Apply activation function and normalization
				output_vector.values[index] = tanh(sum * 0.1) * sqrt(f32(dimensions));
			}
		`;

		const module = this.device.createShaderModule({
			code: shaderCode,
		});

		const bindGroupLayout = this.device.createBindGroupLayout({
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
			],
		});

		const pipeline = this.device.createComputePipeline({
			layout: this.device.createPipelineLayout({
				bindGroupLayouts: [bindGroupLayout],
			}),
			compute: {
				module,
				entryPoint: 'main',
			},
		});

		return { module, pipeline, bindGroupLayout };
	}

	private async computeEmbeddingWebGL(inputVector: number[], dimensions: number): Promise<number[]> {
		if (!this.webglFallback || !this.canvas) throw new Error('WebGL not available');

		const gl = this.webglFallback;

		// Create and compile vertex shader
		const vertexShaderSource = `#version 300 es
			in vec2 a_position;
			void main() {
				gl_Position = vec4(a_position, 0.0, 1.0);
			}
		`;

		// Fragment shader for embedding computation
		const fragmentShaderSource = `#version 300 es
			precision highp float;
			
			uniform sampler2D u_input_texture;
			uniform int u_input_size;
			uniform int u_dimensions;
			uniform int u_output_index;
			
			out vec4 fragColor;
			
			void main() {
				int index = int(gl_FragCoord.x) + int(gl_FragCoord.y) * ${dimensions};
				if (index >= u_dimensions) discard;
				
				float sum = 0.0;
				for (int i = 0; i < u_input_size; i++) {
					vec2 texCoord = vec2(float(i % ${Math.ceil(Math.sqrt(inputVector.length))}), 
					                    float(i / ${Math.ceil(Math.sqrt(inputVector.length))})) / ${Math.ceil(Math.sqrt(inputVector.length))}.0;
					float inputValue = texture(u_input_texture, texCoord).r;
					float weight = sin(float(index * i) * 0.001 + float(index) * 0.1);
					sum += inputValue * weight;
				}
				
				float result = tanh(sum * 0.1) * sqrt(float(u_dimensions));
				fragColor = vec4(result, result, result, 1.0);
			}
		`;

		// Create and link shader program
		const program = this.createShaderProgram(gl, vertexShaderSource, fragmentShaderSource);
		if (!program) throw new Error('Failed to create WebGL shader program');

		// Setup framebuffer for computation
		const texture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, dimensions, 1, 0, gl.RGBA, gl.FLOAT, null);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

		const framebuffer = gl.createFramebuffer();
		gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

		// Setup input texture
		const inputTexture = gl.createTexture();
		const textureSize = Math.ceil(Math.sqrt(inputVector.length));
		const paddedInput = new Float32Array(textureSize * textureSize);
		for (let i = 0; i < inputVector.length; i++) {
			paddedInput[i] = inputVector[i];
		}

		gl.bindTexture(gl.TEXTURE_2D, inputTexture);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, textureSize, textureSize, 0, gl.RED, gl.FLOAT, paddedInput);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

		// Render computation
		gl.useProgram(program);
		gl.uniform1i(gl.getUniformLocation(program, 'u_input_texture'), 0);
		gl.uniform1i(gl.getUniformLocation(program, 'u_input_size'), inputVector.length);
		gl.uniform1i(gl.getUniformLocation(program, 'u_dimensions'), dimensions);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, inputTexture);

		gl.viewport(0, 0, dimensions, 1);
		gl.drawArrays(gl.TRIANGLES, 0, 6);

		// Read result
		const result = new Float32Array(dimensions * 4);
		gl.readPixels(0, 0, dimensions, 1, gl.RGBA, gl.FLOAT, result);

		// Extract only the red channel (we stored the result in all channels)
		const output = new Array(dimensions);
		for (let i = 0; i < dimensions; i++) {
			output[i] = result[i * 4];
		}

		// Cleanup
		gl.deleteTexture(texture);
		gl.deleteTexture(inputTexture);
		gl.deleteFramebuffer(framebuffer);
		gl.deleteProgram(program);

		return output;
	}

	private createShaderProgram(gl: WebGL2RenderingContext, vertexSource: string, fragmentSource: string): WebGLProgram | null {
		const vertexShader = this.compileShader(gl, gl.VERTEX_SHADER, vertexSource);
		const fragmentShader = this.compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

		if (!vertexShader || !fragmentShader) return null;

		const program = gl.createProgram();
		if (!program) return null;

		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);

		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			console.error('WebGL program linking failed:', gl.getProgramInfoLog(program));
			gl.deleteProgram(program);
			return null;
		}

		return program;
	}

	private compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null {
		const shader = gl.createShader(type);
		if (!shader) return null;

		gl.shaderSource(shader, source);
		gl.compileShader(shader);

		if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
			console.error('WebGL shader compilation failed:', gl.getShaderInfoLog(shader));
			gl.deleteShader(shader);
			return null;
		}

		return shader;
	}

	private computeEmbeddingCPU(inputVector: number[], dimensions: number): number[] {
		const output = new Array(dimensions);
		
		for (let i = 0; i < dimensions; i++) {
			let sum = 0;
			for (let j = 0; j < inputVector.length; j++) {
				const weight = Math.sin(i * j * 0.001 + i * 0.1);
				sum += inputVector[j] * weight;
			}
			output[i] = Math.tanh(sum * 0.1) * Math.sqrt(dimensions);
		}
		
		return output;
	}

	// Vector similarity computation
	async computeSimilarity(vector1: number[], vector2: number[]): Promise<number> {
		if (vector1.length !== vector2.length) {
			throw new Error('Vectors must have the same dimensions');
		}

		const startTime = performance.now();
		let similarity: number;

		try {
			if (this.isWebGPUAvailable) {
				similarity = await this.computeSimilarityWebGPU(vector1, vector2);
				this.performanceStats.webgpuOpsCount++;
			} else if (this.webglFallback) {
				similarity = await this.computeSimilarityWebGL(vector1, vector2);
				this.performanceStats.webglOpsCount++;
			} else {
				similarity = this.computeSimilarityCPU(vector1, vector2);
			}

			const processingTime = performance.now() - startTime;
			this.updatePerformanceStats(processingTime);

			return similarity;
		} catch (error: any) {
			console.error('Similarity computation failed:', error);
			return this.computeSimilarityCPU(vector1, vector2);
		}
	}

	private async computeSimilarityWebGPU(vector1: number[], vector2: number[]): Promise<number> {
		// WebGPU cosine similarity computation
		// Implementation similar to embedding computation but for dot product and norms
		// Simplified for brevity - would implement full compute shader
		return this.computeSimilarityCPU(vector1, vector2);
	}

	private async computeSimilarityWebGL(vector1: number[], vector2: number[]): Promise<number> {
		// WebGL similarity computation using fragment shaders
		// Simplified for brevity - would implement full WebGL computation
		return this.computeSimilarityCPU(vector1, vector2);
	}

	private computeSimilarityCPU(vector1: number[], vector2: number[]): number {
		let dotProduct = 0;
		let norm1 = 0;
		let norm2 = 0;

		for (let i = 0; i < vector1.length; i++) {
			dotProduct += vector1[i] * vector2[i];
			norm1 += vector1[i] * vector1[i];
			norm2 += vector2[i] * vector2[i];
		}

		const magnitude = Math.sqrt(norm1) * Math.sqrt(norm2);
		return magnitude === 0 ? 0 : dotProduct / magnitude;
	}

	private updatePerformanceStats(processingTime: number): void {
		this.performanceStats.operationsCompleted++;
		this.performanceStats.totalProcessingTime += processingTime;
		this.performanceStats.averageProcessingTime = 
			this.performanceStats.totalProcessingTime / this.performanceStats.operationsCompleted;
	}

	getPerformanceStats() {
		return {
			...this.performanceStats,
			webgpuPercentage: (this.performanceStats.webgpuOpsCount / this.performanceStats.operationsCompleted) * 100,
			webglPercentage: (this.performanceStats.webglOpsCount / this.performanceStats.operationsCompleted) * 100,
			isWebGPUAvailable: this.isWebGPUAvailable,
			hasWebGLFallback: !!this.webglFallback
		};
	}

	dispose(): void {
		// Cleanup WebGPU resources
		if (this.device) {
			this.device.destroy();
		}

		// Cleanup WebGL resources
		if (this.webglFallback && this.canvas) {
			const gl = this.webglFallback;
			const loseContext = gl.getExtension('WEBGL_lose_context');
			if (loseContext) {
				loseContext.loseContext();
			}
		}

		// Clear caches
		this.shaderCache.clear();
		
		console.log('🧹 WebGPU/WebGL resources cleaned up');
	}
}

// Singleton instance for global use
export const webgpuPolyfill = new WebGPUPolyfill();