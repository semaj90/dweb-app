import { browser } from '$app/environment';

export interface CudaHealthStatus {
	status: 'healthy' | 'unhealthy' | 'error';
	cuda_available: boolean;
	gpu_info?: {
		device_count: number;
		compute_capability: string;
		queue_size: number;
	};
	service?: string;
	timestamp?: number;
	error?: string;
	details?: string;
}

export interface CudaStats {
	cuda_stats: {
		cache_stats: {
			hit_rate: number;
			size: number;
		};
		cuda_info: {
			compute_capability: string;
			device_count: number;
			initialized: boolean;
		};
		queue_stats: {
			online: boolean;
			size: number;
		};
		service: string;
		t5_config: {
			model_size: string;
			num_layers: number;
			num_heads: number;
			hidden_size: number;
			vocab_size: number;
			max_position_embeddings: number;
			dropout_rate: number;
			use_gpu: boolean;
			cuda_device_id: number;
		};
		uptime: string;
	};
	timestamp: number;
	source: string;
}

export interface CudaComputeRequest {
	query: string;
	metadata?: Record<string, any>;
	options?: {
		use_cache?: boolean;
		priority?: 'low' | 'normal' | 'high';
		timeout?: number;
	};
}

export interface CudaComputeResponse {
	result?: any;
	processing_time?: number;
	cache_hit?: boolean;
	gpu_utilization?: number;
	error?: string;
	details?: string;
}

class CudaService {
	private baseUrl = '/api/cuda';

	/**
	 * Check CUDA server health and GPU availability
	 */
	async getHealth(): Promise<CudaHealthStatus> {
		if (!browser) {
			return {
				status: 'error',
				cuda_available: false,
				error: 'Not available server-side'
			};
		}

		try {
			const response = await fetch(`${this.baseUrl}/health`);
			return await response.json();
		} catch (error) {
			return {
				status: 'error',
				cuda_available: false,
				error: 'Failed to check CUDA health',
				details: error instanceof Error ? error.message : String(error)
			};
		}
	}

	/**
	 * Get detailed CUDA performance statistics
	 */
	async getStats(): Promise<CudaStats | { error: string; details?: string }> {
		if (!browser) {
			return { error: 'Not available server-side' };
		}

		try {
			const response = await fetch(`${this.baseUrl}/stats`);
			
			if (!response.ok) {
				const errorData = await response.json();
				return errorData;
			}

			return await response.json();
		} catch (error) {
			return {
				error: 'Failed to retrieve CUDA stats',
				details: error instanceof Error ? error.message : String(error)
			};
		}
	}

	/**
	 * Submit computation request to CUDA server
	 */
	async compute(request: CudaComputeRequest): Promise<CudaComputeResponse> {
		if (!browser) {
			return { error: 'Not available server-side' };
		}

		try {
			const response = await fetch(`${this.baseUrl}/compute`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify(request),
			});

			const result = await response.json();

			if (!response.ok) {
				return {
					error: result.error || 'CUDA computation failed',
					details: result.details
				};
			}

			return result;
		} catch (error) {
			return {
				error: 'Failed to submit CUDA computation',
				details: error instanceof Error ? error.message : String(error)
			};
		}
	}

	/**
	 * Enhanced compute with RTX tensor core optimization
	 */
	async enhancedCompute(
		query: string,
		options: {
			use_tensor_cores?: boolean;
			quantization?: '4bit' | '8bit' | 'fp16' | 'fp32';
			negative_latent_space?: boolean;
			graph_dimensions?: '3D' | '4D';
		} = {}
	): Promise<CudaComputeResponse> {
		return this.compute({
			query,
			metadata: {
				enhanced: true,
				rtx_optimization: true,
				...options
			},
			options: {
				use_cache: true,
				priority: 'high'
			}
		});
	}

	/**
	 * Check if CUDA server is ready for processing
	 */
	async isReady(): Promise<boolean> {
		const health = await this.getHealth();
		return health.status === 'healthy' && health.cuda_available;
	}

	/**
	 * Get RTX tensor core information
	 */
	async getTensorCoreInfo(): Promise<{
		available: boolean;
		compute_capability: string;
		tensor_core_generation?: string;
		performance_estimate?: string;
	}> {
		const health = await this.getHealth();
		
		if (!health.cuda_available || !health.gpu_info) {
			return { available: false, compute_capability: 'unknown' };
		}

		const capability = health.gpu_info.compute_capability;
		const capabilityFloat = parseFloat(capability);

		// RTX 3060 Ti has compute capability 8.6 (4th gen Tensor Cores)
		let tensorCoreGeneration = 'unknown';
		let performanceEstimate = 'unknown';

		if (capabilityFloat >= 8.6) {
			tensorCoreGeneration = '4th generation';
			performanceEstimate = 'Excellent (RTX 30/40 series)';
		} else if (capabilityFloat >= 7.5) {
			tensorCoreGeneration = '3rd generation';
			performanceEstimate = 'Very Good (RTX 20 series)';
		} else if (capabilityFloat >= 7.0) {
			tensorCoreGeneration = '2nd generation';
			performanceEstimate = 'Good (V100, Titan V)';
		} else if (capabilityFloat >= 6.0) {
			tensorCoreGeneration = '1st generation';
			performanceEstimate = 'Fair (P100 series)';
		}

		return {
			available: true,
			compute_capability: capability,
			tensor_core_generation: tensorCoreGeneration,
			performance_estimate: performanceEstimate
		};
	}
}

export const cudaService = new CudaService();
export default cudaService;