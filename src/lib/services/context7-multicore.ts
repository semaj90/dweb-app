/**
 * Context7 Multicore Service
 * Integrates with MCP servers, Go services, and multicore processing
 */

// Type definitions for Context7 Multicore integration
type Context7Response = {
	success: boolean;
	data?: unknown;
	error?: string;
	processingTime: number;
	workerId?: number;
	operation: string;
	timestamp: string;
};

type MCPServerConfig = {
	name: string;
	port: number;
	type: string;
	enabled: boolean;
};

type WorkerMetrics = {
	totalRequests: number;
	averageResponseTime: number;
	errorRate: number;
};

type TensorData = {
	id: string;
	shape: number[];
	data: number[];
	dtype: string;
};

type ProcessingJob = {
	id: string;
	operation: string;
	data: unknown;
	priority: 'high' | 'normal' | 'low';
	timeout: number;
	useGPU: boolean;
	createdAt: Date;
	status: 'pending' | 'processing' | 'completed' | 'error' | 'cancelled';
	assignedWorkerId?: number;
};

interface Context7MulticoreConfig {
	basePort: number;
	workerCount: number;
	enableGPU: boolean;
	mcpServers: MCPServerConfig[];
	goServices: GoServiceConfig[];
	loadBalancerPort: number;
}

interface GoServiceConfig {
	name: string;
	port: number;
	type: 'llama' | 'simd' | 'tensor' | 'rag' | 'recommendation';
	enabled: boolean;
	gpuEnabled?: boolean;
}

interface WorkerStatus {
	workerId: number;
	port: number;
	status: 'active' | 'idle' | 'busy' | 'error';
	activeJobs: number;
	totalProcessed: number;
	lastActivity: Date;
	capabilities: string[];
}

interface MulticoreMetrics {
	totalWorkers: number;
	activeWorkers: number;
	totalJobs: number;
	processingTime: number;
	throughput: number;
	errorRate: number;
	gpuUtilization?: number;
}

export class Context7MulticoreService {
	private config: Context7MulticoreConfig;
	private workers: Map<number, WorkerStatus> = new Map();
	private jobQueue: ProcessingJob[] = [];
	private metrics: MulticoreMetrics;
	private loadBalancerUrl: string;

	constructor(config?: Partial<Context7MulticoreConfig>) {
		this.config = {
			basePort: 4100,
			workerCount: 8,
			enableGPU: true,
			loadBalancerPort: 8099,
			mcpServers: [
				{
					name: 'context7-multicore',
					port: 4100,
					type: 'multicore',
					enabled: true
				},
				{
					name: 'context7-filesystem',
					port: 4101,
					type: 'filesystem',
					enabled: true
				}
			],
			goServices: [
				{
					name: 'go-llama-chat',
					port: 8099,
					type: 'llama',
					enabled: true,
					gpuEnabled: true
				},
				{
					name: 'enhanced-rag',
					port: 8094,
					type: 'rag',
					enabled: true
				},
				{
					name: 'recommendation-service',
					port: 8096,
					type: 'recommendation',
					enabled: true
				},
				{
					name: 'simd-parser',
					port: 8097,
					type: 'simd',
					enabled: true,
					gpuEnabled: true
				}
			],
			...config
		};

		this.loadBalancerUrl = `http://localhost:${this.config.loadBalancerPort}`;
		this.initializeMetrics();
	}

	private initializeMetrics(): void {
		this.metrics = {
			totalWorkers: this.config.workerCount,
			activeWorkers: 0,
			totalJobs: 0,
			processingTime: 0,
			throughput: 0,
			errorRate: 0,
			gpuUtilization: 0
		};
	}

	async initialize(): Promise<boolean> {
		try {
			console.log('🚀 Initializing Context7 Multicore Service...');

			// Check load balancer status
			const loadBalancerStatus = await this.checkLoadBalancer();
			if (!loadBalancerStatus) {
				console.warn('⚠️ Load balancer not responding, starting in degraded mode');
			}

			// Initialize workers
			await this.initializeWorkers();

			// Test MCP servers
			await this.testMCPServers();

			// Test Go services
			await this.testGoServices();

			console.log('✅ Context7 Multicore Service initialized successfully');
			return true;
		} catch (error) {
			console.error('❌ Failed to initialize Context7 Multicore Service:', error);
			return false;
		}
	}

	private async checkLoadBalancer(): Promise<boolean> {
		try {
			const controller = new AbortController();
			const timeoutId = setTimeout(() => controller.abort(), 3000);
			
			const response = await fetch(`${this.loadBalancerUrl}/status`, {
				signal: controller.signal
			});
			
			clearTimeout(timeoutId);
			return response.ok;
		} catch {
			return false;
		}
	}

	private async initializeWorkers(): Promise<void> {
		for (let i = 0; i < this.config.workerCount; i++) {
			const workerId = i + 1;
			const port = this.config.basePort + i;
			
			const worker: WorkerStatus = {
				workerId,
				port,
				status: 'idle',
				activeJobs: 0,
				totalProcessed: 0,
				lastActivity: new Date(),
				capabilities: ['memory-graph', 'semantic-search', 'error-analysis']
			};

			this.workers.set(workerId, worker);

			// Test worker connectivity
			try {
				const controller = new AbortController();
				const timeoutId = setTimeout(() => controller.abort(), 2000);
				
				const response = await fetch(`http://localhost:${port}/health`, {
					signal: controller.signal
				});
				
				clearTimeout(timeoutId);
				if (response.ok) {
					worker.status = 'active';
					this.metrics.activeWorkers++;
				} else {
					worker.status = 'error';
				}
			} catch {
				worker.status = 'error';
			}
		}

		console.log(`📊 Initialized ${this.metrics.activeWorkers}/${this.config.workerCount} workers`);
	}

	private async testMCPServers(): Promise<void> {
		for (const server of this.config.mcpServers) {
			if (!server.enabled) continue;

			try {
				const controller = new AbortController();
				const timeoutId = setTimeout(() => controller.abort(), 2000);
				
				const response = await fetch(`http://localhost:${server.port}/health`, {
					signal: controller.signal
				});
				
				clearTimeout(timeoutId);
				if (response.ok) {
					console.log(`✅ MCP Server ${server.name} is healthy`);
				} else {
					console.warn(`⚠️ MCP Server ${server.name} returned error status`);
				}
			} catch {
				console.warn(`⚠️ MCP Server ${server.name} is not responding`);
			}
		}
	}

	private async testGoServices(): Promise<void> {
		for (const service of this.config.goServices) {
			if (!service.enabled) continue;

			try {
				const controller = new AbortController();
				const timeoutId = setTimeout(() => controller.abort(), 2000);
				
				const response = await fetch(`http://localhost:${service.port}/status`, {
					signal: controller.signal
				});
				
				clearTimeout(timeoutId);
				if (response.ok) {
					console.log(`✅ Go Service ${service.name} is healthy`);
				} else {
					console.warn(`⚠️ Go Service ${service.name} returned error status`);
				}
			} catch {
				console.warn(`⚠️ Go Service ${service.name} is not responding`);
			}
		}
	}

	async processWithContext7(
		operation: string,
		data: unknown,
		options?: {
			priority?: 'high' | 'normal' | 'low';
			timeout?: number;
			useGPU?: boolean;
			workerPreference?: number;
		}
	): Promise<Context7Response> {
		const startTime = performance.now();
		const job: ProcessingJob = {
			id: `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
			operation,
			data,
			priority: options?.priority || 'normal',
			timeout: options?.timeout || 30000,
			useGPU: options?.useGPU || false,
			createdAt: new Date(),
			status: 'pending'
		};

		try {
			// Select appropriate worker
			const worker = this.selectWorker(job, options?.workerPreference);
			if (!worker) {
				throw new Error('No available workers');
			}

			// Update job and worker status
			job.status = 'processing';
			job.assignedWorkerId = worker.workerId;
			worker.status = 'busy';
			worker.activeJobs++;

			// Process based on operation type
			let result: unknown;
			switch (operation) {
				case 'memory-graph':
					result = await this.processMemoryGraph(worker, data);
					break;
				case 'semantic-search':
					result = await this.processSemanticSearch(worker, data);
					break;
				case 'error-analysis':
					result = await this.processErrorAnalysis(worker, data);
					break;
				case 'tensor-processing':
					result = await this.processTensorData(worker, data);
					break;
				case 'llama-chat':
					result = await this.processLlamaChat(data);
					break;
				case 'enhanced-rag':
					result = await this.processEnhancedRAG(data);
					break;
				default:
					result = await this.processGeneric(worker, operation, data);
			}

			// Update metrics
			const processingTime = performance.now() - startTime;
			this.updateMetrics(processingTime, true);
			
			// Update worker status
			worker.status = 'active';
			worker.activeJobs--;
			worker.totalProcessed++;
			worker.lastActivity = new Date();

			return {
				success: true,
				data: result,
				processingTime,
				workerId: worker.workerId,
				operation,
				timestamp: new Date().toISOString()
			};

		} catch (error) {
			this.updateMetrics(performance.now() - startTime, false);
			
			return {
				success: false,
				error: error instanceof Error ? error.message : 'Unknown error',
				processingTime: performance.now() - startTime,
				operation,
				timestamp: new Date().toISOString()
			};
		}
	}

	private selectWorker(job: ProcessingJob, preferredWorkerId?: number): WorkerStatus | null {
		// Try preferred worker first
		if (preferredWorkerId && this.workers.has(preferredWorkerId)) {
			const worker = this.workers.get(preferredWorkerId)!;
			if (worker.status === 'active' || worker.status === 'idle') {
				return worker;
			}
		}

		// Find best available worker
		const availableWorkers = Array.from(this.workers.values())
			.filter(w => w.status === 'active' || w.status === 'idle')
			.sort((a, b) => a.activeJobs - b.activeJobs);

		return availableWorkers.length > 0 ? availableWorkers[0] : null;
	}

	private async processMemoryGraph(worker: WorkerStatus, data: unknown): Promise<any> {
		const response = await fetch(`http://localhost:${worker.port}/mcp/memory/create-relations`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});

		if (!response.ok) {
			throw new Error(`Memory graph processing failed: ${response.statusText}`);
		}

		return response.json();
	}

	private async processSemanticSearch(worker: WorkerStatus, data: unknown): Promise<any> {
		const response = await fetch(`http://localhost:${worker.port}/mcp/memory/read-graph`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});

		if (!response.ok) {
			throw new Error(`Semantic search failed: ${response.statusText}`);
		}

		return response.json();
	}

	private async processErrorAnalysis(worker: WorkerStatus, data: unknown): Promise<any> {
		const response = await fetch(`http://localhost:${worker.port}/mcp/error-analysis/index`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});

		if (!response.ok) {
			throw new Error(`Error analysis failed: ${response.statusText}`);
		}

		return response.json();
	}

	private async processTensorData(worker: WorkerStatus, data: TensorData): Promise<any> {
		// Route to appropriate Go service based on data type
		const simdService = this.config.goServices.find(s => s.type === 'simd');
		if (!simdService || !simdService.enabled) {
			throw new Error('SIMD tensor service not available');
		}

		const response = await fetch(`http://localhost:${simdService.port}/api/tensor/process`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});

		if (!response.ok) {
			throw new Error(`Tensor processing failed: ${response.statusText}`);
		}

		return response.json();
	}

	private async processLlamaChat(data: unknown): Promise<any> {
		const llamaService = this.config.goServices.find(s => s.type === 'llama');
		if (!llamaService || !llamaService.enabled) {
			throw new Error('Go-Llama service not available');
		}

		const response = await fetch(`http://localhost:${llamaService.port}/api/chat`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});

		if (!response.ok) {
			throw new Error(`Llama chat failed: ${response.statusText}`);
		}

		return response.json();
	}

	private async processEnhancedRAG(data: unknown): Promise<any> {
		const ragService = this.config.goServices.find(s => s.type === 'rag');
		if (!ragService || !ragService.enabled) {
			throw new Error('Enhanced RAG service not available');
		}

		const response = await fetch(`http://localhost:${ragService.port}/api/rag`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data)
		});

		if (!response.ok) {
			throw new Error(`Enhanced RAG failed: ${response.statusText}`);
		}

		return response.json();
	}

	private async processGeneric(worker: WorkerStatus, operation: string, data: unknown): Promise<any> {
		// Route through load balancer for generic operations
		const response = await fetch(`${this.loadBalancerUrl}/api/process`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				operation,
				data,
				workerId: worker.workerId
			})
		});

		if (!response.ok) {
			throw new Error(`Generic processing failed: ${response.statusText}`);
		}

		return response.json();
	}

	private updateMetrics(processingTime: number, success: boolean): void {
		this.metrics.totalJobs++;
		this.metrics.processingTime = (this.metrics.processingTime + processingTime) / 2;
		
		if (!success) {
			this.metrics.errorRate = (this.metrics.errorRate + 1) / this.metrics.totalJobs;
		}

		// Calculate throughput (jobs per second)
		this.metrics.throughput = 1000 / this.metrics.processingTime;
	}

	async getSystemStatus(): Promise<{
		service: string;
		status: string;
		workers: WorkerStatus[];
		metrics: MulticoreMetrics;
		services: unknown[];
		timestamp: string;
	}> {
		// Update GPU utilization if available
		if (this.config.enableGPU) {
			try {
				const response = await fetch(`${this.loadBalancerUrl}/metrics/gpu`);
				if (response.ok) {
					const gpuData = await response.json();
					this.metrics.gpuUtilization = gpuData.utilization || 0;
				}
			} catch {
				// GPU metrics not available
			}
		}

		return {
			service: 'Context7 Multicore',
			status: this.metrics.activeWorkers > 0 ? 'operational' : 'degraded',
			workers: Array.from(this.workers.values()),
			metrics: this.metrics,
			services: this.config.goServices.map(s => ({
				name: s.name,
				port: s.port,
				type: s.type,
				enabled: s.enabled,
				gpuEnabled: s.gpuEnabled
			})),
			timestamp: new Date().toISOString()
		};
	}

	async healthCheck(): Promise<boolean> {
		return this.metrics.activeWorkers > 0 && this.metrics.errorRate < 0.5;
	}

	async shutdown(): Promise<void> {
		console.log('🛑 Shutting down Context7 Multicore Service...');
		
		// Cancel pending jobs
		this.jobQueue.forEach(job => {
			if (job.status === 'pending') {
				job.status = 'cancelled';
			}
		});

		// Reset worker statuses
		this.workers.forEach(worker => {
			worker.status = 'idle';
			worker.activeJobs = 0;
		});

		console.log('✅ Context7 Multicore Service shutdown complete');
	}
}

// Export default instance
export const context7Multicore = new Context7MulticoreService();

// Export types for external use
export type {
	Context7MulticoreConfig,
	GoServiceConfig,
	WorkerStatus,
	MulticoreMetrics
};