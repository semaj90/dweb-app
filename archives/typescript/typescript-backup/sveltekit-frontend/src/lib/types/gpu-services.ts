// GPU Services Type Definitions for Legal AI Platform
// TypeScript interfaces for Go GPU Orchestrator integration

export type GPUTaskType = 
	| 'embedding' 
	| 'similarity' 
	| 'autoindex' 
	| 'som_train' 
	| 'matrix_multiply' 
	| 'batch_process';

export interface GPUTask {
	id?: string;
	type: GPUTaskType;
	data: number[];
	metadata?: Record<string, any>;
	priority?: number;
	timestamp?: string;
	service_origin?: string;
}

export interface GPUResult {
	task_id: string;
	type: GPUTaskType;
	result: number[];
	status: 'success' | 'error' | 'processing';
	process_time: number;
	error?: string;
	timestamp: string;
}

export interface GPUStatus {
	orchestrator_status: 'running' | 'stopped' | 'error';
	workers_active: number;
	queue_length: number;
	total_workers: number;
	uptime: string;
	cuda_available: boolean;
	load_balancer: boolean;
	services_managed: number;
}

export interface GPUMetrics {
	total_tasks: number;
	completed_tasks: number;
	failed_tasks: number;
	average_process_time: number;
	queue_length: number;
	active_workers: number;
	gpu_utilization: number;
	memory_usage: number;
	start_time: string;
	last_update: string;
}

export interface WorkerStatus {
	id: number;
	busy: boolean;
	tasks_processed: number;
	last_activity: string;
	current_task?: string;
}

export interface ServiceInfo {
	name: string;
	port: number;
	type: 'AI/RAG' | 'File/Upload' | 'Protocol' | 'Infrastructure' | 'XState' | 'Monitoring';
	gpu_enabled: boolean;
	status: 'running' | 'stopped' | 'pending' | 'error';
	last_health_check: string;
	protocols: string[];
}

export interface ServiceRegistry {
	services: Record<string, ServiceInfo>;
}

export interface GPUHealth {
	status: 'healthy' | 'unhealthy' | 'degraded';
	timestamp: number;
	gpu: boolean;
	redis: 'healthy' | 'unhealthy';
	workers: number;
	queue_size: number;
}

export interface LoadBalancerStatus {
	enabled: boolean;
	status: 'active' | 'inactive' | 'error';
	services_managed: number;
}

export interface RouteRequest {
	service: string;
	method: 'GET' | 'POST' | 'PUT' | 'DELETE';
	path: string;
	data?: Record<string, any>;
}

export interface BatchGPUTask {
	tasks: GPUTask[];
	max_concurrent?: number;
	priority?: number;
}

export interface BatchGPUResult {
	total: number;
	successful: number;
	failed: number;
	results: GPUResult[];
	errors: Array<{ error: string }>;
}

// Legal AI Specific Types
export interface LegalEmbeddingTask extends GPUTask {
	type: 'embedding';
	metadata: {
		document_id: string;
		document_type: 'contract' | 'case_law' | 'regulation' | 'evidence';
		practice_area: string;
		jurisdiction: string;
		chunk_index?: number;
	};
}

export interface LegalSimilarityTask extends GPUTask {
	type: 'similarity';
	metadata: {
		query_document_id: string;
		comparison_document_id: string;
		similarity_threshold: number;
		practice_area: string;
	};
}

export interface LegalDocumentProcessingPipeline {
	document_id: string;
	tasks: (LegalEmbeddingTask | LegalSimilarityTask)[];
	priority: 'high' | 'medium' | 'low';
	practice_area: string;
	estimated_completion_time?: number;
}

// Service Integration Types
export interface ServiceProtocolConfig {
	http: {
		base_url: string;
		timeout: number;
		retry_attempts: number;
	};
	grpc?: {
		address: string;
		tls_enabled: boolean;
	};
	quic?: {
		address: string;
		certificate_path: string;
	};
	websocket?: {
		url: string;
		reconnect_attempts: number;
	};
}

export interface GPUServiceClient {
	submitTask: (task: GPUTask) => Promise<GPUResult>;
	submitBatch: (batch: BatchGPUTask) => Promise<BatchGPUResult>;
	getStatus: () => Promise<GPUStatus>;
	getMetrics: () => Promise<GPUMetrics>;
	getHealth: () => Promise<GPUHealth>;
	getWorkers: () => Promise<WorkerStatus[]>;
	getServices: () => Promise<ServiceRegistry>;
}

// Performance Monitoring Types
export interface GPUPerformanceMetrics {
	throughput: {
		tasks_per_second: number;
		embeddings_per_second: number;
		similarity_queries_per_second: number;
	};
	latency: {
		p50: number;
		p95: number;
		p99: number;
		average: number;
	};
	resource_usage: {
		gpu_utilization: number;
		memory_usage: number;
		worker_utilization: number;
		queue_utilization: number;
	};
	error_rates: {
		task_failure_rate: number;
		service_error_rate: number;
		timeout_rate: number;
	};
}

// Configuration Types
export interface GPUOrchestratorConfig {
	port: string;
	redis_addr: string;
	cuda_worker_path: string;
	max_cuda_workers: number;
	worker_pool_size: number;
	health_check_interval: number;
	load_balancer_enabled: boolean;
}

// Error Types
export interface GPUServiceError {
	code: 'GPU_UNAVAILABLE' | 'QUEUE_FULL' | 'TASK_TIMEOUT' | 'WORKER_ERROR' | 'SERVICE_DOWN';
	message: string;
	details?: Record<string, any>;
	timestamp: string;
	retry_after?: number;
}