// GPU Service Client for Legal AI Platform
// Frontend service client for GPU orchestration and task management

import { dev } from '$app/environment';
import type {
	GPUServiceClient,
	GPUTask,
	GPUResult,
	GPUStatus,
	GPUMetrics,
	GPUHealth,
	WorkerStatus,
	ServiceRegistry,
	BatchGPUTask,
	BatchGPUResult,
	LegalEmbeddingTask,
	LegalSimilarityTask,
	GPUServiceError
} from '$lib/types/gpu-services';

class GPUServiceClientImpl implements GPUServiceClient {
	private baseUrl: string;
	private timeout: number;
	private retryAttempts: number;

	constructor() {
		this.baseUrl = '/api/gpu';
		this.timeout = 30000; // 30 seconds
		this.retryAttempts = 3;
	}

	// Core GPU Task Submission
	async submitTask(task: GPUTask): Promise<GPUResult> {
		try {
			const response = await this.makeRequest('POST', '?action=process', task);
			return await response.json();
		} catch (error) {
			throw this.handleError('Failed to submit GPU task', error);
		}
	}

	// Batch Task Processing
	async submitBatch(batch: BatchGPUTask): Promise<BatchGPUResult> {
		try {
			const response = await this.makeRequest('POST', '?action=batch', batch);
			const result = await response.json();
			return result.batch_results;
		} catch (error) {
			throw this.handleError('Failed to submit batch tasks', error);
		}
	}

	// GPU Status Information
	async getStatus(): Promise<GPUStatus> {
		try {
			const response = await this.makeRequest('GET', '?action=status');
			return await response.json();
		} catch (error) {
			throw this.handleError('Failed to get GPU status', error);
		}
	}

	// GPU Performance Metrics
	async getMetrics(): Promise<GPUMetrics> {
		try {
			const response = await this.makeRequest('GET', '?action=metrics');
			return await response.json();
		} catch (error) {
			throw this.handleError('Failed to get GPU metrics', error);
		}
	}

	// GPU Health Check
	async getHealth(): Promise<GPUHealth> {
		try {
			const response = await this.makeRequest('GET', '?action=health');
			return await response.json();
		} catch (error) {
			throw this.handleError('Failed to get GPU health', error);
		}
	}

	// Worker Status
	async getWorkers(): Promise<WorkerStatus[]> {
		try {
			const response = await this.makeRequest('GET', '?action=workers');
			const result = await response.json();
			return result.workers || [];
		} catch (error) {
			throw this.handleError('Failed to get worker status', error);
		}
	}

	// Service Registry
	async getServices(): Promise<ServiceRegistry> {
		try {
			const response = await this.makeRequest('GET', '?action=services');
			return await response.json();
		} catch (error) {
			throw this.handleError('Failed to get service registry', error);
		}
	}

	// Legal AI Specific Methods

	// Process Legal Document Embedding
	async processLegalEmbedding(
		text: string, 
		documentId: string, 
		documentType: 'contract' | 'case_law' | 'regulation' | 'evidence',
		practiceArea: string,
		jurisdiction: string
	): Promise<GPUResult> {
		const task: LegalEmbeddingTask = {
			type: 'embedding',
			data: this.textToFloatArray(text),
			metadata: {
				document_id: documentId,
				document_type: documentType,
				practice_area: practiceArea,
				jurisdiction: jurisdiction
			},
			priority: 7, // High priority for legal processing
			service_origin: 'legal-ai-frontend'
		};

		return this.submitTask(task);
	}

	// Process Legal Document Similarity
	async processLegalSimilarity(
		queryEmbedding: number[],
		comparisonEmbedding: number[],
		queryDocId: string,
		comparisonDocId: string,
		practiceArea: string,
		threshold: number = 0.8
	): Promise<GPUResult> {
		const combinedData = [...queryEmbedding, ...comparisonEmbedding];
		
		const task: LegalSimilarityTask = {
			type: 'similarity',
			data: combinedData,
			metadata: {
				query_document_id: queryDocId,
				comparison_document_id: comparisonDocId,
				similarity_threshold: threshold,
				practice_area: practiceArea
			},
			priority: 8, // Very high priority for similarity queries
			service_origin: 'legal-similarity-search'
		};

		return this.submitTask(task);
	}

	// Batch Process Legal Documents
	async processLegalDocumentBatch(
		documents: Array<{
			text: string;
			documentId: string;
			documentType: 'contract' | 'case_law' | 'regulation' | 'evidence';
			practiceArea: string;
			jurisdiction: string;
		}>
	): Promise<BatchGPUResult> {
		const tasks: LegalEmbeddingTask[] = documents.map(doc => ({
			type: 'embedding',
			data: this.textToFloatArray(doc.text),
			metadata: {
				document_id: doc.documentId,
				document_type: doc.documentType,
				practice_area: doc.practiceArea,
				jurisdiction: doc.jurisdiction
			},
			priority: 6,
			service_origin: 'legal-batch-processor'
		}));

		return this.submitBatch({ tasks, max_concurrent: 5 });
	}

	// Vector Search for Legal Cases
	async searchLegalCases(
		queryEmbedding: number[],
		caseEmbeddings: Array<{ id: string; embedding: number[]; metadata: any }>,
		limit: number = 10
	): Promise<Array<{ caseId: string; similarity: number; metadata: any }>> {
		const results: Array<{ caseId: string; similarity: number; metadata: any }> = [];
		
		// Process similarities in batches to avoid overwhelming the GPU
		const batchSize = 20;
		for (let i = 0; i < caseEmbeddings.length; i += batchSize) {
			const batch = caseEmbeddings.slice(i, i + batchSize);
			
			const batchTasks: LegalSimilarityTask[] = batch.map(caseEmb => ({
				type: 'similarity',
				data: [...queryEmbedding, ...caseEmb.embedding],
				metadata: {
					query_document_id: 'query',
					comparison_document_id: caseEmb.id,
					similarity_threshold: 0.0,
					practice_area: caseEmb.metadata.practice_area || 'general'
				},
				service_origin: 'legal-case-search'
			}));

			const batchResult = await this.submitBatch({ tasks: batchTasks });
			
			if (batchResult.results) {
				for (let j = 0; j < batchResult.results.length; j++) {
					const result = batchResult.results[j];
					if (result.status === 'success' && result.result.length > 0) {
						const similarity = result.result[0]; // First value should be similarity score
						const caseData = batch[j];
						results.push({
							caseId: caseData.id,
							similarity: similarity,
							metadata: caseData.metadata
						});
					}
				}
			}
		}

		// Sort by similarity and return top results
		return results
			.sort((a, b) => b.similarity - a.similarity)
			.slice(0, limit);
	}

	// Utility Methods

	// Convert text to float array (simplified tokenization)
	private textToFloatArray(text: string): number[] {
		// This is a simplified version - in production you'd use proper tokenization
		const normalized = text.toLowerCase().replace(/[^\w\s]/g, '');
		const words = normalized.split(/\s+/).filter(w => w.length > 0);
		
		// Create a simple hash-based embedding (384 dimensions)
		const embedding = new Array(384).fill(0);
		
		for (let i = 0; i < words.length; i++) {
			const word = words[i];
			let hash = 0;
			for (let j = 0; j < word.length; j++) {
				hash = ((hash << 5) - hash) + word.charCodeAt(j);
				hash = hash & hash; // Convert to 32-bit integer
			}
			
			const index = Math.abs(hash) % embedding.length;
			embedding[index] += 1 / Math.sqrt(words.length);
		}
		
		// Normalize the embedding
		const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
		if (magnitude > 0) {
			for (let i = 0; i < embedding.length; i++) {
				embedding[i] /= magnitude;
			}
		}
		
		return embedding;
	}

	// HTTP Request Helper
	private async makeRequest(
		method: 'GET' | 'POST' | 'PUT' | 'DELETE',
		endpoint: string,
		data?: any
	): Promise<Response> {
		const url = `${this.baseUrl}${endpoint}`;
		const options: RequestInit = {
			method,
			headers: {
				'Content-Type': 'application/json'
			}
		};

		if (data && (method === 'POST' || method === 'PUT')) {
			options.body = JSON.stringify(data);
		}

		// Implement retry logic
		for (let attempt = 0; attempt < this.retryAttempts; attempt++) {
			try {
				const controller = new AbortController();
				const timeoutId = setTimeout(() => controller.abort(), this.timeout);
				
				options.signal = controller.signal;
				
				const response = await fetch(url, options);
				clearTimeout(timeoutId);
				
				if (!response.ok) {
					throw new Error(`HTTP ${response.status}: ${response.statusText}`);
				}
				
				return response;
			} catch (error) {
				if (attempt === this.retryAttempts - 1) {
					throw error;
				}
				
				// Exponential backoff
				const delay = Math.pow(2, attempt) * 1000;
				await new Promise(resolve => setTimeout(resolve, delay));
			}
		}
		
		throw new Error('Max retry attempts exceeded');
	}

	// Error Handling
	private handleError(message: string, error: any): GPUServiceError {
		if (dev) {
			console.error(message, error);
		}
		
		let errorCode: GPUServiceError['code'] = 'SERVICE_DOWN';
		
		if (error instanceof TypeError && error.message.includes('fetch')) {
			errorCode = 'SERVICE_DOWN';
		} else if (error.message?.includes('timeout')) {
			errorCode = 'TASK_TIMEOUT';
		} else if (error.message?.includes('queue full')) {
			errorCode = 'QUEUE_FULL';
		} else if (error.message?.includes('GPU')) {
			errorCode = 'GPU_UNAVAILABLE';
		}
		
		return {
			code: errorCode,
			message,
			details: { originalError: error.message || error },
			timestamp: new Date().toISOString()
		};
	}
}

// Singleton instance
export const gpuServiceClient = new GPUServiceClientImpl();

// Helper functions for common operations
export async function isGPUAvailable(): Promise<boolean> {
	try {
		const health = await gpuServiceClient.getHealth();
		return health.status === 'healthy' && health.gpu;
	} catch {
		return false;
	}
}

export async function getGPUUtilization(): Promise<number> {
	try {
		const metrics = await gpuServiceClient.getMetrics();
		return metrics.gpu_utilization || 0;
	} catch {
		return 0;
	}
}

export async function processLegalDocument(
	text: string,
	metadata: {
		documentId: string;
		documentType: 'contract' | 'case_law' | 'regulation' | 'evidence';
		practiceArea: string;
		jurisdiction: string;
	}
): Promise<number[]> {
	const result = await gpuServiceClient.processLegalEmbedding(
		text,
		metadata.documentId,
		metadata.documentType,
		metadata.practiceArea,
		metadata.jurisdiction
	);
	
	if (result.status !== 'success') {
		throw new Error(`Document processing failed: ${result.error}`);
	}
	
	return result.result;
}