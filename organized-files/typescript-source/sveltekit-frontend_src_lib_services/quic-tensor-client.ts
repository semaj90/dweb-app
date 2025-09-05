// QUIC Tensor Client - TypeScript Integration Layer
// Connects SvelteKit frontend to Go tensor-tiling backend
// Provides type-safe interface for tensor operations

export interface TensorMetadata {
	document_type: string;
	practice_area: string;
	jurisdiction: string;
	embedding_model: string;
	processing_type: 'chunk' | 'sentence' | 'paragraph';
	legal_entities: string[];
	context: Record<string, any>;
}

export interface Tensor4DInfo {
	tensor_id: string;
	shape: [number, number, number, number];
	tiles: number;
	created: string;
	document_id: string;
	metadata: TensorMetadata;
}

export interface TensorTileInfo {
	tensor_id: string;
	tiles: string[];
	count: number;
}

export interface TricubicResult {
	result: number[];
	coordinates: [number, number, number];
	dimension: number;
	interpolation_type: string;
	quic_processing_time_ms: number;
}

export interface QuicStreamStatus {
	active_streams: number;
	max_concurrent: number;
	total_streams: number;
	utilization_percent: number;
	active_operations: Record<string, number>;
}

export class QuicTensorClient {
	private baseUrl: string;
	private apiVersion: string = 'v1';

	constructor(baseUrl: string = '') {
		this.baseUrl = baseUrl;
	}

	/**
	 * Create 4D tensor from legal document embeddings
	 */
	async createTensor4D(
		documentId: string,
		embeddings: number[][],
		metadata: TensorMetadata
	): Promise<Tensor4DInfo> {
		const response = await fetch(`${this.baseUrl}/api/v1/quic/tensor?op=create`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				document_id: documentId,
				embeddings,
				metadata
			})
		});

		if (!response.ok) {
			const error = await response.text();
			throw new Error(`Failed to create tensor: ${error}`);
		}

		return await response.json();
	}

	/**
	 * Perform tricubic interpolation on tensor
	 */
	async tricubicInterpolation(
		tensorId: string,
		coordinates: [number, number, number],
		parameters: {
			points: number[][][];
			coordinates: [number, number, number];
			smoothness: number;
		}
	): Promise<TricubicResult> {
		const response = await fetch(`${this.baseUrl}/api/v1/quic/tensor?op=interpolate`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				tensor_id: tensorId,
				coordinates,
				parameters
			})
		});

		if (!response.ok) {
			const error = await response.text();
			throw new Error(`Failed to perform tricubic interpolation: ${error}`);
		}

		return await response.json();
	}

	/**
	 * Get tensor information
	 */
	async getTensorInfo(tensorId: string): Promise<Tensor4DInfo> {
		const response = await fetch(
			`${this.baseUrl}/api/v1/quic/tensor?tensor_id=${tensorId}&op=info`,
			{
				method: 'GET',
			}
		);

		if (!response.ok) {
			if (response.status === 404) {
				throw new Error(`Tensor ${tensorId} not found`);
			}
			const error = await response.text();
			throw new Error(`Failed to get tensor info: ${error}`);
		}

		return await response.json();
	}

	/**
	 * Get tensor tiles information
	 */
	async getTensorTiles(tensorId: string): Promise<TensorTileInfo> {
		const response = await fetch(
			`${this.baseUrl}/api/v1/quic/tensor?tensor_id=${tensorId}&op=tiles`,
			{
				method: 'GET',
			}
		);

		if (!response.ok) {
			if (response.status === 404) {
				throw new Error(`Tensor ${tensorId} not found`);
			}
			const error = await response.text();
			throw new Error(`Failed to get tensor tiles: ${error}`);
		}

		return await response.json();
	}

	/**
	 * Execute tensor operation (SOM update, aggregation, etc.)
	 */
	async executeTensorOperation(
		tensorId: string,
		operation: 'tricubic' | 'som_update' | 'tile_aggregate',
		parameters: Record<string, any> = {}
	): Promise<any> {
		const response = await fetch(
			`${this.baseUrl}/api/v1/quic/tensor?tensor_id=${tensorId}`,
			{
				method: 'PUT',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					operation,
					parameters
				})
			}
		);

		if (!response.ok) {
			const error = await response.text();
			throw new Error(`Failed to execute tensor operation: ${error}`);
		}

		return await response.json();
	}

	/**
	 * Get QUIC stream status
	 */
	async getStreamStatus(): Promise<QuicStreamStatus> {
		const response = await fetch(`${this.baseUrl}/api/v1/quic?action=status`, {
			method: 'GET',
		});

		if (!response.ok) {
			const error = await response.text();
			throw new Error(`Failed to get stream status: ${error}`);
		}

		return await response.json();
	}

	/**
	 * Check service health
	 */
	async checkHealth(): Promise<{
		quic_tensor_api: string;
		go_backend: string;
		go_backend_url: string;
		timestamp: string;
	}> {
		const response = await fetch(`${this.baseUrl}/api/v1/quic/tensor`, {
			method: 'OPTIONS',
		});

		const result = await response.json();
		return result;
	}

	/**
	 * Process legal document for tensor creation
	 * Convenience method for common legal AI workflow
	 */
	async processLegalDocument(
		documentId: string,
		documentText: string,
		documentType: string,
		practiceArea: string,
		jurisdiction: string = 'US'
	): Promise<Tensor4DInfo> {
		// Simulate embeddings generation (in production, call your embedding service)
		const embeddings = this.generateMockEmbeddings(documentText);

		const metadata: TensorMetadata = {
			document_type: documentType,
			practice_area: practiceArea,
			jurisdiction,
			embedding_model: 'nomic-embed-text',
			processing_type: 'chunk',
			legal_entities: this.extractLegalEntities(documentText),
			context: {
				document_length: documentText.length,
				processed_at: new Date().toISOString(),
				language: 'en'
			}
		};

		return await this.createTensor4D(documentId, embeddings, metadata);
	}

	/**
	 * Generate mock embeddings for testing
	 * In production, replace with actual embedding service call
	 */
	private generateMockEmbeddings(text: string): number[][] {
		const chunks = this.chunkText(text, 512); // 512 char chunks
		return chunks.map(() => {
			// Generate 384-dimensional embedding (matching nomic-embed-text)
			return Array.from({ length: 384 }, () => Math.random() * 2 - 1);
		});
	}

	/**
	 * Simple text chunking
	 */
	private chunkText(text: string, chunkSize: number): string[] {
		const chunks = [];
		for (let i = 0; i < text.length; i += chunkSize) {
			chunks.push(text.substring(i, i + chunkSize));
		}
		return chunks;
	}

	/**
	 * Extract legal entities from text (simplified)
	 */
	private extractLegalEntities(text: string): string[] {
		const entities = [];
		const lowerText = text.toLowerCase();

		// Common legal entity patterns
		const patterns = [
			/\b(?:plaintiff|defendant|respondent|petitioner)\b/g,
			/\b(?:contract|agreement|lease|deed|will|trust)\b/g,
			/\b(?:court|judge|attorney|counsel|law firm)\b/g,
			/\b(?:case|lawsuit|litigation|proceeding)\b/g
		];

		for (const pattern of patterns) {
			const matches = lowerText.match(pattern);
			if (matches) {
				entities.push(...matches);
			}
		}

		// Remove duplicates and return
		return [...new Set(entities)];
	}

	/**
	 * Batch process multiple documents
	 */
	async batchProcessDocuments(
		documents: Array<{
			id: string;
			text: string;
			type: string;
			practiceArea: string;
			jurisdiction?: string;
		}>
	): Promise<Tensor4DInfo[]> {
		const results = [];

		for (const doc of documents) {
			try {
				const result = await this.processLegalDocument(
					doc.id,
					doc.text,
					doc.type,
					doc.practiceArea,
					doc.jurisdiction
				);
				results.push(result);
			} catch (error) {
				console.error(`Failed to process document ${doc.id}:`, error);
				// Continue with other documents
			}
		}

		return results;
	}
}

// Default client instance
export const quicTensorClient = new QuicTensorClient();

// Utility functions for tensor operations
export const TensorUtils = {
	/**
	 * Calculate tensor memory usage estimate
	 */
	estimateMemoryUsage(shape: [number, number, number, number]): number {
		const [batch, depth, height, width] = shape;
		const elements = batch * depth * height * width;
		return elements * 4; // 4 bytes per float32
	},

	/**
	 * Calculate optimal tile size based on available memory
	 */
	calculateOptimalTileSize(
		tensorShape: [number, number, number, number],
		availableMemoryMB: number
	): [number, number, number, number] {
		const maxElements = (availableMemoryMB * 1024 * 1024) / 4; // Convert MB to float32 elements
		const [batch, depth, height, width] = tensorShape;
		
		// Simple heuristic - divide largest dimensions
		const totalElements = batch * depth * height * width;
		
		if (totalElements <= maxElements) {
			return tensorShape; // No tiling needed
		}

		// Calculate scale factor
		const scaleFactor = Math.pow(maxElements / totalElements, 0.25); // 4th root for 4D
		
		return [
			Math.max(1, Math.floor(batch * scaleFactor)),
			Math.max(1, Math.floor(depth * scaleFactor)),
			Math.max(1, Math.floor(height * scaleFactor)),
			Math.max(1, Math.floor(width * scaleFactor))
		] as [number, number, number, number];
	},

	/**
	 * Format tensor shape for display
	 */
	formatTensorShape(shape: [number, number, number, number]): string {
		const [batch, depth, height, width] = shape;
		return `${batch}×${depth}×${height}×${width}`;
	},

	/**
	 * Calculate compression ratio estimate
	 */
	estimateCompressionRatio(tensorData: number[]): number {
		// Simple compression estimate based on data variance
		const mean = tensorData.reduce((sum, val) => sum + val, 0) / tensorData.length;
		const variance = tensorData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / tensorData.length;
		
		// Higher variance = lower compression ratio
		return Math.max(1.5, 10 - variance * 5);
	}
};