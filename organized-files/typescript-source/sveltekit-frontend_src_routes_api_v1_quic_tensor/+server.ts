// QUIC Tensor Operations API
// Direct integration with Go tensor-tiling backend
// Supports 4D tensor operations with Redis streaming

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Tensor operation types matching Go backend
interface Tensor4DRequest {
	document_id: string;
	embeddings: number[][];
	metadata: {
		document_type: string;
		practice_area: string;
		jurisdiction: string;
		embedding_model: string;
		processing_type: 'chunk' | 'sentence' | 'paragraph';
		legal_entities: string[];
		context: Record<string, any>;
	};
}

interface TricubicRequest {
	tensor_id: string;
	coordinates: [number, number, number];
	parameters: {
		points: number[][][];
		coordinates: [number, number, number];
		smoothness: number;
	};
}

interface TensorOperationRequest {
	operation: 'tricubic' | 'som_update' | 'tile_aggregate';
	parameters: Record<string, any>;
}

// Configuration for Go backend integration
const GO_TENSOR_SERVICE = process.env.TENSOR_GPU_SERVICE_URL || 'http://localhost:8085';

// Forward request to Go tensor-tiling service
async function callGoTensorService(endpoint: string, data: any): Promise<any> {
	try {
		const response = await fetch(`${GO_TENSOR_SERVICE}/api/tensor${endpoint}`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'User-Agent': 'SvelteKit-QUIC-Client'
			},
			body: JSON.stringify(data)
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`Go service error (${response.status}): ${errorText}`);
		}

		return await response.json();
	} catch (err) {
		console.error('🚫 Go tensor service call failed:', err);
		throw err;
	}
}

// Create 4D tensor from legal document embeddings
export const POST: RequestHandler = async ({ request, url }) => {
	try {
		const operation = url.searchParams.get('op') || 'create';
		const startTime = Date.now();

		switch (operation) {
			case 'create': {
				const tensorRequest: Tensor4DRequest = await request.json();
				
				// Validate request
				if (!tensorRequest.document_id || !tensorRequest.embeddings) {
					return error(400, 'document_id and embeddings are required');
				}

				if (tensorRequest.embeddings.length === 0) {
					return error(400, 'embeddings cannot be empty');
				}

				// Forward to Go backend
				const result = await callGoTensorService('/create', tensorRequest);
				
				const processingTime = Date.now() - startTime;

				return json({
					...result,
					quic_processing_time_ms: processingTime,
					service: 'go-tensor-tiling',
					api_version: 'v1'
				}, {
					headers: {
						'X-QUIC-Processing-Time': processingTime.toString(),
						'X-Tensor-Service': 'go-backend',
						'X-Operation': 'create'
					}
				});
			}

			case 'interpolate': {
				const interpolationRequest: TricubicRequest = await request.json();
				
				// Validate request
				if (!interpolationRequest.tensor_id || !interpolationRequest.coordinates) {
					return error(400, 'tensor_id and coordinates are required');
				}

				// Forward to Go backend
				const result = await callGoTensorService('/interpolate', interpolationRequest);
				
				const processingTime = Date.now() - startTime;

				return json({
					...result,
					quic_processing_time_ms: processingTime,
					interpolation_type: 'tricubic',
					coordinates: interpolationRequest.coordinates
				}, {
					headers: {
						'X-QUIC-Processing-Time': processingTime.toString(),
						'X-Tensor-Service': 'go-backend',
						'X-Operation': 'interpolate'
					}
				});
			}

			default:
				return error(400, `Unsupported operation: ${operation}`);
		}

	} catch (err) {
		console.error('🚫 QUIC Tensor API error:', err);
		
		if (err instanceof Error && err.message.includes('Go service error')) {
			return error(502, `Backend service error: ${err.message}`);
		}

		return error(500, `Tensor processing error: ${err instanceof Error ? err.message : 'Unknown error'}`);
	}
};

// Get tensor information and tiles
export const GET: RequestHandler = async ({ url }) => {
	try {
		const tensorId = url.searchParams.get('tensor_id');
		const operation = url.searchParams.get('op') || 'info';

		if (!tensorId) {
			return error(400, 'tensor_id parameter is required');
		}

		const startTime = Date.now();

		switch (operation) {
			case 'info': {
				// Get tensor information from Go backend
				const response = await fetch(`${GO_TENSOR_SERVICE}/api/tensor/${tensorId}`);
				
				if (!response.ok) {
					if (response.status === 404) {
						return error(404, 'Tensor not found');
					}
					throw new Error(`Go service error: ${response.status}`);
				}

				const tensorInfo = await response.json();
				const processingTime = Date.now() - startTime;

				return json({
					...tensorInfo,
					quic_processing_time_ms: processingTime,
					retrieved_at: new Date().toISOString()
				}, {
					headers: {
						'X-QUIC-Processing-Time': processingTime.toString(),
						'X-Tensor-Service': 'go-backend',
						'X-Operation': 'info'
					}
				});
			}

			case 'tiles': {
				// Get tensor tiles from Go backend
				const response = await fetch(`${GO_TENSOR_SERVICE}/api/tensor/${tensorId}/tiles`);
				
				if (!response.ok) {
					if (response.status === 404) {
						return error(404, 'Tensor not found');
					}
					throw new Error(`Go service error: ${response.status}`);
				}

				const tilesInfo = await response.json();
				const processingTime = Date.now() - startTime;

				return json({
					...tilesInfo,
					quic_processing_time_ms: processingTime,
					retrieved_at: new Date().toISOString()
				}, {
					headers: {
						'X-QUIC-Processing-Time': processingTime.toString(),
						'X-Tensor-Service': 'go-backend',
						'X-Operation': 'tiles'
					}
				});
			}

			case 'status': {
				// Get processing status for tensor operations
				return json({
					tensor_id: tensorId,
					status: 'active', // Placeholder
					go_backend_url: GO_TENSOR_SERVICE,
					quic_enabled: true,
					processing_time_ms: Date.now() - startTime
				});
			}

			default:
				return error(400, `Unsupported GET operation: ${operation}`);
		}

	} catch (err) {
		console.error('🚫 QUIC Tensor GET API error:', err);
		return error(500, `Error retrieving tensor data: ${err instanceof Error ? err.message : 'Unknown error'}`);
	}
};

// Tensor operations (SOM updates, aggregations, etc.)
export const PUT: RequestHandler = async ({ request, url }) => {
	try {
		const tensorId = url.searchParams.get('tensor_id');
		
		if (!tensorId) {
			return error(400, 'tensor_id parameter is required');
		}

		const operationRequest: TensorOperationRequest = await request.json();
		
		if (!operationRequest.operation) {
			return error(400, 'operation field is required');
		}

		const startTime = Date.now();

		// Forward to Go backend tensor operation endpoint
		const result = await callGoTensorService(`/${tensorId}/operation`, {
			operation: operationRequest.operation,
			parameters: operationRequest.parameters || {}
		});

		const processingTime = Date.now() - startTime;

		return json({
			...result,
			tensor_id: tensorId,
			operation: operationRequest.operation,
			quic_processing_time_ms: processingTime,
			submitted_at: new Date().toISOString()
		}, {
			headers: {
				'X-QUIC-Processing-Time': processingTime.toString(),
				'X-Tensor-Service': 'go-backend',
				'X-Operation': operationRequest.operation
			}
		});

	} catch (err) {
		console.error('🚫 QUIC Tensor PUT API error:', err);
		return error(500, `Error processing tensor operation: ${err instanceof Error ? err.message : 'Unknown error'}`);
	}
};

// Health check endpoint
export const OPTIONS: RequestHandler = async () => {
	try {
		// Check Go backend health
		const healthResponse = await fetch(`${GO_TENSOR_SERVICE}/health`, {
			method: 'GET',
			timeout: 5000
		} as any);

		const isHealthy = healthResponse.ok;
		
		return json({
			quic_tensor_api: 'healthy',
			go_backend: isHealthy ? 'healthy' : 'unhealthy',
			go_backend_url: GO_TENSOR_SERVICE,
			timestamp: new Date().toISOString()
		}, {
			status: isHealthy ? 200 : 503,
			headers: {
				'X-Service-Health': isHealthy ? 'healthy' : 'degraded',
				'X-Backend-Status': isHealthy ? 'connected' : 'disconnected'
			}
		});

	} catch (err) {
		console.error('🚫 QUIC Tensor health check error:', err);
		return json({
			quic_tensor_api: 'healthy',
			go_backend: 'unreachable',
			go_backend_url: GO_TENSOR_SERVICE,
			error: err instanceof Error ? err.message : 'Unknown error',
			timestamp: new Date().toISOString()
		}, {
			status: 503,
			headers: {
				'X-Service-Health': 'degraded',
				'X-Backend-Status': 'unreachable'
			}
		});
	}
};