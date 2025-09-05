import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { GPUService } from '$lib/gpu/nes-gpu-integration';

let gpuService: GPUService | null = null;

async function getGPUService() {
	if (!gpuService) {
		gpuService = new GPUService();
		try {
			await gpuService.initialize();
		} catch (err) {
			console.error('GPU service initialization failed:', err);
			throw error(503, { message: 'GPU service unavailable' });
		}
	}
	return gpuService;
}

export const POST: RequestHandler = async ({ request }) => {
	try {
		const { 
			queries, 
			batchSize = 32, 
			threshold = 0.7, 
			useGPU = true,
			priority = 'normal',
			includeMetadata = true 
		} = await request.json();

		if (!queries || !Array.isArray(queries) || queries.length === 0) {
			throw error(400, { message: 'Queries array is required and must not be empty' });
		}

		if (queries.length > 1000) {
			throw error(400, { message: 'Maximum 1000 queries per batch' });
		}

		const gpu = await getGPUService();
		const startTime = Date.now();

		// Process queries in batches for optimal GPU utilization
		const results = [];
		const batchCount = Math.ceil(queries.length / batchSize);
		
		for (let i = 0; i < batchCount; i++) {
			const batchStart = i * batchSize;
			const batchEnd = Math.min(batchStart + batchSize, queries.length);
			const batchQueries = queries.slice(batchStart, batchEnd);

			try {
				const batchResults = await gpu.batchProcess({
					queries: batchQueries,
					threshold,
					useGPU,
					priority,
					includeMetadata
				});

				results.push(...batchResults);

			} catch (batchError) {
				console.warn(`Batch ${i + 1} failed:`, batchError);
				
				// Add failed batch with error info
				batchQueries.forEach(query => {
					results.push({
						query,
						documents: [],
						error: 'Batch processing failed',
						success: false
					});
				});
			}
		}

		const totalProcessingTime = Date.now() - startTime;

		return json({
			success: true,
			batchId: `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
			totalQueries: queries.length,
			processedQueries: results.length,
			results,
			metadata: {
				batchSize,
				batchCount,
				totalProcessingTime,
				avgTimePerQuery: totalProcessingTime / queries.length,
				gpuUsed: useGPU,
				timestamp: new Date().toISOString(),
				successRate: results.filter(r => r.success !== false).length / results.length
			}
		});

	} catch (err) {
		console.error('Batch processing API error:', err);
		
		if (err.status) {
			throw err;
		}

		return json({
			success: false,
			error: err instanceof Error ? err.message : 'Batch processing failed',
			timestamp: new Date().toISOString()
		}, { status: 500 });
	}
};

export const GET: RequestHandler = async ({ url }) => {
	try {
		const batchId = url.searchParams.get('batchId');
		
		if (!batchId) {
			// Return batch processing capabilities
			return json({
				success: true,
				capabilities: {
					maxQueriesPerBatch: 1000,
					recommendedBatchSize: 32,
					supportedFormats: ['text', 'json'],
					gpuAcceleration: true,
					parallelProcessing: true
				},
				endpoints: {
					batch: '/api/gpu/batch',
					status: '/api/gpu/batch?batchId={id}',
					health: '/api/gpu/health'
				}
			});
		}

		// In a real implementation, you'd check batch status from a job queue
		return json({
			success: true,
			batchId,
			status: 'completed', // This would be dynamic based on actual batch status
			message: 'Batch status checking not implemented - batches process synchronously'
		});

	} catch (err) {
		console.error('Batch status API error:', err);
		
		return json({
			success: false,
			error: err instanceof Error ? err.message : 'Status check failed'
		}, { status: 500 });
	}
};