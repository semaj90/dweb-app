import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { vectorSearch } from '$lib/server/search/vector-search';
import { GPUService } from '$lib/gpu/nes-gpu-integration';

let gpuService: GPUService | null = null;

async function getGPUService() {
	if (!gpuService) {
		gpuService = new GPUService();
		try {
			await gpuService.initialize();
		} catch (err) {
			console.warn('GPU service initialization failed, falling back to vector search:', err);
			gpuService = null;
		}
	}
	return gpuService;
}

export const GET: RequestHandler = async ({ url }) => {
	try {
		const query = url.searchParams.get('q');
		const limit = parseInt(url.searchParams.get('limit') || '20');
		const threshold = parseFloat(url.searchParams.get('threshold') || '0.7');
		const type = url.searchParams.get('type');
		const caseId = url.searchParams.get('caseId');
		const useGPU = url.searchParams.get('useGPU') === 'true';

		if (!query) {
			throw error(400, { message: 'Query parameter is required' });
		}

		const filters: Record<string, any> = {};
		if (type) filters.type = type;
		if (caseId) filters.caseId = caseId;

		let searchResults;
		let processingMetadata = {};

		// Try GPU-accelerated search first if requested
		if (useGPU) {
			const gpu = await getGPUService();
			if (gpu) {
				try {
					const gpuResults = await gpu.search({
						query,
						limit: Math.min(limit, 50),
						threshold: Math.max(0.1, Math.min(threshold, 1.0)),
						useGPU: true,
						includeMetadata: true,
						filters
					});

					searchResults = gpuResults.documents;
					processingMetadata = {
						processingTime: gpuResults.processingTime,
						gpuUsed: gpuResults.gpuUsed,
						cacheHit: gpuResults.cacheHit,
						totalResults: gpuResults.total,
						method: 'neo4j-gpu'
					};
				} catch (gpuError) {
					console.warn('GPU search failed, falling back to vector search:', gpuError);
				}
			}
		}

		// Fallback to traditional vector search
		if (!searchResults) {
			const startTime = Date.now();
			searchResults = await vectorSearch(query, {
				limit,
				threshold,
				filters,
				useCache: true,
				fallbackToQdrant: true
			});
			
			processingMetadata = {
				processingTime: Date.now() - startTime,
				gpuUsed: false,
				cacheHit: false,
				totalResults: searchResults.length,
				method: 'vector-search'
			};
		}

		return json({
			success: true,
			query,
			data: searchResults,
			metadata: {
				...processingMetadata,
				timestamp: new Date().toISOString(),
				filters: Object.keys(filters).length > 0 ? filters : undefined
			}
		});
	} catch (err) {
		console.error('Search API error:', err);
		
		if (err.status) {
			throw err; // Re-throw SvelteKit errors
		}

		return json(
			{
				success: false,
				error: err instanceof Error ? err.message : 'Search failed',
				timestamp: new Date().toISOString()
			},
			{ status: 500 }
		);
	}
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const { query, limit = 20, threshold = 0.7, filters = {}, useGPU = false } = await request.json();

		if (!query || typeof query !== 'string') {
			throw error(400, { message: 'Query is required and must be a string' });
		}

		// Forward to GET handler with same logic
		const url = new URL('?dummy=1', 'http://localhost');
		url.searchParams.set('q', query);
		url.searchParams.set('limit', limit.toString());
		url.searchParams.set('threshold', threshold.toString());
		url.searchParams.set('useGPU', useGPU.toString());
		
		// Add filters as individual parameters
		Object.entries(filters).forEach(([key, value]) => {
			if (value != null) {
				url.searchParams.set(key, value.toString());
			}
		});

		return await GET({ url } as any);

	} catch (err) {
		console.error('Search POST API error:', err);
		
		if (err.status) {
			throw err;
		}

		return json({
			success: false,
			error: err instanceof Error ? err.message : 'Unknown error occurred',
			timestamp: new Date().toISOString()
		}, { status: 500 });
	}
};
