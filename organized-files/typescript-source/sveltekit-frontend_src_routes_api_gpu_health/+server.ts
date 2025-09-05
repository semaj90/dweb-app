import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { GPUService } from '$lib/gpu/nes-gpu-integration';

let gpuService: GPUService | null = null;

async function getGPUService() {
	if (!gpuService) {
		gpuService = new GPUService();
	}
	return gpuService;
}

export const GET: RequestHandler = async ({ url }) => {
	const detailed = url.searchParams.get('detailed') === 'true';
	const includeMetrics = url.searchParams.get('metrics') === 'true';
	
	try {
		const gpu = await getGPUService();
		const startTime = Date.now();
		
		// Basic health check
		const healthStatus = {
			service: 'gpu-neo4j-worker',
			status: 'unknown',
			timestamp: new Date().toISOString(),
			uptime: process.uptime(),
			version: '1.0.0'
		};

		// Test GPU service initialization
		let gpuAvailable = false;
		let neo4jConnected = false;
		let redisConnected = false;
		let initError = null;

		try {
			await gpu.initialize();
			gpuAvailable = true;
		} catch (err) {
			initError = err instanceof Error ? err.message : 'GPU initialization failed';
		}

		// Test Neo4j connection
		try {
			const neo4jHealth = await gpu.checkNeo4jHealth();
			neo4jConnected = neo4jHealth.connected;
		} catch (err) {
			console.warn('Neo4j health check failed:', err);
		}

		// Test Redis connection  
		try {
			const redisHealth = await gpu.checkRedisHealth();
			redisConnected = redisHealth.connected;
		} catch (err) {
			console.warn('Redis health check failed:', err);
		}

		// Determine overall status
		if (gpuAvailable && neo4jConnected && redisConnected) {
			healthStatus.status = 'healthy';
		} else if (gpuAvailable || neo4jConnected) {
			healthStatus.status = 'degraded';
		} else {
			healthStatus.status = 'unhealthy';
		}

		const responseTime = Date.now() - startTime;

		const basicResponse = {
			...healthStatus,
			checks: {
				gpu: { status: gpuAvailable ? 'pass' : 'fail', error: initError },
				neo4j: { status: neo4jConnected ? 'pass' : 'fail' },
				redis: { status: redisConnected ? 'pass' : 'fail' }
			},
			responseTime
		};

		if (!detailed && !includeMetrics) {
			return json(basicResponse);
		}

		// Detailed health information
		const detailedInfo: any = { ...basicResponse };

		if (detailed) {
			try {
				// Get system information
				detailedInfo.system = {
					platform: process.platform,
					arch: process.arch,
					nodeVersion: process.version,
					memory: {
						used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
						total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
						external: Math.round(process.memoryUsage().external / 1024 / 1024)
					}
				};

				// Get GPU information if available
				if (gpuAvailable) {
					detailedInfo.gpu = {
						model: 'RTX 3060 Ti', // This would be detected dynamically
						memory: '8GB',
						compute: '8.6',
						tensorCores: true,
						cudaVersion: '12.0' // This would be detected
					};
				}

				// Get service configuration
				detailedInfo.configuration = {
					neo4jUrl: process.env.NEO4J_URL || 'bolt://localhost:7687',
					redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
					batchSize: 32,
					maxConcurrentQueries: 100,
					cacheEnabled: true
				};

			} catch (err) {
				detailedInfo.detailedInfoError = err instanceof Error ? err.message : 'Failed to get detailed info';
			}
		}

		if (includeMetrics && gpuAvailable) {
			try {
				// Get performance metrics
				const metrics = await gpu.getMetrics();
				detailedInfo.metrics = {
					totalRequests: metrics.totalRequests || 0,
					averageResponseTime: metrics.averageResponseTime || 0,
					cacheHitRatio: metrics.cacheHitRatio || 0,
					errorRate: metrics.errorRate || 0,
					gpuUtilization: metrics.gpuUtilization || 0,
					memoryUtilization: metrics.memoryUtilization || 0
				};
			} catch (err) {
				detailedInfo.metricsError = err instanceof Error ? err.message : 'Failed to get metrics';
			}
		}

		return json(detailedInfo);

	} catch (err) {
		console.error('Health check API error:', err);
		
		return json({
			service: 'gpu-neo4j-worker',
			status: 'unhealthy',
			error: err instanceof Error ? err.message : 'Health check failed',
			timestamp: new Date().toISOString(),
			checks: {
				gpu: { status: 'fail', error: 'Service initialization failed' },
				neo4j: { status: 'unknown' },
				redis: { status: 'unknown' }
			}
		}, { status: 500 });
	}
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const { action } = await request.json();

		switch (action) {
			case 'restart':
				// Reset GPU service instance
				gpuService = null;
				return json({ 
					success: true, 
					message: 'GPU service reset - will reinitialize on next request' 
				});

			case 'clear-cache':
				try {
					const gpu = await getGPUService();
					await gpu.clearCache();
					return json({ 
						success: true, 
						message: 'Cache cleared successfully' 
					});
				} catch (err) {
					return json({ 
						success: false, 
						error: err instanceof Error ? err.message : 'Cache clear failed' 
					}, { status: 500 });
				}

			case 'warm-up':
				try {
					const gpu = await getGPUService();
					await gpu.initialize();
					
					// Perform a test query to warm up the system
					await gpu.search({
						query: 'test query for system warmup',
						limit: 1,
						threshold: 0.5,
						useGPU: true
					});

					return json({ 
						success: true, 
						message: 'System warmed up successfully' 
					});
				} catch (err) {
					return json({ 
						success: false, 
						error: err instanceof Error ? err.message : 'Warm up failed' 
					}, { status: 500 });
				}

			default:
				return json({ 
					success: false, 
					error: `Unknown action: ${action}`,
					availableActions: ['restart', 'clear-cache', 'warm-up']
				}, { status: 400 });
		}

	} catch (err) {
		console.error('Health action API error:', err);
		
		return json({
			success: false,
			error: err instanceof Error ? err.message : 'Health action failed'
		}, { status: 500 });
	}
};