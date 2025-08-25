import { json, type RequestHandler } from '@sveltejs/kit';

export const POST: RequestHandler = async ({ request }) => {
	try {
		const data = await request.json();
		
		// Log service worker metrics for debugging
		console.log('📊 Service Worker Metrics:', {
			timestamp: new Date().toISOString(),
			...data
		});

		// In a production environment, you might want to:
		// - Store metrics in a database
		// - Send to analytics service
		// - Process performance data
		
		return json({
			success: true,
			message: 'Service worker metrics received',
			timestamp: new Date().toISOString()
		});
	} catch (error) {
		console.error('❌ Error processing service worker metrics:', error);
		
		return json({
			success: false,
			error: 'Failed to process service worker metrics'
		}, { status: 400 });
	}
};

export const GET: RequestHandler = async () => {
	// Return basic service worker status information
	return json({
		endpoint: '/api/metrics/service-worker',
		status: 'active',
		description: 'Service worker metrics collection endpoint',
		methods: ['GET', 'POST'],
		timestamp: new Date().toISOString()
	});
};