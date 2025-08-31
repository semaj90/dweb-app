import type { RequestHandler } from './$types';

// Enhanced NATS API Integration
// Production-ready NATS messaging endpoints with full 17-subject pattern support

import { EnhancedNATSMessagingService } from '$lib/services/enhanced-nats-messaging';
import type { 
	LegalAIMessage, 
	MessageBatch,
	ProcessingResult,
	NATSConfig,
	SubscriptionOptions 
} from '$lib/types/nats-messaging';

// Global NATS service instance
let natsService: EnhancedNATSMessagingService | null = null;

// Initialize NATS service
function getNATSService(): EnhancedNATSMessagingService {
	if (!natsService) {
		natsService = new EnhancedNATSMessagingService({
			servers: ['ws://localhost:4222', 'ws://localhost:4223'],
			user: 'legal_ai_client',
			pass: 'legal_ai_2024',
			name: 'SvelteKit Legal AI Client',
			max_reconnect_attempts: -1,
		});
		
		// Auto-connect on first use
		natsService.connect().catch(error => {
			console.error('NATS auto-connect failed:', error);
		});
	}
	return natsService;
}

/** POST /api/v1/nats - Publish message or perform NATS operations */
export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();
		const nats = getNATSService();

		switch (body.action) {
			case 'publish':
				return await handlePublish(nats, body);
			case 'publish_batch':
				return await handlePublishBatch(nats, body);
			case 'request':
				return await handleRequest(nats, body);
			case 'subscribe':
				return await handleSubscribe(nats, body);
			case 'unsubscribe':
				return await handleUnsubscribe(nats, body);
			case 'create_stream':
				return await handleCreateStream(nats, body);
			case 'create_consumer':
				return await handleCreateConsumer(nats, body);
			default:
				return json({ 
					success: false, 
					error: `Unsupported action: ${body.action}` 
				}, { status: 400 });
		}
	} catch (error: any) {
		console.error('NATS API Error:', error);
		return json({
			success: false,
			error: 'NATS operation failed',
			details: error instanceof Error ? error.message : 'Unknown error',
		}, { status: 500 });
	}
};

/** GET /api/v1/nats - Get NATS system status and metrics */
export const GET: RequestHandler = async () => {
	try {
		const nats = getNATSService();
		
		const [metrics, systemStatus] = await Promise.all([
			nats.getMetrics(),
			nats.getSystemStatus(),
		]);

		return json({
			service: 'Enhanced NATS Messaging',
			status: 'operational',
			version: '2.0.0',
			timestamp: new Date().toISOString(),
			connection_status: systemStatus.connection_status,
			metrics,
			system_status: systemStatus,
			supported_subjects: {
				case_management: [
					'legal.case.created',
					'legal.case.updated', 
					'legal.case.closed'
				],
				document_processing: [
					'legal.document.uploaded',
					'legal.document.processed',
					'legal.document.analyzed',
					'legal.document.indexed'
				],
				ai_analysis: [
					'legal.ai.analysis.started',
					'legal.ai.analysis.completed',
					'legal.ai.analysis.failed'
				],
				search_retrieval: [
					'legal.search.query',
					'legal.search.results'
				],
				real_time_communication: [
					'legal.chat.message',
					'legal.chat.response',
					'legal.chat.streaming'
				],
				system_monitoring: [
					'system.health',
					'system.metrics'
				],
			},
			capabilities: {
				message_publishing: true,
				batch_publishing: true,
				request_reply: true,
				stream_processing: true,
				durable_consumers: true,
				wildcard_subscriptions: true,
				message_persistence: true,
				real_time_streaming: true,
			},
		});
	} catch (error: any) {
		return json({
			service: 'Enhanced NATS Messaging',
			status: 'degraded',
			error: 'Unable to get NATS status',
			details: error instanceof Error ? error.message : 'Unknown error',
		}, { status: 503 });
	}
};

// Handler functions

async function handlePublish(nats: EnhancedNATSMessagingService, body: any): Promise<any> {
	if (!body.subject || body.data === undefined) {
		throw new Error('Subject and data are required for publish');
	}

	await nats.publish(body.subject, body.data, body.options);

	return json({
		success: true,
		action: 'publish',
		subject: body.subject,
		message_id: body.options?.correlation_id || 'auto-generated',
		timestamp: new Date().toISOString(),
	});
}

async function handlePublishBatch(nats: EnhancedNATSMessagingService, body: any): Promise<any> {
	if (!body.messages || !Array.isArray(body.messages)) {
		throw new Error('Messages array is required for batch publish');
	}

	await nats.publishBatch(body.messages);

	return json({
		success: true,
		action: 'publish_batch',
		message_count: body.messages.length,
		timestamp: new Date().toISOString(),
	});
}

async function handleRequest(nats: EnhancedNATSMessagingService, body: any): Promise<any> {
	if (!body.subject || body.data === undefined) {
		throw new Error('Subject and data are required for request');
	}

	const timeout = body.timeout_ms || 5000;
	const response = await nats.request(body.subject, body.data, timeout);

	return json({
		success: true,
		action: 'request',
		subject: body.subject,
		response,
		timestamp: new Date().toISOString(),
	});
}

async function handleSubscribe(nats: EnhancedNATSMessagingService, body: any): Promise<any> {
	if (!body.subject) {
		throw new Error('Subject is required for subscription');
	}

	// For HTTP API, we can't maintain persistent subscriptions
	// This would typically be used with WebSocket connections
	return json({
		success: false,
		error: 'HTTP subscriptions not supported',
		suggestion: 'Use WebSocket endpoint for real-time subscriptions',
		websocket_url: '/api/v1/nats/ws',
	}, { status: 400 });
}

async function handleUnsubscribe(nats: EnhancedNATSMessagingService, body: any): Promise<any> {
	if (!body.subject) {
		throw new Error('Subject is required for unsubscribe');
	}

	await nats.unsubscribe(body.subject);

	return json({
		success: true,
		action: 'unsubscribe',
		subject: body.subject,
		timestamp: new Date().toISOString(),
	});
}

async function handleCreateStream(nats: EnhancedNATSMessagingService, body: any): Promise<any> {
	if (!body.stream_config) {
		throw new Error('Stream configuration is required');
	}

	await nats.createStream(body.stream_config);

	return json({
		success: true,
		action: 'create_stream',
		stream_name: body.stream_config.name,
		subjects: body.stream_config.subjects,
		timestamp: new Date().toISOString(),
	});
}

async function handleCreateConsumer(nats: EnhancedNATSMessagingService, body: any): Promise<any> {
	if (!body.stream_name || !body.consumer_config) {
		throw new Error('Stream name and consumer configuration are required');
	}

	await nats.createConsumer(body.stream_name, body.consumer_config);

	return json({
		success: true,
		action: 'create_consumer',
		stream_name: body.stream_name,
		consumer_name: body.consumer_config.name,
		timestamp: new Date().toISOString(),
	});
}