/**
 * RabbitMQ Queue Statistics API Endpoint
 * 
 * Provides detailed statistics about queue performance and message flow
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url }) => {
	try {
		const queueName = url.searchParams.get('queue');
		const detailed = url.searchParams.get('detailed') === 'true';

		// In a real implementation, this would query actual RabbitMQ Management API
		// For development, we'll simulate realistic queue statistics
		
		const timestamp = new Date().toISOString();
		const baseStats = {
			'legal.docs.process': {
				name: 'legal.docs.process',
				messages: Math.floor(Math.random() * 10),
				messages_ready: Math.floor(Math.random() * 5),
				messages_unacknowledged: Math.floor(Math.random() * 3),
				consumers: 1,
				consumer_utilisation: 0.85,
				memory: 45632,
				state: 'running',
				throughput: {
					publish_rate: 2.3,
					deliver_rate: 2.1,
					ack_rate: 2.1
				}
			},
			'legal.chunks.embed': {
				name: 'legal.chunks.embed', 
				messages: Math.floor(Math.random() * 50),
				messages_ready: Math.floor(Math.random() * 25),
				messages_unacknowledged: Math.floor(Math.random() * 10),
				consumers: 2,
				consumer_utilisation: 0.92,
				memory: 123456,
				state: 'running',
				throughput: {
					publish_rate: 8.7,
					deliver_rate: 8.5,
					ack_rate: 8.3
				}
			},
			'legal.chunks.store': {
				name: 'legal.chunks.store',
				messages: Math.floor(Math.random() * 30),
				messages_ready: Math.floor(Math.random() * 15),
				messages_unacknowledged: Math.floor(Math.random() * 5),
				consumers: 1,
				consumer_utilisation: 0.78,
				memory: 78901,
				state: 'running',
				throughput: {
					publish_rate: 5.2,
					deliver_rate: 5.0,
					ack_rate: 4.9
				}
			},
			'legal.dlq': {
				name: 'legal.dlq',
				messages: Math.floor(Math.random() * 3),
				messages_ready: Math.floor(Math.random() * 3),
				messages_unacknowledged: 0,
				consumers: 0,
				consumer_utilisation: 0,
				memory: 12345,
				state: 'running',
				throughput: {
					publish_rate: 0.1,
					deliver_rate: 0,
					ack_rate: 0
				}
			}
		};

		// If specific queue requested, return only that queue's stats
		if (queueName && baseStats[queueName as keyof typeof baseStats]) {
			const queueStats = baseStats[queueName as keyof typeof baseStats];
			
			if (detailed) {
				return json({
					...queueStats,
					timestamp,
					detailed_metrics: {
						message_rates: {
							last_5_minutes: {
								published: Math.floor(Math.random() * 100),
								delivered: Math.floor(Math.random() * 95),
								acknowledged: Math.floor(Math.random() * 95),
								redelivered: Math.floor(Math.random() * 5)
							},
							last_hour: {
								published: Math.floor(Math.random() * 1000),
								delivered: Math.floor(Math.random() * 950),
								acknowledged: Math.floor(Math.random() * 950),
								redelivered: Math.floor(Math.random() * 50)
							}
						},
						consumer_details: Array.from({ length: queueStats.consumers }, (_, i) => ({
							tag: `consumer_${i + 1}`,
							channel: `channel_${i + 1}`,
							prefetch_count: 10,
							ack_required: true,
							active: true
						}))
					}
				});
			}
			
			return json({ [queueName]: queueStats, timestamp });
		}

		// Return all queue statistics
		const allStats = {
			timestamp,
			total_queues: Object.keys(baseStats).length,
			total_messages: Object.values(baseStats).reduce((sum, queue) => sum + queue.messages, 0),
			total_consumers: Object.values(baseStats).reduce((sum, queue) => sum + queue.consumers, 0),
			queues: baseStats
		};

		if (detailed) {
			allStats['system_metrics'] = {
				memory_usage: '256MB',
				cpu_usage: '15%',
				disk_usage: '45%',
				uptime: '2d 14h 35m',
				erlang_processes: 428,
				file_descriptors: {
					used: 156,
					available: 65536
				},
				connection_count: 12,
				channel_count: 24
			};
		}

		return json(allStats, {
			headers: {
				'Cache-Control': 'max-age=30', // Cache for 30 seconds
				'X-Queue-Count': String(Object.keys(baseStats).length)
			}
		});

	} catch (error) {
		console.error('Failed to fetch queue statistics:', error);
		
		return json({
			error: 'Failed to fetch queue statistics',
			details: error instanceof Error ? error.message : 'Unknown error',
			timestamp: new Date().toISOString()
		}, { status: 500 });
	}
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const { action, queue } = await request.json();

		// Handle queue management actions
		switch (action) {
			case 'purge':
				if (!queue) {
					return json({ error: 'Queue name required for purge action' }, { status: 400 });
				}
				
				console.log(`🧹 Purging queue: ${queue}`);
				return json({
					action: 'purge',
					queue,
					result: 'success',
					messages_purged: Math.floor(Math.random() * 50),
					timestamp: new Date().toISOString()
				});

			case 'reset_stats':
				console.log('📊 Resetting queue statistics');
				return json({
					action: 'reset_stats',
					result: 'success',
					timestamp: new Date().toISOString()
				});

			default:
				return json({ error: `Unknown action: ${action}` }, { status: 400 });
		}

	} catch (error) {
		console.error('Queue management action failed:', error);
		
		return json({
			error: 'Queue management action failed',
			details: error instanceof Error ? error.message : 'Unknown error',
			timestamp: new Date().toISOString()
		}, { status: 500 });
	}
};