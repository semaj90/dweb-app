import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getNATSService } from '$lib/services/nats-messaging-service.js';

/**
 * GET /api/v1/nats/status
 * Get NATS server health and connection status
 */
export const GET: RequestHandler = async () => {
  try {
    const natsService = getNATSService();
    
    if (!natsService) {
      return json({
        service: 'not_initialized',
        connected: false,
        error: 'NATS service not available',
        timestamp: Date.now()
      });
    }

    const status = natsService.connectionStatus;
    const metrics = natsService.messageStats;
    const subscriptions = natsService.getSubscriptionStats();

    return json({
      service: 'initialized',
      status: {
        connected: status.connected,
        connecting: status.connecting,
        disconnected: status.disconnected,
        reconnecting: status.reconnecting,
        error: status.error,
        lastConnected: status.lastConnected,
        reconnectAttempts: status.reconnectAttempts
      },
      metrics: {
        subscriptions: status.subscriptions,
        publishedMessages: metrics.published,
        receivedMessages: metrics.received,
        queuedMessages: metrics.queued,
        bytesIn: status.bytesIn,
        bytesOut: status.bytesOut
      },
      subscriptions: subscriptions.map(sub => ({
        subject: sub.subject,
        messageCount: sub.messageCount,
        createdAt: sub.createdAt,
        uptime: Date.now() - sub.createdAt
      })),
      health: {
        overall: status.connected ? 'healthy' : (status.connecting ? 'connecting' : 'unhealthy'),
        uptime: status.lastConnected ? Date.now() - status.lastConnected : 0,
        performance: {
          averageLatency: 'N/A', // Would be calculated from actual metrics
          throughput: metrics.published + metrics.received,
          errorRate: status.error ? 1 : 0
        }
      },
      timestamp: Date.now()
    });

  } catch (error) {
    console.error('NATS status API error:', error);
    return json({ 
      error: 'Failed to get NATS status',
      details: error.message,
      timestamp: Date.now()
    }, { status: 500 });
  }
};