import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types.js';
import { URL } from "url";

// Import worker and services with error handling
let documentProcessingWorker: any;
let rabbitMQService: any;

try {
  documentProcessingWorker = await import('$lib/workers/document-processing-worker').then(m => m.documentProcessingWorker);
} catch (error) {
  console.warn('Document processing worker not available:', error);
  documentProcessingWorker = {
    getStats: () => ({ isRunning: false, processedCount: 0, failedCount: 0, successRate: 0 }),
    start: () => Promise.reject(new Error('Worker not available')),
    stop: () => Promise.reject(new Error('Worker not available'))
  };
}

try {
  rabbitMQService = await import('$lib/services/rabbitmq-service').then(m => m.rabbitMQService);
} catch (error) {
  console.warn('RabbitMQ service not available:', error);
  rabbitMQService = {
    getQueueStats: () => Promise.resolve({}),
    healthCheck: () => Promise.resolve({ healthy: false, queues: [] })
  };
}


export const GET: RequestHandler = async ({ url }) => {
  try {
    const action = url.searchParams.get('action');

    switch (action) {
      case 'status':
        const workerStats = documentProcessingWorker.getStats();
        const queueStats = await rabbitMQService.getQueueStats();
        const healthCheck = await rabbitMQService.healthCheck();

        return json({
          worker: workerStats,
          queues: queueStats,
          messaging: healthCheck,
          timestamp: new Date().toISOString()
        });

      case 'health':
        const health = await rabbitMQService.healthCheck();
        const stats = documentProcessingWorker.getStats();

        return json({
          status: health.healthy && stats.isRunning ? 'healthy' : 'unhealthy',
          worker: {
            running: stats.isRunning,
            processed: stats.processedCount,
            failed: stats.failedCount,
            successRate: stats.successRate
          },
          messaging: {
            connected: health.healthy,
            queues: health.queues
          }
        });

      default:
        return json({
          error: 'Invalid action. Use ?action=status or ?action=health'
        }, { status: 400 });
    }

  } catch (error: any) {
    console.error('Worker API error:', error);

    return json({
      error: 'Failed to get worker status',
      details: error.message
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action } = await request.json();

    switch (action) {
      case 'start':
        if (documentProcessingWorker.getStats().isRunning) {
          return json({
            message: 'Worker is already running',
            status: 'running'
          });
        }

        await documentProcessingWorker.start();

        return json({
          message: 'Document processing worker started successfully',
          status: 'started'
        });

      case 'stop':
        if (!documentProcessingWorker.getStats().isRunning) {
          return json({
            message: 'Worker is not running',
            status: 'stopped'
          });
        }

        await documentProcessingWorker.stop();

        return json({
          message: 'Document processing worker stopped successfully',
          status: 'stopped'
        });

      case 'restart':
        if (documentProcessingWorker.getStats().isRunning) {
          await documentProcessingWorker.stop();
          // Wait a moment before restarting
          await new Promise(resolve => setTimeout(resolve, 1000));
        }

        await documentProcessingWorker.start();

        return json({
          message: 'Document processing worker restarted successfully',
          status: 'restarted'
        });

      default:
        return json({
          error: 'Invalid action. Use start, stop, or restart'
        }, { status: 400 });
    }

  } catch (error: any) {
    console.error('Worker control error:', error);

    return json({
      error: 'Failed to control worker',
      details: error.message
    }, { status: 500 });
  }
};