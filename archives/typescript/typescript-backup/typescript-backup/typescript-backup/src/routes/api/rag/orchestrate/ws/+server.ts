/**
 * WebSocket Handler for Real-time RAG Orchestration Updates
 * 
 * Provides real-time updates for:
 * - Document processing progress
 * - System health status
 * - Service alerts
 * - Performance metrics
 */

import type { RequestHandler } from './$types';
import { ragCoordinator } from '$lib/orchestration/production-rag-coordinator';
import { serviceHealthMonitor } from '$lib/monitoring/service-health-monitor';

const connectedClients = new Map<string, {
  websocket: any;
  subscriptions: Set<string>;
  lastActivity: number;
}>();

// Cleanup inactive connections every 5 minutes
setInterval(() => {
  const now = Date.now();
  const timeout = 5 * 60 * 1000; // 5 minutes
  
  connectedClients.forEach((client, clientId) => {
    if (now - client.lastActivity > timeout) {
      console.log(`[WebSocket] 🧹 Cleaning up inactive client: ${clientId}`);
      client.websocket.close();
      connectedClients.delete(clientId);
    }
  });
}, 5 * 60 * 1000);

export const GET: RequestHandler = async ({ url, request }): Promise<any> => {
  // Check if this is a WebSocket upgrade request
  const upgradeHeader = request.headers.get('upgrade');
  if (!upgradeHeader || upgradeHeader !== 'websocket') {
    return new Response('Expected WebSocket upgrade', { status: 426 });
  }

  try {
    // Extract connection parameters
    const jobId = url.searchParams.get('jobId');
    const subscriptions = url.searchParams.get('subscribe')?.split(',') || [];
    const clientId = `client_${Date.now()}_${Math.random().toString(36).substring(2)}`;

    // Create WebSocket connection (this is a simplified example)
    // In a real implementation, you'd use a proper WebSocket library
    
    console.log(`[WebSocket] 🔌 New connection: ${clientId}`);
    
    // Setup event handlers for the RAG coordinator
    const handleJobProgress = (job: any) => {
      if (!jobId || job.jobId === jobId) {
        broadcastToClient(clientId, {
          type: 'job:progress',
          data: {
            jobId: job.jobId,
            status: job.status,
            progress: job.progress,
            currentStage: getCurrentStage(job),
            timestamp: Date.now()
          }
        });
      }
    };

    const handleJobCompleted = (job: any) => {
      if (!jobId || job.jobId === jobId) {
        broadcastToClient(clientId, {
          type: 'job:completed',
          data: {
            jobId: job.jobId,
            status: job.status,
            processingTime: job.processingTime,
            results: {
              chunks: job.textChunks?.length || 0,
              embeddings: job.embeddings?.length || 0,
              summary: job.summary,
              keyTerms: job.keyTerms
            },
            timestamp: Date.now()
          }
        });
      }
    };

    const handleStageCompleted = ({ jobId: eventJobId, stage, job }: any) => {
      if (!jobId || eventJobId === jobId) {
        broadcastToClient(clientId, {
          type: 'stage:completed',
          data: {
            jobId: eventJobId,
            stage,
            progress: job.progress,
            stageTime: job.stages[stage]?.processingTime,
            timestamp: Date.now()
          }
        });
      }
    };

    const handleStageFailed = ({ jobId: eventJobId, stage, error }: any) => {
      if (!jobId || eventJobId === jobId) {
        broadcastToClient(clientId, {
          type: 'stage:failed',
          data: {
            jobId: eventJobId,
            stage,
            error,
            timestamp: Date.now()
          }
        });
      }
    };

    const handleHealthUpdate = (healthStatus: any) => {
      if (subscriptions.includes('health') || subscriptions.includes('all')) {
        broadcastToClient(clientId, {
          type: 'health:update',
          data: {
            overall: healthStatus.status,
            services: healthStatus.summary,
            timestamp: Date.now()
          }
        });
      }
    };

    const handleServiceHealth = ({ serviceName, metrics, critical }: any) => {
      if (subscriptions.includes('services') || subscriptions.includes('all')) {
        broadcastToClient(clientId, {
          type: 'service:health',
          data: {
            serviceName,
            status: metrics.status,
            responseTime: metrics.responseTime,
            critical,
            timestamp: Date.now()
          }
        });
      }
    };

    const handleAlert = (alert: any) => {
      if (subscriptions.includes('alerts') || subscriptions.includes('all')) {
        broadcastToClient(clientId, {
          type: 'alert',
          data: {
            ...alert,
            timestamp: Date.now()
          }
        });
      }
    };

    // Register event listeners
    ragCoordinator.on('job:progress', handleJobProgress);
    ragCoordinator.on('document:processed', handleJobCompleted);
    ragCoordinator.on('stage:completed', handleStageCompleted);
    ragCoordinator.on('stage:failed', handleStageFailed);
    
    serviceHealthMonitor.on('health:update', handleHealthUpdate);
    serviceHealthMonitor.on('service:health', handleServiceHealth);
    serviceHealthMonitor.on('alert', handleAlert);

    // Simulate WebSocket connection (in real implementation, this would be actual WebSocket)
    const mockWebSocket = {
      send: (message: string) => {
        console.log(`[WebSocket] 📤 Sending to ${clientId}:`, message);
      },
      close: () => {
        console.log(`[WebSocket] 🔌 Connection closed: ${clientId}`);
        
        // Cleanup event listeners
        ragCoordinator.off('job:progress', handleJobProgress);
        ragCoordinator.off('document:processed', handleJobCompleted);
        ragCoordinator.off('stage:completed', handleStageCompleted);
        ragCoordinator.off('stage:failed', handleStageFailed);
        
        serviceHealthMonitor.off('health:update', handleHealthUpdate);
        serviceHealthMonitor.off('service:health', handleServiceHealth);
        serviceHealthMonitor.off('alert', handleAlert);
        
        connectedClients.delete(clientId);
      }
    };

    // Store client connection
    connectedClients.set(clientId, {
      websocket: mockWebSocket,
      subscriptions: new Set(subscriptions),
      lastActivity: Date.now()
    });

    // Send initial connection confirmation
    broadcastToClient(clientId, {
      type: 'connection:established',
      data: {
        clientId,
        subscriptions,
        jobId,
        timestamp: Date.now()
      }
    });

    // If monitoring a specific job, send current status
    if (jobId) {
      const job = await ragCoordinator.getJobStatus(jobId);
      if (job) {
        broadcastToClient(clientId, {
          type: 'job:status',
          data: {
            jobId: job.jobId,
            status: job.status,
            progress: job.progress,
            currentStage: getCurrentStage(job),
            timestamp: Date.now()
          }
        });
      }
    }

    // Send initial health status if subscribed
    if (subscriptions.includes('health') || subscriptions.includes('all')) {
      const healthStatus = serviceHealthMonitor.getOverallHealthStatus();
      broadcastToClient(clientId, {
        type: 'health:status',
        data: {
          overall: healthStatus.status,
          services: healthStatus.summary,
          timestamp: Date.now()
        }
      });
    }

    // Return a response indicating WebSocket upgrade would happen
    // In a real implementation, this would upgrade the connection
    return new Response(JSON.stringify({
      message: 'WebSocket connection established',
      clientId,
      subscriptions,
      jobId
    }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });

  } catch (error: any) {
    console.error('[WebSocket] ❌ Connection failed:', error);
    return new Response('WebSocket connection failed', { status: 500 });
  }
};

/**
 * Broadcast message to a specific client
 */
function broadcastToClient(clientId: string, message: any): void {
  const client = connectedClients.get(clientId);
  if (client) {
    client.lastActivity = Date.now();
    client.websocket.send(JSON.stringify(message));
  }
}

/**
 * Get current processing stage from job
 */
function getCurrentStage(job: any): string | null {
  const stages = Object.entries(job.stages);
  
  // Find the first processing stage
  const processingStage = stages.find(([_, stage]: [string, any]) => stage.status === 'processing');
  if (processingStage) {
    return processingStage[0];
  }
  
  // Find the last completed stage
  const completedStages = stages.filter(([_, stage]: [string, any]) => stage.status === 'completed');
  if (completedStages.length > 0) {
    return completedStages[completedStages.length - 1][0];
  }
  
  return null;
}

/**
 * Broadcast to all clients with specific subscription
 */
export function broadcastToSubscribers(subscription: string, message: any): void {
  connectedClients.forEach((client, clientId) => {
    if (client.subscriptions.has(subscription) || client.subscriptions.has('all')) {
      broadcastToClient(clientId, message);
    }
  });
}

// Export connection stats
export function getConnectionStats(): {
  totalConnections: number;
  subscriptions: { [key: string]: number };
} {
  const subscriptionCounts: { [key: string]: number } = {};
  
  connectedClients.forEach((client) => {
    client.subscriptions.forEach((sub) => {
      subscriptionCounts[sub] = (subscriptionCounts[sub] || 0) + 1;
    });
  });

  return {
    totalConnections: connectedClients.size,
    subscriptions: subscriptionCounts
  };
}