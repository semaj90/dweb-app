#!/usr/bin/env node

/**
 * Legal AI Message Routing Configuration
 * 
 * Configures NATS-based message routing between all orchestration components
 * with proper error handling, dead letter queues, and monitoring.
 */

const { connect, StringCodec, JSONCodec } = require('nats');

class LegalAIMessageRouter {
  constructor() {
    this.natsConnection = null;
    this.subscribers = new Map();
    this.publishers = new Map();
    this.messageStats = new Map();
    this.deadLetterQueue = 'legal.dlq';
    
    // Codecs for different message types
    this.stringCodec = StringCodec();
    this.jsonCodec = JSONCodec();
  }

  async initialize() {
    console.log('[MESSAGE-ROUTER] Initializing Legal AI Message Router...');
    
    try {
      // Connect to NATS server
      this.natsConnection = await connect({
        servers: ['nats://localhost:4222'],
        name: 'legal-ai-message-router',
        maxReconnectAttempts: 10,
        reconnectTimeWait: 2000
      });
      
      console.log('[MESSAGE-ROUTER] Connected to NATS server');
      
      // Set up message routes
      await this.setupMessageRoutes();
      
      // Set up monitoring
      this.setupMonitoring();
      
      console.log('[MESSAGE-ROUTER] Message router initialized successfully');
      
    } catch (error) {
      console.error('[MESSAGE-ROUTER] Failed to initialize:', error);
      throw error;
    }
  }

  async setupMessageRoutes() {
    console.log('[MESSAGE-ROUTER] Setting up message routes...');
    
    // Legal Document Processing Route
    await this.setupLegalDocumentRoute();
    
    // AI Analysis Route
    await this.setupAIAnalysisRoute();
    
    // Vector Operations Route
    await this.setupVectorOperationsRoute();
    
    // System Events Route
    await this.setupSystemEventsRoute();
    
    // Workflow Orchestration Route
    await this.setupWorkflowRoute();
    
    // Health Monitoring Route
    await this.setupHealthMonitoringRoute();
  }

  async setupLegalDocumentRoute() {
    const subject = 'legal.document.*';
    
    // Document processing subscriber
    const documentSub = this.natsConnection.subscribe(subject, {
      queue: 'legal-document-processors'
    });
    
    this.subscribers.set('legal-documents', documentSub);
    
    (async () => {
      for await (const msg of documentSub) {
        try {
          const data = this.jsonCodec.decode(msg.data);
          console.log(`[DOCUMENT-ROUTER] Processing: ${msg.subject}`, data);
          
          // Route to appropriate service based on document type
          switch (data.type) {
            case 'pdf':
              await this.routeToService('node-cluster:legal', msg);
              break;
            case 'contract':
              await this.routeToService('ai-analysis', msg);
              break;
            case 'evidence':
              await this.routeToService('xstate-orchestrator', msg);
              break;
            default:
              await this.routeToDeadLetter(msg, 'Unknown document type');
          }
          
          // Acknowledge message
          msg.respond(this.stringCodec.encode('ACK'));
          
        } catch (error) {
          console.error('[DOCUMENT-ROUTER] Error processing message:', error);
          await this.routeToDeadLetter(msg, error.message);
        }
      }
    })();
    
    console.log('[MESSAGE-ROUTER] Legal document route configured');
  }

  async setupAIAnalysisRoute() {
    const subject = 'legal.ai.*';
    
    const aiSub = this.natsConnection.subscribe(subject, {
      queue: 'ai-analysis-processors'
    });
    
    this.subscribers.set('ai-analysis', aiSub);
    
    (async () => {
      for await (const msg of aiSub) {
        try {
          const data = this.jsonCodec.decode(msg.data);
          console.log(`[AI-ROUTER] Processing: ${msg.subject}`, data);
          
          // Route based on AI operation type
          switch (data.operation) {
            case 'embedding':
              await this.routeToService('webgpu-engine', msg);
              break;
            case 'classification':
              await this.routeToService('node-cluster:ai', msg);
              break;
            case 'precedent-search':
              await this.routeToService('vector-similarity', msg);
              break;
            case 'multi-agent':
              await this.routeToService('xstate-orchestrator', msg);
              break;
            default:
              await this.routeToService('kratos', msg); // Default to main service
          }
          
          msg.respond(this.stringCodec.encode('ACK'));
          
        } catch (error) {
          console.error('[AI-ROUTER] Error processing message:', error);
          await this.routeToDeadLetter(msg, error.message);
        }
      }
    })();
    
    console.log('[MESSAGE-ROUTER] AI analysis route configured');
  }

  async setupVectorOperationsRoute() {
    const subject = 'legal.vector.*';
    
    const vectorSub = this.natsConnection.subscribe(subject, {
      queue: 'vector-processors'
    });
    
    this.subscribers.set('vector-operations', vectorSub);
    
    (async () => {
      for await (const msg of vectorSub) {
        try {
          const data = this.jsonCodec.decode(msg.data);
          console.log(`[VECTOR-ROUTER] Processing: ${msg.subject}`, data);
          
          // Route to appropriate vector service
          switch (data.operation) {
            case 'search':
              await this.routeToService('node-cluster:vector', msg);
              break;
            case 'index':
              await this.routeToService('windows-services:vector', msg);
              break;
            case 'gpu-acceleration':
              await this.routeToService('webgpu-engine', msg);
              break;
            default:
              await this.routeToService('qdrant', msg);
          }
          
          msg.respond(this.stringCodec.encode('ACK'));
          
        } catch (error) {
          console.error('[VECTOR-ROUTER] Error processing message:', error);
          await this.routeToDeadLetter(msg, error.message);
        }
      }
    })();
    
    console.log('[MESSAGE-ROUTER] Vector operations route configured');
  }

  async setupSystemEventsRoute() {
    const subject = 'legal.system.*';
    
    const systemSub = this.natsConnection.subscribe(subject);
    this.subscribers.set('system-events', systemSub);
    
    (async () => {
      for await (const msg of systemSub) {
        try {
          const data = this.jsonCodec.decode(msg.data);
          console.log(`[SYSTEM-ROUTER] Event: ${msg.subject}`, data);
          
          // Route system events to appropriate handlers
          switch (data.type) {
            case 'health-check':
              await this.routeToService('orchestration-controller', msg);
              break;
            case 'error':
              await this.routeToService('elk-stack', msg);
              break;
            case 'metric':
              await this.routeToService('monitoring', msg);
              break;
            case 'alert':
              await this.routeToService('notification-service', msg);
              break;
          }
          
          // System events don't require ACK
          
        } catch (error) {
          console.error('[SYSTEM-ROUTER] Error processing system event:', error);
        }
      }
    })();
    
    console.log('[MESSAGE-ROUTER] System events route configured');
  }

  async setupWorkflowRoute() {
    const subject = 'legal.workflow.*';
    
    const workflowSub = this.natsConnection.subscribe(subject, {
      queue: 'workflow-processors'
    });
    
    this.subscribers.set('workflow', workflowSub);
    
    (async () => {
      for await (const msg of workflowSub) {
        try {
          const data = this.jsonCodec.decode(msg.data);
          console.log(`[WORKFLOW-ROUTER] Processing: ${msg.subject}`, data);
          
          // Route to XState orchestrator
          await this.routeToService('xstate-orchestrator', msg);
          
          msg.respond(this.stringCodec.encode('ACK'));
          
        } catch (error) {
          console.error('[WORKFLOW-ROUTER] Error processing workflow:', error);
          await this.routeToDeadLetter(msg, error.message);
        }
      }
    })();
    
    console.log('[MESSAGE-ROUTER] Workflow route configured');
  }

  async setupHealthMonitoringRoute() {
    const subject = 'legal.health.*';
    
    const healthSub = this.natsConnection.subscribe(subject);
    this.subscribers.set('health-monitoring', healthSub);
    
    (async () => {
      for await (const msg of healthSub) {
        try {
          const data = this.jsonCodec.decode(msg.data);
          console.log(`[HEALTH-ROUTER] Health update: ${msg.subject}`, data);
          
          // Update health status in orchestration controller
          await this.updateHealthStatus(data);
          
        } catch (error) {
          console.error('[HEALTH-ROUTER] Error processing health update:', error);
        }
      }
    })();
    
    console.log('[MESSAGE-ROUTER] Health monitoring route configured');
  }

  async routeToService(serviceName, message) {
    const targetSubject = `service.${serviceName}.request`;
    
    try {
      // Forward message to target service
      const response = await this.natsConnection.request(
        targetSubject,
        message.data,
        { timeout: 30000 }
      );
      
      console.log(`[MESSAGE-ROUTER] Routed to ${serviceName}: ${response.subject}`);
      
      // Update routing statistics
      this.updateMessageStats(serviceName, 'success');
      
      return response;
      
    } catch (error) {
      console.error(`[MESSAGE-ROUTER] Failed to route to ${serviceName}:`, error);
      this.updateMessageStats(serviceName, 'error');
      throw error;
    }
  }

  async routeToDeadLetter(message, reason) {
    const dlqData = {
      originalSubject: message.subject,
      originalData: this.jsonCodec.decode(message.data),
      error: reason,
      timestamp: new Date().toISOString(),
      retryCount: 0
    };
    
    this.natsConnection.publish(
      this.deadLetterQueue,
      this.jsonCodec.encode(dlqData)
    );
    
    console.log(`[MESSAGE-ROUTER] Routed to dead letter queue: ${reason}`);
  }

  updateMessageStats(serviceName, status) {
    const key = `${serviceName}.${status}`;
    const current = this.messageStats.get(key) || 0;
    this.messageStats.set(key, current + 1);
  }

  async updateHealthStatus(healthData) {
    // Update health status in orchestration controller
    try {
      await this.natsConnection.publish(
        'orchestrator.health.update',
        this.jsonCodec.encode(healthData)
      );
    } catch (error) {
      console.error('[MESSAGE-ROUTER] Failed to update health status:', error);
    }
  }

  setupMonitoring() {
    // Periodic statistics reporting
    setInterval(() => {
      this.reportStatistics();
    }, 30000); // Every 30 seconds
    
    // Connection monitoring
    this.natsConnection.closed().then((err) => {
      if (err) {
        console.error('[MESSAGE-ROUTER] NATS connection closed:', err);
      } else {
        console.log('[MESSAGE-ROUTER] NATS connection closed gracefully');
      }
    });
  }

  reportStatistics() {
    const stats = {
      timestamp: new Date().toISOString(),
      subscribers: this.subscribers.size,
      messageStats: Object.fromEntries(this.messageStats),
      connectionStatus: this.natsConnection.info
    };
    
    // Publish statistics to monitoring
    this.natsConnection.publish(
      'legal.system.stats.message-router',
      this.jsonCodec.encode(stats)
    );
    
    console.log('[MESSAGE-ROUTER] Statistics reported:', stats);
  }

  // Message publishing utilities
  async publishDocumentEvent(documentId, eventType, data) {
    const subject = `legal.document.${eventType}`;
    const message = {
      documentId,
      eventType,
      data,
      timestamp: new Date().toISOString()
    };
    
    this.natsConnection.publish(subject, this.jsonCodec.encode(message));
  }

  async publishAIRequest(operation, data, replyTo) {
    const subject = `legal.ai.${operation}`;
    const message = {
      operation,
      data,
      replyTo,
      timestamp: new Date().toISOString()
    };
    
    if (replyTo) {
      return await this.natsConnection.request(
        subject,
        this.jsonCodec.encode(message),
        { timeout: 60000 }
      );
    } else {
      this.natsConnection.publish(subject, this.jsonCodec.encode(message));
    }
  }

  async publishSystemEvent(eventType, data) {
    const subject = `legal.system.${eventType}`;
    const message = {
      eventType,
      data,
      source: 'message-router',
      timestamp: new Date().toISOString()
    };
    
    this.natsConnection.publish(subject, this.jsonCodec.encode(message));
  }

  getRoutingStats() {
    return {
      subscribers: Array.from(this.subscribers.keys()),
      messageStats: Object.fromEntries(this.messageStats),
      connectionInfo: this.natsConnection?.info || null
    };
  }

  async shutdown() {
    console.log('[MESSAGE-ROUTER] Shutting down message router...');
    
    // Close all subscribers
    for (const [name, sub] of this.subscribers) {
      sub.unsubscribe();
      console.log(`[MESSAGE-ROUTER] Unsubscribed from ${name}`);
    }
    
    // Close NATS connection
    if (this.natsConnection) {
      await this.natsConnection.close();
      console.log('[MESSAGE-ROUTER] NATS connection closed');
    }
  }
}

// Export for use in orchestration controller
module.exports = LegalAIMessageRouter;

// Standalone execution
if (require.main === module) {
  const router = new LegalAIMessageRouter();
  
  router.initialize().catch((error) => {
    console.error('[MESSAGE-ROUTER] Startup failed:', error);
    process.exit(1);
  });
  
  // Graceful shutdown
  process.on('SIGINT', async () => {
    await router.shutdown();
    process.exit(0);
  });
}