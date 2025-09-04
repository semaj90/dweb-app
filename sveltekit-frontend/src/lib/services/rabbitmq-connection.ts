/**
 * RabbitMQ Connection Service
 * Handles connection management, channel creation, and message publishing
 */

import amqp from 'amqplib';
import { 
  getRabbitMQConfig, 
  getRabbitMQConnectionURL, 
  EXCHANGES, 
  QUEUES, 
  ROUTING_KEYS,
  type ConsumerConfig,
  getConsumerConfig,
  HEALTH_CHECK 
} from '$lib/config/rabbitmq-config';

export interface JobMessage {
  id: string;
  type: string;
  payload: Record<string, any>;
  priority?: number;
  timestamp?: number;
  correlationId?: string;
  replyTo?: string;
  retryCount?: number;
  maxRetries?: number;
  createdBy?: string;
  metadata?: Record<string, any>;
}

export interface JobResult {
  success: boolean;
  result?: any;
  error?: string;
  processingTime?: number;
  timestamp?: number;
}

class RabbitMQConnectionService {
  private connection: amqp.Connection | null = null;
  private channels: Map<string, amqp.Channel> = new Map();
  private isConnected: boolean = false;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10;
  private reconnectDelay: number = 5000;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private consumers: Map<string, { consumer: any; config: ConsumerConfig }> = new Map();

  constructor() {
    // Bind methods to preserve context
    this.connect = this.connect.bind(this);
    this.disconnect = this.disconnect.bind(this);
    this.publish = this.publish.bind(this);
    this.subscribe = this.subscribe.bind(this);
  }

  /**
   * Establish connection to RabbitMQ server
   */
  async connect(): Promise<boolean> {
    try {
      if (this.isConnected && this.connection) {
        return true;
      }

      const connectionUrl = getRabbitMQConnectionURL();
      console.log('🐰 Connecting to RabbitMQ...');

      this.connection = await amqp.connect(connectionUrl);
      this.isConnected = true;
      this.reconnectAttempts = 0;

      // Set up connection event handlers
      this.connection.on('error', (err) => {
        console.error('🐰 RabbitMQ connection error:', err);
        this.isConnected = false;
        this.handleConnectionLoss();
      });

      this.connection.on('close', () => {
        console.log('🐰 RabbitMQ connection closed');
        this.isConnected = false;
        this.handleConnectionLoss();
      });

      // Initialize exchanges and queues
      await this.setupInfrastructure();

      // Start health check
      this.startHealthCheck();

      console.log('✅ RabbitMQ connected successfully');
      return true;

    } catch (error) {
      console.error('❌ RabbitMQ connection failed:', error);
      this.isConnected = false;
      
      // Attempt reconnection
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        console.log(`🔄 Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${this.reconnectDelay}ms`);
        setTimeout(() => this.connect(), this.reconnectDelay);
      } else {
        console.error('❌ Max reconnection attempts reached');
      }
      
      return false;
    }
  }

  /**
   * Handle connection loss and attempt reconnection
   */
  private async handleConnectionLoss(): Promise<void> {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }

    // Clear channels
    this.channels.clear();
    this.consumers.clear();

    // Attempt reconnection
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      await new Promise(resolve => setTimeout(resolve, this.reconnectDelay));
      await this.connect();
    }
  }

  /**
   * Set up exchanges and queues
   */
  private async setupInfrastructure(): Promise<void> {
    if (!this.connection) {
      throw new Error('No RabbitMQ connection available');
    }

    const channel = await this.connection.createChannel();
    const config = getRabbitMQConfig();

    // Create exchanges
    for (const exchange of config.exchanges) {
      await channel.assertExchange(exchange.name, exchange.type, exchange.options);
      console.log(`✅ Exchange '${exchange.name}' ready`);
    }

    // Create queues and bind to exchanges
    for (const queue of config.queues) {
      await channel.assertQueue(queue.name, queue.options);
      
      if (queue.exchange && queue.routingKey) {
        await channel.bindQueue(queue.name, queue.exchange, queue.routingKey);
        console.log(`✅ Queue '${queue.name}' bound to '${queue.exchange}' with key '${queue.routingKey}'`);
      } else {
        console.log(`✅ Queue '${queue.name}' ready`);
      }
    }

    await channel.close();
  }

  /**
   * Get or create a channel
   */
  private async getChannel(channelId: string = 'default'): Promise<amqp.Channel> {
    if (!this.connection) {
      throw new Error('No RabbitMQ connection available');
    }

    let channel = this.channels.get(channelId);
    if (!channel) {
      channel = await this.connection.createChannel();
      this.channels.set(channelId, channel);

      // Handle channel errors
      channel.on('error', (err) => {
        console.error(`🐰 Channel ${channelId} error:`, err);
        this.channels.delete(channelId);
      });

      channel.on('close', () => {
        console.log(`🐰 Channel ${channelId} closed`);
        this.channels.delete(channelId);
      });
    }

    return channel;
  }

  /**
   * Publish a message to an exchange
   */
  async publish(
    exchange: string,
    routingKey: string,
    message: JobMessage,
    options: amqp.Options.Publish = {}
  ): Promise<boolean> {
    try {
      if (!this.isConnected) {
        await this.connect();
      }

      const channel = await this.getChannel('publisher');
      
      // Prepare message
      const messageBuffer = Buffer.from(JSON.stringify(message));
      const publishOptions: amqp.Options.Publish = {
        persistent: true,
        timestamp: Date.now(),
        messageId: message.id,
        correlationId: message.correlationId || message.id,
        priority: message.priority || 5,
        replyTo: message.replyTo,
        ...options
      };

      // Publish message
      const success = channel.publish(exchange, routingKey, messageBuffer, publishOptions);
      
      if (success) {
        console.log(`📤 Published message ${message.id} to ${exchange}:${routingKey}`);
        return true;
      } else {
        console.warn(`⚠️ Failed to publish message ${message.id} - channel buffer full`);
        return false;
      }

    } catch (error) {
      console.error('❌ Failed to publish message:', error);
      throw error;
    }
  }

  /**
   * Publish a job to a specific queue
   */
  async publishJob(
    queueName: string,
    jobType: string,
    payload: Record<string, any>,
    options: {
      priority?: number;
      correlationId?: string;
      replyTo?: string;
      maxRetries?: number;
      createdBy?: string;
      metadata?: Record<string, any>;
    } = {}
  ): Promise<string> {
    const jobId = `${jobType}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const message: JobMessage = {
      id: jobId,
      type: jobType,
      payload,
      timestamp: Date.now(),
      retryCount: 0,
      maxRetries: options.maxRetries || 3,
      ...options
    };

    // Determine routing key based on job type
    let routingKey: string;
    let exchange: string = EXCHANGES.LEGAL_AI_MAIN;

    switch (jobType) {
      case 'generate_embedding':
        routingKey = ROUTING_KEYS.GENERATE_EMBEDDING;
        break;
      case 'analyze_document':
        routingKey = ROUTING_KEYS.ANALYZE_DOCUMENT;
        break;
      case 'extract_entities':
        routingKey = ROUTING_KEYS.EXTRACT_ENTITIES;
        break;
      case 'summarize_content':
        routingKey = ROUTING_KEYS.SUMMARIZE_CONTENT;
        break;
      case 'classify_document':
        routingKey = ROUTING_KEYS.CLASSIFY_DOCUMENT;
        break;
      case 'find_similar_cases':
        routingKey = ROUTING_KEYS.FIND_SIMILAR_CASES;
        break;
      default:
        routingKey = `job.${jobType}`;
    }

    await this.publish(exchange, routingKey, message);
    return jobId;
  }

  /**
   * Subscribe to messages from a queue
   */
  async subscribe(
    queueName: string,
    handler: (message: JobMessage) => Promise<JobResult>,
    options: Partial<ConsumerConfig> = {}
  ): Promise<string> {
    try {
      if (!this.isConnected) {
        await this.connect();
      }

      const channel = await this.getChannel(`consumer_${queueName}`);
      const config = { ...getConsumerConfig(queueName), ...options };

      // Set channel prefetch
      await channel.prefetch(config.prefetchCount);

      const consumerTag = await channel.consume(queueName, async (msg) => {
        if (!msg) return;

        try {
          // Parse message
          const jobMessage: JobMessage = JSON.parse(msg.content.toString());
          console.log(`📥 Processing message ${jobMessage.id} from ${queueName}`);

          const startTime = Date.now();

          // Process message with retry logic
          let result: JobResult;
          let attempts = 0;
          const maxAttempts = config.retryAttempts + 1;

          while (attempts < maxAttempts) {
            try {
              result = await handler(jobMessage);
              break;
            } catch (error) {
              attempts++;
              console.error(`❌ Attempt ${attempts}/${maxAttempts} failed for message ${jobMessage.id}:`, error);

              if (attempts >= maxAttempts) {
                result = {
                  success: false,
                  error: error instanceof Error ? error.message : String(error),
                  processingTime: Date.now() - startTime,
                  timestamp: Date.now()
                };
                break;
              } else {
                // Wait before retry
                const delay = config.exponentialBackoff 
                  ? config.retryDelay * Math.pow(2, attempts - 1)
                  : config.retryDelay;
                await new Promise(resolve => setTimeout(resolve, delay));
              }
            }
          }

          result.processingTime = Date.now() - startTime;
          result.timestamp = Date.now();

          if (result.success) {
            console.log(`✅ Message ${jobMessage.id} processed successfully in ${result.processingTime}ms`);
            channel.ack(msg);
          } else {
            console.error(`❌ Message ${jobMessage.id} failed after ${attempts} attempts`);
            
            // Check if we should retry or move to DLQ
            const retryCount = (jobMessage.retryCount || 0) + 1;
            const maxRetries = jobMessage.maxRetries || 3;

            if (retryCount <= maxRetries) {
              // Retry by publishing to retry queue
              const retryMessage: JobMessage = {
                ...jobMessage,
                retryCount
              };
              
              await this.publish(EXCHANGES.LEGAL_AI_MAIN, ROUTING_KEYS.RETRY_FAILED_JOB, retryMessage);
              console.log(`🔄 Message ${jobMessage.id} sent for retry (${retryCount}/${maxRetries})`);
            } else {
              // Move to dead letter queue
              await this.publish(EXCHANGES.LEGAL_AI_DLX, ROUTING_KEYS.MOVE_TO_DLQ, jobMessage);
              console.log(`💀 Message ${jobMessage.id} moved to dead letter queue`);
            }
            
            channel.ack(msg);
          }

        } catch (error) {
          console.error('❌ Failed to process message:', error);
          channel.nack(msg, false, false); // Don't requeue automatically
        }
      }, { noAck: config.autoAck });

      this.consumers.set(queueName, { consumer: consumerTag, config });
      console.log(`✅ Subscribed to queue '${queueName}' with consumer tag '${consumerTag.consumerTag}'`);
      
      return consumerTag.consumerTag;

    } catch (error) {
      console.error(`❌ Failed to subscribe to queue '${queueName}':`, error);
      throw error;
    }
  }

  /**
   * Unsubscribe from a queue
   */
  async unsubscribe(queueName: string): Promise<void> {
    const consumer = this.consumers.get(queueName);
    if (consumer) {
      const channel = this.channels.get(`consumer_${queueName}`);
      if (channel) {
        await channel.cancel(consumer.consumer.consumerTag);
        console.log(`✅ Unsubscribed from queue '${queueName}'`);
      }
      this.consumers.delete(queueName);
    }
  }

  /**
   * Get queue information
   */
  async getQueueInfo(queueName: string): Promise<{ messageCount: number; consumerCount: number }> {
    try {
      const channel = await this.getChannel('info');
      const queueInfo = await channel.checkQueue(queueName);
      return {
        messageCount: queueInfo.messageCount,
        consumerCount: queueInfo.consumerCount
      };
    } catch (error) {
      console.error(`❌ Failed to get queue info for '${queueName}':`, error);
      throw error;
    }
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{
    connected: boolean;
    channels: number;
    consumers: number;
    queues: Record<string, { messageCount: number; consumerCount: number }>;
  }> {
    const health = {
      connected: this.isConnected,
      channels: this.channels.size,
      consumers: this.consumers.size,
      queues: {} as Record<string, { messageCount: number; consumerCount: number }>
    };

    if (this.isConnected) {
      try {
        // Check key queues
        for (const queueName of Object.values(QUEUES)) {
          try {
            health.queues[queueName] = await this.getQueueInfo(queueName);
          } catch (error) {
            // Queue might not exist yet
            health.queues[queueName] = { messageCount: -1, consumerCount: 0 };
          }
        }
      } catch (error) {
        console.error('❌ Health check failed:', error);
      }
    }

    return health;
  }

  /**
   * Start periodic health check
   */
  private startHealthCheck(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    this.healthCheckInterval = setInterval(async () => {
      try {
        await this.healthCheck();
      } catch (error) {
        console.error('❌ Health check failed:', error);
        if (!this.isConnected) {
          await this.connect();
        }
      }
    }, HEALTH_CHECK.interval);
  }

  /**
   * Disconnect from RabbitMQ
   */
  async disconnect(): Promise<void> {
    console.log('🐰 Disconnecting from RabbitMQ...');

    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }

    // Unsubscribe from all consumers
    for (const queueName of this.consumers.keys()) {
      await this.unsubscribe(queueName);
    }

    // Close all channels
    for (const channel of this.channels.values()) {
      try {
        await channel.close();
      } catch (error) {
        // Channel might already be closed
      }
    }
    this.channels.clear();

    // Close connection
    if (this.connection) {
      try {
        await this.connection.close();
      } catch (error) {
        // Connection might already be closed
      }
      this.connection = null;
    }

    this.isConnected = false;
    console.log('✅ RabbitMQ disconnected');
  }
}

// Singleton instance
export const rabbitMQService = new RabbitMQConnectionService();
export default rabbitMQService;