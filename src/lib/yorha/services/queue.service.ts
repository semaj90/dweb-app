// RabbitMQ Queue Service for YoRHa Interface
import amqp from 'amqplib';
import { env } from '$env/dynamic/private';

const RABBITMQ_URL = env.RABBITMQ_URL || 'amqp://localhost:5672';

export class QueueService {
  private connection: amqp.Connection | null = null;
  private channel: amqp.Channel | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private isConnecting = false;

  // Queue definitions
  private queues = {
    email: 'yorha.email',
    achievements: 'yorha.achievements',
    activity: 'yorha.activity',
    notifications: 'yorha.notifications',
    missions: 'yorha.missions',
    analytics: 'yorha.analytics',
    vectorProcessing: 'yorha.vector.processing'
  };

  // Exchange definitions
  private exchanges = {
    direct: 'yorha.direct',
    topic: 'yorha.topic',
    fanout: 'yorha.fanout'
  };

  constructor() {
    this.connect();
  }

  // Connect to RabbitMQ
  private async connect(): Promise<void> {
    if (this.isConnecting) return;
    this.isConnecting = true;

    try {
      this.connection = await amqp.connect(RABBITMQ_URL);
      this.channel = await this.connection.createChannel();

      // Set up error handlers
      this.connection.on('error', (err) => {
        console.error('RabbitMQ connection error:', err);
        this.reconnect();
      });

      this.connection.on('close', () => {
        console.log('RabbitMQ connection closed');
        this.reconnect();
      });

      // Initialize exchanges and queues
      await this.setupInfrastructure();

      console.log('✅ Connected to RabbitMQ');
      this.isConnecting = false;
    } catch (error) {
      console.error('Failed to connect to RabbitMQ:', error);
      this.isConnecting = false;
      this.reconnect();
    }
  }

  // Reconnect logic
  private reconnect(): void {
    if (this.reconnectTimeout) return;

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, 5000);
  }

  // Set up exchanges and queues
  private async setupInfrastructure(): Promise<void> {
    if (!this.channel) return;

    // Create exchanges
    await this.channel.assertExchange(this.exchanges.direct, 'direct', { durable: true });
    await this.channel.assertExchange(this.exchanges.topic, 'topic', { durable: true });
    await this.channel.assertExchange(this.exchanges.fanout, 'fanout', { durable: true });

    // Create queues
    for (const [key, queueName] of Object.entries(this.queues)) {
      await this.channel.assertQueue(queueName, {
        durable: true,
        arguments: {
          'x-message-ttl': 86400000, // 24 hours
          'x-max-length': 10000
        }
      });
    }

    // Bind queues to exchanges
    await this.channel.bindQueue(this.queues.email, this.exchanges.direct, 'email');
    await this.channel.bindQueue(this.queues.achievements, this.exchanges.direct, 'achievements');
    await this.channel.bindQueue(this.queues.activity, this.exchanges.topic, 'activity.*');
    await this.channel.bindQueue(this.queues.notifications, this.exchanges.fanout, '');
  }

  // Publish message to queue
  async publishMessage(queue: keyof typeof this.queues, message: any): Promise<void> {
    if (!this.channel) {
      console.error('RabbitMQ channel not available');
      return;
    }

    try {
      const queueName = this.queues[queue];
      const buffer = Buffer.from(JSON.stringify(message));

      await this.channel.sendToQueue(queueName, buffer, {
        persistent: true,
        timestamp: Date.now()
      });

      console.log(`Message published to ${queueName}`);
    } catch (error) {
      console.error(`Failed to publish message to ${queue}:`, error);
      throw error;
    }
  }

  // Publish to exchange
  async publishToExchange(exchange: keyof typeof this.exchanges, routingKey: string, message: any): Promise<void> {
    if (!this.channel) {
      console.error('RabbitMQ channel not available');
      return;
    }

    try {
      const exchangeName = this.exchanges[exchange];
      const buffer = Buffer.from(JSON.stringify(message));

      await this.channel.publish(exchangeName, routingKey, buffer, {
        persistent: true,
        timestamp: Date.now()
      });

      console.log(`Message published to exchange ${exchangeName} with routing key ${routingKey}`);
    } catch (error) {
      console.error(`Failed to publish to exchange ${exchange}:`, error);
      throw error;
    }
  }

  // Consume messages from queue
  async consume(
    queue: keyof typeof this.queues,
    handler: (message: any) => Promise<void>,
    options: { prefetch?: number } = {}
  ): Promise<void> {
    if (!this.channel) {
      console.error('RabbitMQ channel not available');
      return;
    }

    try {
      const queueName = this.queues[queue];
      
      // Set prefetch count
      if (options.prefetch) {
        await this.channel.prefetch(options.prefetch);
      }

      await this.channel.consume(queueName, async (msg) => {
        if (!msg) return;

        try {
          const content = JSON.parse(msg.content.toString());
          await handler(content);
          
          // Acknowledge message
          if (this.channel) {
            this.channel.ack(msg);
          }
        } catch (error) {
          console.error(`Error processing message from ${queueName}:`, error);
          
          // Reject message and requeue
          if (this.channel) {
            this.channel.nack(msg, false, true);
          }
        }
      });

      console.log(`Started consuming from ${queueName}`);
    } catch (error) {
      console.error(`Failed to consume from ${queue}:`, error);
      throw error;
    }
  }

  // Batch publish messages
  async batchPublish(messages: Array<{ queue: keyof typeof this.queues; content: any }>): Promise<void> {
    if (!this.channel) {
      console.error('RabbitMQ channel not available');
      return;
    }

    const promises = messages.map(({ queue, content }) => 
      this.publishMessage(queue as keyof typeof this.queues, content)
    );

    await Promise.all(promises);
  }

  // Create delayed message
  async publishDelayedMessage(
    queue: keyof typeof this.queues,
    message: any,
    delayMs: number
  ): Promise<void> {
    if (!this.channel) {
      console.error('RabbitMQ channel not available');
      return;
    }

    try {
      // Create delayed queue
      const delayedQueueName = `${this.queues[queue]}.delayed`;
      await this.channel.assertQueue(delayedQueueName, {
        durable: true,
        arguments: {
          'x-message-ttl': delayMs,
          'x-dead-letter-exchange': '',
          'x-dead-letter-routing-key': this.queues[queue]
        }
      });

      const buffer = Buffer.from(JSON.stringify(message));
      await this.channel.sendToQueue(delayedQueueName, buffer, {
        persistent: true
      });

      console.log(`Delayed message published to ${delayedQueueName} with ${delayMs}ms delay`);
    } catch (error) {
      console.error(`Failed to publish delayed message:`, error);
      throw error;
    }
  }

  // Get queue statistics
  async getQueueStats(queue: keyof typeof this.queues): Promise<any> {
    if (!this.channel) {
      console.error('RabbitMQ channel not available');
      return null;
    }

    try {
      const queueName = this.queues[queue];
      const stats = await this.channel.checkQueue(queueName);
      return {
        name: stats.queue,
        messageCount: stats.messageCount,
        consumerCount: stats.consumerCount
      };
    } catch (error) {
      console.error(`Failed to get queue stats for ${queue}:`, error);
      return null;
    }
  }

  // Close connection
  async close(): Promise<void> {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.channel) {
      await this.channel.close();
      this.channel = null;
    }

    if (this.connection) {
      await this.connection.close();
      this.connection = null;
    }

    console.log('RabbitMQ connection closed');
  }
}

// Singleton instance
let queueServiceInstance: QueueService | null = null;

export function getQueueService(): QueueService {
  if (!queueServiceInstance) {
    queueServiceInstance = new QueueService();
  }
  return queueServiceInstance;
}