/**
 * RabbitMQ Message Queue Service
 * Production-ready messaging system for async task processing
 */

import amqp, { type Connection, type Channel, type Message } from 'amqplib';

// RabbitMQ configuration
const RABBITMQ_CONFIG = {
  url: process.env.RABBITMQ_URL || 'amqp://localhost:5672',
  username: process.env.RABBITMQ_USERNAME || 'guest',
  password: process.env.RABBITMQ_PASSWORD || 'guest',
  vhost: process.env.RABBITMQ_VHOST || '/',
  heartbeat: 60
};

// Queue configurations
const QUEUES = {
  DOCUMENT_PROCESSING: 'document.processing',
  FILE_UPLOAD: 'file.upload',
  VECTOR_EMBEDDING: 'vector.embedding',
  RAG_PROCESSING: 'rag.processing',
  EMAIL_NOTIFICATIONS: 'email.notifications',
  SEARCH_INDEXING: 'search.indexing',
  CASE_UPDATES: 'case.updates',
  EVIDENCE_ANALYSIS: 'evidence.analysis'
} as const;

export interface MessageHandler {
  (message: any, originalMessage: Message): Promise<void>;
}

export class RabbitMQService {
  private static instance: RabbitMQService;
  private connection: Connection | null = null;
  private channel: Channel | null = null;
  private isConnected = false;

  static getInstance(): RabbitMQService {
    if (!RabbitMQService.instance) {
      RabbitMQService.instance = new RabbitMQService();
    }
    return RabbitMQService.instance;
  }

  async connect(): Promise<boolean> {
    try {
      this.connection = await amqp.connect(RABBITMQ_CONFIG.url);
      this.channel = await this.connection.createChannel();
      
      await this.setupQueues();
      
      this.isConnected = true;
      console.log('✅ RabbitMQ connected');
      return true;
    } catch (error) {
      console.error('❌ RabbitMQ connection failed:', error);
      return false;
    }
  }

  private async setupQueues(): Promise<void> {
    if (!this.channel) return;

    for (const queue of Object.values(QUEUES)) {
      await this.channel.assertQueue(queue, { durable: true });
    }
  }

  async publish(queue: string, message: any): Promise<boolean> {
    if (!this.channel) return false;

    try {
      const messageBuffer = Buffer.from(JSON.stringify(message));
      return this.channel.sendToQueue(queue, messageBuffer, { persistent: true });
    } catch (error) {
      console.error('❌ Failed to publish message:', error);
      return false;
    }
  }

  async consume(queue: string, handler: MessageHandler): Promise<void> {
    if (!this.channel) return;

    await this.channel.consume(queue, async (msg) => {
      if (msg) {
        try {
          const content = JSON.parse(msg.content.toString());
          await handler(content, msg);
          this.channel!.ack(msg);
        } catch (error) {
          console.error('❌ Message processing error:', error);
          this.channel!.nack(msg);
        }
      }
    });
  }

  async healthCheck(): Promise<{ status: string; details: any }> {
    try {
      if (!this.isConnected || !this.connection) {
        return { status: 'unhealthy', details: { error: 'Not connected' } };
      }

      return {
        status: 'healthy',
        details: { connected: this.isConnected, queues: Object.keys(QUEUES).length }
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        details: { error: error instanceof Error ? error.message : 'Unknown error' }
      };
    }
  }

  async disconnect(): Promise<void> {
    if (this.channel) await this.channel.close();
    if (this.connection) await this.connection.close();
    this.isConnected = false;
    console.log('👋 RabbitMQ disconnected');
  }
}

export const rabbitmqService = RabbitMQService.getInstance();
export { QUEUES };