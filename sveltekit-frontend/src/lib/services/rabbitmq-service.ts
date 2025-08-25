import { env } from '$env/dynamic/private';
import type { Connection, Channel, Message } from 'amqplib';

interface DocumentProcessingJob {
  documentId: string;
  s3Key: string;
  s3Bucket: string;
  caseId?: string;
  userId?: string;
  originalName: string;
  mimeType: string;
  fileSize: number;
  processingType: 'ocr' | 'embedding' | 'summarization' | 'full_analysis';
  priority: number;
  timestamp: string;
}

interface RabbitMQConfig {
  url: string;
  queues: {
    documentProcessing: string;
    ocrProcessing: string;
    embeddingProcessing: string;
    summarization: string;
  };
  exchanges: {
    documents: string;
    deadLetter: string;
  };
}

class RabbitMQService {
  private connection: Connection | null = null;
  private channel: Channel | null = null;
  private config: RabbitMQConfig;
  private isConnected = false;
  
  constructor() {
    this.config = {
      url: env.RABBITMQ_URL || 'amqp://guest:guest@localhost:5672',
      queues: {
        documentProcessing: 'doc_processing_queue',
        ocrProcessing: 'ocr_processing_queue',
        embeddingProcessing: 'embedding_processing_queue',
        summarization: 'summarization_queue'
      },
      exchanges: {
        documents: 'documents_exchange',
        deadLetter: 'dead_letter_exchange'
      }
    };
  }
  
  async connect(): Promise<void> {
    if (this.isConnected) return;
    
    try {
      // Dynamic import for server-side only
      const amqp = await import('amqplib');
      
      this.connection = await amqp.connect(this.config.url);
      this.channel = await this.connection.createChannel();
      
      // Setup exchanges
      await this.channel.assertExchange(this.config.exchanges.documents, 'direct', { durable: true });
      await this.channel.assertExchange(this.config.exchanges.deadLetter, 'direct', { durable: true });
      
      // Setup queues with dead letter configuration
      const queueOptions = {
        durable: true,
        arguments: {
          'x-dead-letter-exchange': this.config.exchanges.deadLetter,
          'x-message-ttl': 3600000, // 1 hour TTL
          'x-max-retries': 3
        }
      };
      
      await this.channel.assertQueue(this.config.queues.documentProcessing, queueOptions);
      await this.channel.assertQueue(this.config.queues.ocrProcessing, queueOptions);
      await this.channel.assertQueue(this.config.queues.embeddingProcessing, queueOptions);
      await this.channel.assertQueue(this.config.queues.summarization, queueOptions);
      
      // Bind queues to exchange
      await this.channel.bindQueue(this.config.queues.documentProcessing, this.config.exchanges.documents, 'doc.process');
      await this.channel.bindQueue(this.config.queues.ocrProcessing, this.config.exchanges.documents, 'doc.ocr');
      await this.channel.bindQueue(this.config.queues.embeddingProcessing, this.config.exchanges.documents, 'doc.embed');
      await this.channel.bindQueue(this.config.queues.summarization, this.config.exchanges.documents, 'doc.summarize');
      
      // Connection event handlers
      this.connection.on('error', (err) => {
        console.error('RabbitMQ connection error:', err);
        this.isConnected = false;
      });
      
      this.connection.on('close', () => {
        console.log('RabbitMQ connection closed');
        this.isConnected = false;
      });
      
      this.isConnected = true;
      console.log('✅ RabbitMQ connected and configured');
      
    } catch (error) {
      console.error('Failed to connect to RabbitMQ:', error);
      throw error;
    }
  }
  
  async publishDocumentProcessingJob(job: DocumentProcessingJob): Promise<boolean> {
    if (!this.isConnected || !this.channel) {
      await this.connect();
    }
    
    try {
      const message = JSON.stringify({
        ...job,
        timestamp: new Date().toISOString(),
        retryCount: 0
      });
      
      const options = {
        persistent: true,
        priority: job.priority || 5,
        messageId: job.documentId,
        correlationId: job.documentId,
        headers: {
          'x-processing-type': job.processingType,
          'x-document-id': job.documentId,
          'x-user-id': job.userId || 'system'
        }
      };
      
      // Publish to appropriate routing key based on processing type
      let routingKey = 'doc.process';
      switch (job.processingType) {
        case 'ocr':
          routingKey = 'doc.ocr';
          break;
        case 'embedding':
          routingKey = 'doc.embed';
          break;
        case 'summarization':
          routingKey = 'doc.summarize';
          break;
        case 'full_analysis':
          routingKey = 'doc.process';
          break;
      }
      
      const published = this.channel.publish(
        this.config.exchanges.documents,
        routingKey,
        Buffer.from(message),
        options
      );
      
      if (published) {
        console.log(`📨 Document processing job published: ${job.documentId} (${job.processingType})`);
        return true;
      } else {
        console.error('Failed to publish document processing job');
        return false;
      }
      
    } catch (error) {
      console.error('Error publishing document processing job:', error);
      return false;
    }
  }
  
  async publishBatchJobs(jobs: DocumentProcessingJob[]): Promise<{ success: number; failed: number }> {
    const results = { success: 0, failed: 0 };
    
    for (const job of jobs) {
      const published = await this.publishDocumentProcessingJob(job);
      if (published) {
        results.success++;
      } else {
        results.failed++;
      }
    }
    
    return results;
  }
  
  async getQueueStats(): Promise<Record<string, any>> {
    if (!this.isConnected || !this.channel) {
      await this.connect();
    }
    
    try {
      const stats = {};
      
      for (const [queueName, queueKey] of Object.entries(this.config.queues)) {
        const queueInfo = await this.channel!.checkQueue(queueKey);
        stats[queueName] = {
          messageCount: queueInfo.messageCount,
          consumerCount: queueInfo.consumerCount,
          queue: queueKey
        };
      }
      
      return stats;
      
    } catch (error) {
      console.error('Error getting queue stats:', error);
      return {};
    }
  }
  
  async purgeQueue(queueType: keyof RabbitMQConfig['queues']): Promise<boolean> {
    if (!this.isConnected || !this.channel) {
      await this.connect();
    }
    
    try {
      const queueName = this.config.queues[queueType];
      await this.channel!.purgeQueue(queueName);
      console.log(`🗑️ Queue purged: ${queueName}`);
      return true;
      
    } catch (error) {
      console.error(`Error purging queue ${queueType}:`, error);
      return false;
    }
  }
  
  async close(): Promise<void> {
    if (this.connection) {
      await this.connection.close();
      this.isConnected = false;
      console.log('RabbitMQ connection closed');
    }
  }
  
  // Health check
  async healthCheck(): Promise<{ healthy: boolean; queues: Record<string, any> }> {
    try {
      if (!this.isConnected) {
        await this.connect();
      }
      
      const queueStats = await this.getQueueStats();
      
      return {
        healthy: this.isConnected,
        queues: queueStats
      };
      
    } catch (error) {
      return {
        healthy: false,
        queues: {},
        error: error.message
      };
    }
  }
}

// Singleton instance
export const rabbitMQService = new RabbitMQService();

// Types for external use
export type { DocumentProcessingJob };

// Helper function to create processing jobs
export function createDocumentProcessingJob(
  documentId: string,
  s3Key: string,
  s3Bucket: string,
  originalName: string,
  mimeType: string,
  fileSize: number,
  options: {
    caseId?: string;
    userId?: string;
    processingType?: DocumentProcessingJob['processingType'];
    priority?: number;
  } = {}
): DocumentProcessingJob {
  return {
    documentId,
    s3Key,
    s3Bucket,
    originalName,
    mimeType,
    fileSize,
    caseId: options.caseId,
    userId: options.userId,
    processingType: options.processingType || 'full_analysis',
    priority: options.priority || 5,
    timestamp: new Date().toISOString()
  };
}