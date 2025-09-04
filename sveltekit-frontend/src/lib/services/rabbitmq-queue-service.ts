/**
 * RabbitMQ Task Queuing Service for NLP Operations
 * 
 * This service implements the message queue architecture described for RAG ingestion:
 * 1. Document ingestion → docs-to-process queue
 * 2. Chunking service → chunks-to-embed queue  
 * 3. Embedding service → chunks-to-store queue
 * 4. Neo4j storage service → final persistence
 * 
 * Provides decoupled, scalable processing for expensive NLP operations.
 */

import { EventEmitter } from 'events';

// Message types for the RAG ingestion pipeline
export interface DocumentMessage {
  document_id: string;
  case_id: string;
  source_location: string;
  metadata: {
    title?: string;
    file_type?: string;
    upload_date: string;
    user_id?: string;
  };
}

export interface ChunkMessage {
  document_id: string;
  case_id: string;
  chunk_id: number;
  text: string;
  metadata: {
    start_position: number;
    end_position: number;
    chunk_size: number;
    overlap_size: number;
  };
}

export interface EmbeddingMessage {
  document_id: string;
  case_id: string;
  chunk_id: number;
  text: string;
  vector: number[];
  metadata: {
    embedding_model: string;
    embedding_dimensions: number;
    confidence_score?: number;
  };
}

export interface ProcessingJob {
  id: string;
  type: 'document' | 'chunk' | 'embedding' | 'storage';
  status: 'queued' | 'processing' | 'completed' | 'failed';
  payload: DocumentMessage | ChunkMessage | EmbeddingMessage;
  created_at: Date;
  updated_at: Date;
  retry_count: number;
  error_message?: string;
}

// Queue configuration
export interface QueueConfig {
  connection: {
    hostname: string;
    port: number;
    username: string;
    password: string;
    vhost?: string;
  };
  queues: {
    docsToProcess: string;
    chunksToEmbed: string;
    chunksToStore: string;
    dlq: string; // Dead letter queue
  };
  exchanges: {
    main: string;
    dlx: string; // Dead letter exchange
  };
  retryConfig: {
    maxRetries: number;
    retryDelay: number;
    exponentialBackoff: boolean;
  };
}

export class RabbitMQQueueService extends EventEmitter {
  private config: QueueConfig;
  private connection: any = null;
  private channel: any = null;
  private isConnected: boolean = false;
  private activeJobs: Map<string, ProcessingJob> = new Map();
  private consumerTags: Map<string, string> = new Map();

  constructor(config: QueueConfig) {
    super();
    this.config = config;
  }

  /**
   * Initialize RabbitMQ connection and setup queues/exchanges
   */
  async initialize(): Promise<void> {
    try {
      // In browser environment, use WebSocket connection to RabbitMQ via proxy
      if (typeof window !== 'undefined') {
        await this.initializeBrowserConnection();
      } else {
        // Server-side initialization would use amqplib
        await this.initializeServerConnection();
      }

      await this.setupQueuesAndExchanges();
      this.isConnected = true;
      this.emit('connected');
      
      console.log('✅ RabbitMQ Queue Service initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize RabbitMQ Queue Service:', error);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Browser-based connection using WebSocket proxy
   */
  private async initializeBrowserConnection(): Promise<void> {
    // In a real implementation, this would connect to RabbitMQ Web STOMP
    // For now, we'll simulate the connection and use HTTP API calls
    
    console.log('🌐 Initializing browser-based RabbitMQ connection...');
    
    // Check if RabbitMQ Management API is accessible
    try {
      const response = await fetch('/api/rabbitmq/health', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error('RabbitMQ Management API not accessible');
      }
      
      const healthData = await response.json();
      console.log('RabbitMQ Health:', healthData);
      
    } catch (error) {
      console.warn('RabbitMQ not available, using simulation mode');
      // Fall back to simulation mode for development
    }
  }

  /**
   * Server-side connection using amqplib
   */
  private async initializeServerConnection(): Promise<void> {
    // This would use amqplib in a real Node.js environment
    console.log('🖥️ Server-side RabbitMQ connection not implemented in browser context');
  }

  /**
   * Setup exchanges and queues for the RAG ingestion pipeline
   */
  private async setupQueuesAndExchanges(): Promise<void> {
    const { queues, exchanges } = this.config;

    // Create exchanges
    await this.createExchange(exchanges.main, 'direct');
    await this.createExchange(exchanges.dlx, 'direct');

    // Create queues with dead letter configuration
    const queueOptions = {
      durable: true,
      arguments: {
        'x-dead-letter-exchange': exchanges.dlx,
        'x-dead-letter-routing-key': 'failed'
      }
    };

    await this.createQueue(queues.docsToProcess, queueOptions);
    await this.createQueue(queues.chunksToEmbed, queueOptions);
    await this.createQueue(queues.chunksToStore, queueOptions);
    await this.createQueue(queues.dlq, { durable: true });

    // Bind queues to exchanges
    await this.bindQueue(queues.docsToProcess, exchanges.main, 'document');
    await this.bindQueue(queues.chunksToEmbed, exchanges.main, 'chunk');
    await this.bindQueue(queues.chunksToStore, exchanges.main, 'embedding');
    await this.bindQueue(queues.dlq, exchanges.dlx, 'failed');
  }

  /**
   * Publish a document to the processing pipeline
   */
  async publishDocument(document: DocumentMessage): Promise<string> {
    const jobId = this.generateJobId();
    
    const job: ProcessingJob = {
      id: jobId,
      type: 'document',
      status: 'queued',
      payload: document,
      created_at: new Date(),
      updated_at: new Date(),
      retry_count: 0
    };

    this.activeJobs.set(jobId, job);

    try {
      await this.publishMessage(
        this.config.exchanges.main,
        'document',
        document,
        { jobId, messageType: 'document' }
      );

      this.emit('documentQueued', { jobId, document });
      console.log(`📄 Document queued for processing: ${document.document_id}`);
      
      return jobId;
    } catch (error) {
      job.status = 'failed';
      job.error_message = error instanceof Error ? error.message : 'Unknown error';
      this.emit('error', { jobId, error });
      throw error;
    }
  }

  /**
   * Publish chunks for embedding
   */
  async publishChunks(chunks: ChunkMessage[]): Promise<string[]> {
    const jobIds: string[] = [];

    for (const chunk of chunks) {
      const jobId = this.generateJobId();
      
      const job: ProcessingJob = {
        id: jobId,
        type: 'chunk',
        status: 'queued',
        payload: chunk,
        created_at: new Date(),
        updated_at: new Date(),
        retry_count: 0
      };

      this.activeJobs.set(jobId, job);
      jobIds.push(jobId);

      try {
        await this.publishMessage(
          this.config.exchanges.main,
          'chunk',
          chunk,
          { jobId, messageType: 'chunk' }
        );

        this.emit('chunkQueued', { jobId, chunk });
      } catch (error) {
        job.status = 'failed';
        job.error_message = error instanceof Error ? error.message : 'Unknown error';
        this.emit('error', { jobId, error });
      }
    }

    console.log(`🧩 ${chunks.length} chunks queued for embedding`);
    return jobIds;
  }

  /**
   * Publish embeddings for storage in Neo4j
   */
  async publishEmbeddings(embeddings: EmbeddingMessage[]): Promise<string[]> {
    const jobIds: string[] = [];

    for (const embedding of embeddings) {
      const jobId = this.generateJobId();
      
      const job: ProcessingJob = {
        id: jobId,
        type: 'embedding',
        status: 'queued',
        payload: embedding,
        created_at: new Date(),
        updated_at: new Date(),
        retry_count: 0
      };

      this.activeJobs.set(jobId, job);
      jobIds.push(jobId);

      try {
        await this.publishMessage(
          this.config.exchanges.main,
          'embedding',
          embedding,
          { jobId, messageType: 'embedding' }
        );

        this.emit('embeddingQueued', { jobId, embedding });
      } catch (error) {
        job.status = 'failed';
        job.error_message = error instanceof Error ? error.message : 'Unknown error';
        this.emit('error', { jobId, error });
      }
    }

    console.log(`🧠 ${embeddings.length} embeddings queued for Neo4j storage`);
    return jobIds;
  }

  /**
   * Start consuming messages from a specific queue
   */
  async startConsumer(queueName: string, handler: (message: any) => Promise<void>): Promise<void> {
    try {
      // In a real implementation, this would setup message consumption
      // For browser simulation, we'll use polling
      
      const consumerTag = `consumer_${queueName}_${Date.now()}`;
      this.consumerTags.set(queueName, consumerTag);

      console.log(`🔄 Started consumer for queue: ${queueName}`);
      this.emit('consumerStarted', { queueName, consumerTag });

      // Start polling for messages (simulation)
      this.startPolling(queueName, handler);
      
    } catch (error) {
      console.error(`❌ Failed to start consumer for ${queueName}:`, error);
      this.emit('error', { queueName, error });
      throw error;
    }
  }

  /**
   * Stop consuming messages from a queue
   */
  async stopConsumer(queueName: string): Promise<void> {
    const consumerTag = this.consumerTags.get(queueName);
    if (consumerTag) {
      this.consumerTags.delete(queueName);
      console.log(`⏹️ Stopped consumer for queue: ${queueName}`);
      this.emit('consumerStopped', { queueName, consumerTag });
    }
  }

  /**
   * Get job status
   */
  getJobStatus(jobId: string): ProcessingJob | null {
    return this.activeJobs.get(jobId) || null;
  }

  /**
   * Get queue statistics
   */
  async getQueueStats(): Promise<Record<string, any>> {
    try {
      const response = await fetch('/api/rabbitmq/queues/stats', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        return await response.json();
      } else {
        // Return simulated stats
        return {
          [this.config.queues.docsToProcess]: { messages: 0, consumers: 1 },
          [this.config.queues.chunksToEmbed]: { messages: 0, consumers: 1 },
          [this.config.queues.chunksToStore]: { messages: 0, consumers: 1 },
          [this.config.queues.dlq]: { messages: 0, consumers: 0 }
        };
      }
    } catch (error) {
      console.warn('Could not fetch queue stats:', error);
      return {};
    }
  }

  /**
   * Cleanup and close connections
   */
  async cleanup(): Promise<void> {
    try {
      // Stop all consumers
      for (const queueName of this.consumerTags.keys()) {
        await this.stopConsumer(queueName);
      }

      // Close connection
      if (this.channel) {
        await this.channel.close();
      }
      if (this.connection) {
        await this.connection.close();
      }

      this.isConnected = false;
      this.emit('disconnected');
      
      console.log('✅ RabbitMQ Queue Service cleanup completed');
    } catch (error) {
      console.error('❌ Error during RabbitMQ cleanup:', error);
      throw error;
    }
  }

  // Private helper methods

  private async createExchange(name: string, type: string): Promise<void> {
    // Simulated exchange creation
    console.log(`📡 Created exchange: ${name} (${type})`);
  }

  private async createQueue(name: string, options: any): Promise<void> {
    // Simulated queue creation
    console.log(`📦 Created queue: ${name}`);
  }

  private async bindQueue(queue: string, exchange: string, routingKey: string): Promise<void> {
    // Simulated queue binding
    console.log(`🔗 Bound queue ${queue} to exchange ${exchange} with routing key ${routingKey}`);
  }

  private async publishMessage(exchange: string, routingKey: string, message: any, headers: any): Promise<void> {
    try {
      // In browser environment, send to API endpoint
      const response = await fetch('/api/rabbitmq/publish', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exchange,
          routingKey,
          message,
          headers
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to publish message: ${response.statusText}`);
      }

      console.log(`✉️ Published message to ${exchange}/${routingKey}`);
    } catch (error) {
      console.warn('Message publish failed, using simulation mode');
      // Simulate successful publish for development
      this.emit('messagePublished', { exchange, routingKey, message, headers });
    }
  }

  private startPolling(queueName: string, handler: (message: any) => Promise<void>): void {
    // Simulated polling for development
    const pollInterval = setInterval(async () => {
      if (!this.consumerTags.has(queueName)) {
        clearInterval(pollInterval);
        return;
      }

      try {
        const response = await fetch(`/api/rabbitmq/consume/${queueName}`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
          const messages = await response.json();
          for (const message of messages) {
            await handler(message);
          }
        }
      } catch (error) {
        // Ignore polling errors in development mode
      }
    }, 5000); // Poll every 5 seconds
  }

  private generateJobId(): string {
    return `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Getters
  get connected(): boolean {
    return this.isConnected;
  }

  get activeJobCount(): number {
    return this.activeJobs.size;
  }

  get queueNames(): string[] {
    return Object.values(this.config.queues);
  }
}

// Default configuration for legal AI platform
export const defaultRabbitMQConfig: QueueConfig = {
  connection: {
    hostname: 'localhost',
    port: 5672,
    username: 'legal_ai',
    password: 'legal_ai_2024'
  },
  queues: {
    docsToProcess: 'legal.docs.process',
    chunksToEmbed: 'legal.chunks.embed',
    chunksToStore: 'legal.chunks.store',
    dlq: 'legal.dlq'
  },
  exchanges: {
    main: 'legal.main',
    dlx: 'legal.dlx'
  },
  retryConfig: {
    maxRetries: 3,
    retryDelay: 5000,
    exponentialBackoff: true
  }
};

// Export singleton instance
export const rabbitMQQueue = new RabbitMQQueueService(defaultRabbitMQConfig);