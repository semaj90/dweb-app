// Neural Sprite Engine Service Connections
// Integrates the Neural Sprite Engine with PostgreSQL, Redis, RabbitMQ, and MinIO
// Provides type-safe service connections for the legal AI platform

import { writable, type Readable } from 'svelte/store';
import postgres from 'postgres';
import { createClient, type RedisClientType } from 'redis';
import amqp, { type Connection, type Channel } from 'amqplib';
import { Client as MinIOClient } from 'minio';
import type { CanvasSprite, SOMNode, UserActivity, ProcessingTask } from '$lib/engines/neural-sprite-engine';

// Configuration interface
export interface NeuralServiceConfig {
  postgresql: {
    connectionString: string;
    maxConnections?: number;
    enablePgVector?: boolean;
  };
  redis: {
    url: string;
    keyPrefix?: string;
    ttl?: number;
  };
  rabbitmq: {
    url: string;
    exchange?: string;
    queues?: {
      neuralTasks: string;
      somUpdates: string;
      predictions: string;
    };
  };
  minio: {
    endpoint: string;
    port: number;
    accessKey: string;
    secretKey: string;
    bucket?: string;
  };
}

// Service connection status
export interface ServiceStatus {
  name: string;
  connected: boolean;
  lastCheck: Date;
  error?: string;
  metrics?: Record<string, any>;
}

// Neural Service Connection Manager
export class NeuralServiceConnections {
  private config: NeuralServiceConfig;
  private sql?: ReturnType<typeof postgres>;
  private redis?: RedisClientType;
  private rabbitmq?: Connection;
  private rabbitmqChannel?: Channel;
  private minio?: MinIOClient;

  // Reactive stores for service status
  public status = writable<Record<string, ServiceStatus>>({});
  public isConnected = writable<boolean>(false);
  public connectionErrors = writable<string[]>([]);

  constructor(config: NeuralServiceConfig) {
    this.config = config;
  }

  /**
   * Initialize all service connections
   */
  async initialize(): Promise<boolean> {
    console.log('🔌 Initializing Neural Service Connections...');
    
    const results = await Promise.allSettled([
      this.initializePostgreSQL(),
      this.initializeRedis(),
      this.initializeRabbitMQ(),
      this.initializeMinIO()
    ]);

    const errors: string[] = [];
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        const serviceName = ['PostgreSQL', 'Redis', 'RabbitMQ', 'MinIO'][index];
        errors.push(`${serviceName}: ${result.reason}`);
      }
    });

    this.connectionErrors.set(errors);
    const allConnected = errors.length === 0;
    this.isConnected.set(allConnected);

    if (allConnected) {
      console.log('✅ All neural service connections initialized');
      await this.setupSchemas();
    } else {
      console.warn('⚠️ Some service connections failed:', errors);
    }

    return allConnected;
  }

  /**
   * Initialize PostgreSQL connection with pgvector support
   */
  private async initializePostgreSQL(): Promise<void> {
    try {
      this.sql = postgres(this.config.postgresql.connectionString, {
        max: this.config.postgresql.maxConnections || 10,
        idle_timeout: 20,
        connect_timeout: 10,
      });

      // Test connection
      await this.sql`SELECT 1 as test`;

      // Check for pgvector extension
      if (this.config.postgresql.enablePgVector) {
        const vectorCheck = await this.sql`
          SELECT extname FROM pg_extension WHERE extname = 'vector';
        `;
        
        if (vectorCheck.length === 0) {
          console.warn('⚠️ pgvector extension not found - vector operations will be limited');
        } else {
          console.log('✅ pgvector extension available');
        }
      }

      this.updateStatus('PostgreSQL', true);
      console.log('✅ PostgreSQL connection initialized');
    } catch (error) {
      this.updateStatus('PostgreSQL', false, error instanceof Error ? error.message : String(error));
      throw new Error(`PostgreSQL connection failed: ${error}`);
    }
  }

  /**
   * Initialize Redis connection for caching
   */
  private async initializeRedis(): Promise<void> {
    try {
      this.redis = createClient({ url: this.config.redis.url }) as RedisClientType;
      
      this.redis.on('error', (err) => {
        console.error('Redis error:', err);
        this.updateStatus('Redis', false, err.message);
      });

      await this.redis.connect();
      
      // Test connection
      await this.redis.ping();

      this.updateStatus('Redis', true);
      console.log('✅ Redis connection initialized');
    } catch (error) {
      this.updateStatus('Redis', false, error instanceof Error ? error.message : String(error));
      throw new Error(`Redis connection failed: ${error}`);
    }
  }

  /**
   * Initialize RabbitMQ connection for neural task distribution
   */
  private async initializeRabbitMQ(): Promise<void> {
    try {
      this.rabbitmq = await amqp.connect(this.config.rabbitmq.url);
      this.rabbitmqChannel = await this.rabbitmq.createChannel();

      // Setup exchanges and queues
      const exchange = this.config.rabbitmq.exchange || 'neural_engine';
      await this.rabbitmqChannel.assertExchange(exchange, 'topic', { durable: true });

      const queues = this.config.rabbitmq.queues || {
        neuralTasks: 'neural_tasks',
        somUpdates: 'som_updates',
        predictions: 'predictions'
      };

      for (const [key, queueName] of Object.entries(queues)) {
        await this.rabbitmqChannel.assertQueue(queueName, { durable: true });
        await this.rabbitmqChannel.bindQueue(queueName, exchange, `neural.${key}`);
      }

      this.updateStatus('RabbitMQ', true);
      console.log('✅ RabbitMQ connection initialized');
    } catch (error) {
      this.updateStatus('RabbitMQ', false, error instanceof Error ? error.message : String(error));
      throw new Error(`RabbitMQ connection failed: ${error}`);
    }
  }

  /**
   * Initialize MinIO connection for large file storage
   */
  private async initializeMinIO(): Promise<void> {
    try {
      this.minio = new MinIOClient({
        endPoint: this.config.minio.endpoint,
        port: this.config.minio.port,
        useSSL: false,
        accessKey: this.config.minio.accessKey,
        secretKey: this.config.minio.secretKey,
      });

      // Test connection
      await this.minio.listBuckets();

      // Ensure bucket exists
      const bucket = this.config.minio.bucket || 'neural-engine';
      const exists = await this.minio.bucketExists(bucket);
      if (!exists) {
        await this.minio.makeBucket(bucket);
        console.log(`✅ Created MinIO bucket: ${bucket}`);
      }

      this.updateStatus('MinIO', true);
      console.log('✅ MinIO connection initialized');
    } catch (error) {
      this.updateStatus('MinIO', false, error instanceof Error ? error.message : String(error));
      throw new Error(`MinIO connection failed: ${error}`);
    }
  }

  /**
   * Setup database schemas for neural engine
   */
  private async setupSchemas(): Promise<void> {
    if (!this.sql) return;

    try {
      // Create neural sprites table with vector support
      await this.sql`
        CREATE TABLE IF NOT EXISTS neural_sprites (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          sequence INTEGER NOT NULL,
          json_state TEXT NOT NULL,
          metadata JSONB NOT NULL,
          embedding vector(16),
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW(),
          usage_count INTEGER DEFAULT 0,
          predicted_next TEXT[] DEFAULT '{}'
        );
      `;

      // Create indexes for performance
      await this.sql`
        CREATE INDEX IF NOT EXISTS idx_neural_sprites_name ON neural_sprites(name);
        CREATE INDEX IF NOT EXISTS idx_neural_sprites_usage ON neural_sprites(usage_count DESC);
        CREATE INDEX IF NOT EXISTS idx_neural_sprites_created ON neural_sprites(created_at DESC);
      `;

      // Create vector index if pgvector is available
      if (this.config.postgresql.enablePgVector) {
        try {
          await this.sql`
            CREATE INDEX IF NOT EXISTS idx_neural_sprites_embedding 
            ON neural_sprites USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
          `;
        } catch (vectorError) {
          console.warn('⚠️ Could not create vector index - pgvector may not be available');
        }
      }

      // Create SOM nodes table
      await this.sql`
        CREATE TABLE IF NOT EXISTS som_nodes (
          id TEXT PRIMARY KEY,
          position JSONB NOT NULL,
          weights vector(16) NOT NULL,
          activation_history REAL[] DEFAULT '{}',
          connected_sprites TEXT[] DEFAULT '{}',
          learning_rate REAL NOT NULL,
          neighborhood_radius REAL NOT NULL,
          created_at TIMESTAMPTZ DEFAULT NOW(),
          updated_at TIMESTAMPTZ DEFAULT NOW()
        );
      `;

      // Create user activities table
      await this.sql`
        CREATE TABLE IF NOT EXISTS neural_user_activities (
          id TEXT PRIMARY KEY,
          action TEXT NOT NULL,
          context JSONB NOT NULL,
          timestamp TIMESTAMPTZ DEFAULT NOW(),
          canvas_state TEXT,
          sequence INTEGER NOT NULL,
          session_id TEXT
        );
      `;

      console.log('✅ Neural engine database schemas initialized');
    } catch (error) {
      console.error('❌ Failed to setup schemas:', error);
      throw error;
    }
  }

  /**
   * Save sprite to PostgreSQL with vector embedding
   */
  async saveSprite(sprite: CanvasSprite): Promise<void> {
    if (!this.sql) throw new Error('PostgreSQL not connected');

    const embedding = sprite.embedding ? `[${sprite.embedding.join(',')}]` : null;

    await this.sql`
      INSERT INTO neural_sprites (
        id, name, sequence, json_state, metadata, embedding, 
        created_at, usage_count, predicted_next
      ) VALUES (
        ${sprite.id}, ${sprite.name}, ${sprite.sequence}, ${sprite.jsonState},
        ${JSON.stringify(sprite.metadata)}, ${embedding}::vector,
        ${new Date(sprite.createdAt)}, ${sprite.usageCount}, 
        ${sprite.predictedNext || []}
      )
      ON CONFLICT (id) 
      DO UPDATE SET 
        usage_count = EXCLUDED.usage_count,
        predicted_next = EXCLUDED.predicted_next,
        updated_at = NOW()
    `;
  }

  /**
   * Load sprite from PostgreSQL by ID
   */
  async loadSprite(id: string): Promise<CanvasSprite | null> {
    if (!this.sql) throw new Error('PostgreSQL not connected');

    const result = await this.sql`
      SELECT * FROM neural_sprites WHERE id = ${id}
    `;

    if (result.length === 0) return null;

    const row = result[0];
    return {
      id: row.id,
      name: row.name,
      sequence: row.sequence,
      jsonState: row.json_state,
      metadata: row.metadata,
      embedding: row.embedding ? Array.from(row.embedding) : undefined,
      createdAt: new Date(row.created_at).getTime(),
      usageCount: row.usage_count,
      predictedNext: row.predicted_next
    };
  }

  /**
   * Find similar sprites using vector similarity
   */
  async findSimilarSprites(embedding: number[], limit: number = 10): Promise<CanvasSprite[]> {
    if (!this.sql || !this.config.postgresql.enablePgVector) {
      throw new Error('PostgreSQL with pgvector not available');
    }

    const embeddingVector = `[${embedding.join(',')}]`;

    const results = await this.sql`
      SELECT *, (embedding <=> ${embeddingVector}::vector) as distance
      FROM neural_sprites 
      WHERE embedding IS NOT NULL
      ORDER BY distance
      LIMIT ${limit}
    `;

    return results.map(row => ({
      id: row.id,
      name: row.name,
      sequence: row.sequence,
      jsonState: row.json_state,
      metadata: row.metadata,
      embedding: Array.from(row.embedding),
      createdAt: new Date(row.created_at).getTime(),
      usageCount: row.usage_count,
      predictedNext: row.predicted_next
    }));
  }

  /**
   * Cache SOM node in Redis
   */
  async cacheSOMNode(node: SOMNode): Promise<void> {
    if (!this.redis) throw new Error('Redis not connected');

    const key = `${this.config.redis.keyPrefix || 'neural'}:som:${node.id}`;
    const ttl = this.config.redis.ttl || 3600; // 1 hour default

    await this.redis.setEx(key, ttl, JSON.stringify({
      ...node,
      weights: Array.from(node.weights)
    }));
  }

  /**
   * Load SOM node from Redis cache
   */
  async loadCachedSOMNode(nodeId: string): Promise<SOMNode | null> {
    if (!this.redis) throw new Error('Redis not connected');

    const key = `${this.config.redis.keyPrefix || 'neural'}:som:${nodeId}`;
    const cached = await this.redis.get(key);

    if (!cached) return null;

    const data = JSON.parse(cached);
    return {
      ...data,
      weights: new Float32Array(data.weights)
    };
  }

  /**
   * Publish neural processing task to RabbitMQ
   */
  async publishTask(task: ProcessingTask): Promise<void> {
    if (!this.rabbitmqChannel) throw new Error('RabbitMQ not connected');

    const exchange = this.config.rabbitmq.exchange || 'neural_engine';
    const routingKey = `neural.${task.type}`;

    await this.rabbitmqChannel.publish(
      exchange,
      routingKey,
      Buffer.from(JSON.stringify(task)),
      { persistent: true }
    );
  }

  /**
   * Store large sprite state in MinIO
   */
  async storeSpriteLargeState(spriteId: string, jsonState: string): Promise<string> {
    if (!this.minio) throw new Error('MinIO not connected');

    const bucket = this.config.minio.bucket || 'neural-engine';
    const objectName = `sprites/${spriteId}/state.json`;

    await this.minio.putObject(
      bucket,
      objectName,
      Buffer.from(jsonState, 'utf8'),
      {
        'Content-Type': 'application/json',
        'x-amz-meta-sprite-id': spriteId,
        'x-amz-meta-created': new Date().toISOString()
      }
    );

    return objectName;
  }

  /**
   * Load large sprite state from MinIO
   */
  async loadSpriteLargeState(objectName: string): Promise<string> {
    if (!this.minio) throw new Error('MinIO not connected');

    const bucket = this.config.minio.bucket || 'neural-engine';
    
    return new Promise((resolve, reject) => {
      let data = '';
      this.minio!.getObject(bucket, objectName, (err, stream) => {
        if (err) {
          reject(err);
          return;
        }

        stream!.on('data', (chunk) => {
          data += chunk.toString('utf8');
        });

        stream!.on('end', () => {
          resolve(data);
        });

        stream!.on('error', reject);
      });
    });
  }

  /**
   * Get service health metrics
   */
  async getHealthMetrics(): Promise<Record<string, any>> {
    const metrics: Record<string, any> = {};

    // PostgreSQL metrics
    if (this.sql) {
      try {
        const spriteCount = await this.sql`SELECT COUNT(*) as count FROM neural_sprites`;
        const somCount = await this.sql`SELECT COUNT(*) as count FROM som_nodes`;
        metrics.postgresql = {
          sprites: parseInt(spriteCount[0].count),
          somNodes: parseInt(somCount[0].count),
          connected: true
        };
      } catch (error) {
        metrics.postgresql = { connected: false, error: error instanceof Error ? error.message : String(error) };
      }
    }

    // Redis metrics
    if (this.redis) {
      try {
        const info = await this.redis.info('memory');
        metrics.redis = {
          memory: info,
          connected: true
        };
      } catch (error) {
        metrics.redis = { connected: false, error: error instanceof Error ? error.message : String(error) };
      }
    }

    return metrics;
  }

  /**
   * Update service status
   */
  private updateStatus(serviceName: string, connected: boolean, error?: string): void {
    this.status.update(current => ({
      ...current,
      [serviceName]: {
        name: serviceName,
        connected,
        lastCheck: new Date(),
        error
      }
    }));
  }

  /**
   * Clean up all connections
   */
  async disconnect(): Promise<void> {
    console.log('🔌 Disconnecting neural service connections...');

    const disconnectPromises: Promise<void>[] = [];

    if (this.sql) {
      disconnectPromises.push(this.sql.end());
    }

    if (this.redis) {
      disconnectPromises.push(this.redis.disconnect());
    }

    if (this.rabbitmq) {
      disconnectPromises.push(this.rabbitmq.close());
    }

    await Promise.allSettled(disconnectPromises);
    console.log('✅ Neural service connections disconnected');
  }
}

// Factory function for creating service connections
export function createNeuralServiceConnections(config: NeuralServiceConfig): NeuralServiceConnections {
  return new NeuralServiceConnections(config);
}

// Helper function to load configuration from environment
export function createConfigFromEnv(): NeuralServiceConfig {
  return {
    postgresql: {
      connectionString: process.env.VITE_POSTGRESQL_URL || 'postgresql://postgres@localhost:5432/legal_ai_db',
      maxConnections: 10,
      enablePgVector: true
    },
    redis: {
      url: process.env.VITE_REDIS_URL || 'redis://localhost:6379',
      keyPrefix: 'neural',
      ttl: 3600
    },
    rabbitmq: {
      url: process.env.VITE_RABBITMQ_URL || 'amqp://localhost:5672',
      exchange: 'neural_engine',
      queues: {
        neuralTasks: 'neural_tasks',
        somUpdates: 'som_updates',
        predictions: 'predictions'
      }
    },
    minio: {
      endpoint: process.env.VITE_MINIO_ENDPOINT?.split(':')[0] || 'localhost',
      port: parseInt(process.env.VITE_MINIO_ENDPOINT?.split(':')[1] || '9000'),
      accessKey: 'minioadmin',
      secretKey: 'minioadmin123',
      bucket: 'neural-engine'
    }
  };
}