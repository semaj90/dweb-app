/**
 * YoRHa Embedding Service
 * Dedicated embedding server with llama.cpp GGUF support and uniform API
 */

import express from 'express';
import { Server, ServerCredentials } from '@grpc/grpc-js';
import { loadPackageDefinition } from '@grpc/proto-loader';
import { loadSync } from '@grpc/proto-loader';
import { LlamaCpp, LlamaModel, LlamaContext, LlamaEmbedding } from 'node-llama-cpp';
import Redis from 'ioredis';
import amqp from 'amqplib';
import { z } from 'zod';
import pino from 'pino';

// Configuration
const config = {
  grpc: {
    port: parseInt(process.env.EMBEDDING_GRPC_PORT || '50051')
  },
  http: {
    port: parseInt(process.env.EMBEDDING_HTTP_PORT || '8092')
  },
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD
  },
  rabbitmq: {
    url: process.env.AMQP_URL || 'amqp://localhost'
  },
  models: {
    defaultEmbedding: process.env.DEFAULT_EMBEDDING_MODEL || 'nomic-embed-text',
    modelsDir: process.env.MODELS_DIR || './models',
    maxConcurrency: parseInt(process.env.MAX_CONCURRENT_EMBEDDINGS || '4')
  }
};

// Logger
const logger = pino({
  transport: { target: 'pino-pretty' }
});

// Model Manager
class ModelManager {
  private models: Map<string, { model: LlamaModel; context: LlamaContext }> = new Map();
  private embedding: LlamaEmbedding | null = null;
  private llamaCpp: LlamaCpp;

  constructor() {
    this.llamaCpp = new LlamaCpp();
  }

  async loadEmbeddingModel(modelName: string, modelPath: string): Promise<void> {
    try {
      logger.info(`Loading embedding model: ${modelName} from ${modelPath}`);
      
      const model = await this.llamaCpp.loadModel({
        modelPath,
        // Embedding-specific optimizations
        gpuLayers: -1, // Use all GPU layers
        contextSize: 512, // Smaller context for embeddings
        batchSize: 512,
        threads: 4
      });

      const context = await model.createContext({
        contextSize: 512
      });

      const embedding = new LlamaEmbedding({ context });
      
      this.models.set(modelName, { model, context });
      this.embedding = embedding;

      logger.info(`✅ Model ${modelName} loaded successfully`);
    } catch (error) {
      logger.error(`❌ Failed to load model ${modelName}:`, error);
      throw error;
    }
  }

  async generateEmbedding(text: string, modelName: string = config.models.defaultEmbedding): Promise<number[]> {
    if (!this.embedding) {
      throw new Error(`Embedding model not loaded: ${modelName}`);
    }

    try {
      const startTime = Date.now();
      const embedding = await this.embedding.getEmbedding(text);
      const processingTime = Date.now() - startTime;

      logger.debug(`Generated embedding for ${text.length} chars in ${processingTime}ms`);
      return Array.from(embedding);
    } catch (error) {
      logger.error('Embedding generation failed:', error);
      throw error;
    }
  }

  async batchEmbedding(texts: string[], modelName: string = config.models.defaultEmbedding): Promise<number[][]> {
    if (!this.embedding) {
      throw new Error(`Embedding model not loaded: ${modelName}`);
    }

    try {
      const startTime = Date.now();
      const embeddings = await Promise.all(
        texts.map(text => this.embedding!.getEmbedding(text))
      );
      const processingTime = Date.now() - startTime;

      logger.debug(`Generated ${embeddings.length} embeddings in ${processingTime}ms`);
      return embeddings.map(emb => Array.from(emb));
    } catch (error) {
      logger.error('Batch embedding failed:', error);
      throw error;
    }
  }

  getLoadedModels(): string[] {
    return Array.from(this.models.keys());
  }

  isModelLoaded(modelName: string): boolean {
    return this.models.has(modelName);
  }
}

// Cache Manager
class EmbeddingCache {
  private redis: Redis;

  constructor() {
    this.redis = new Redis(config.redis);
    this.redis.on('error', (error) => {
      logger.error('Redis connection error:', error);
    });
  }

  private getCacheKey(text: string, model: string): string {
    // Create deterministic hash of text + model
    const crypto = require('crypto');
    const hash = crypto.createHash('sha256')
      .update(`${text}:${model}`)
      .digest('hex')
      .slice(0, 16);
    return `embed:${model}:${hash}`;
  }

  async getCachedEmbedding(text: string, model: string): Promise<number[] | null> {
    try {
      const key = this.getCacheKey(text, model);
      const cached = await this.redis.get(key);
      
      if (cached) {
        logger.debug(`Cache hit for embedding: ${key}`);
        return JSON.parse(cached);
      }

      return null;
    } catch (error) {
      logger.warn('Cache retrieval failed:', error);
      return null;
    }
  }

  async cacheEmbedding(text: string, model: string, embedding: number[], ttl: number = 3600): Promise<void> {
    try {
      const key = this.getCacheKey(text, model);
      await this.redis.setex(key, ttl, JSON.stringify(embedding));
      logger.debug(`Cached embedding: ${key}`);
    } catch (error) {
      logger.warn('Cache storage failed:', error);
    }
  }

  async batchCacheEmbeddings(items: Array<{text: string; model: string; embedding: number[]}>): Promise<void> {
    try {
      const pipeline = this.redis.pipeline();
      
      for (const item of items) {
        const key = this.getCacheKey(item.text, item.model);
        pipeline.setex(key, 3600, JSON.stringify(item.embedding));
      }

      await pipeline.exec();
      logger.debug(`Batch cached ${items.length} embeddings`);
    } catch (error) {
      logger.warn('Batch cache storage failed:', error);
    }
  }

  async disconnect(): Promise<void> {
    await this.redis.disconnect();
  }
}

// Embedding Service Implementation
class EmbeddingService {
  private modelManager: ModelManager;
  private cache: EmbeddingCache;
  private requestCount = 0;
  private totalLatency = 0;

  constructor() {
    this.modelManager = new ModelManager();
    this.cache = new EmbeddingCache();
  }

  async initialize(): Promise<void> {
    // Load default embedding model
    const defaultModel = config.models.defaultEmbedding;
    const modelPath = `${config.models.modelsDir}/${defaultModel}.gguf`;
    
    try {
      await this.modelManager.loadEmbeddingModel(defaultModel, modelPath);
      logger.info('✅ Embedding service initialized');
    } catch (error) {
      logger.error('❌ Failed to initialize embedding service:', error);
      throw error;
    }
  }

  // gRPC Service Methods
  async embed(call: any, callback: any): Promise<void> {
    const startTime = Date.now();
    this.requestCount++;

    try {
      const { id, text, model = config.models.defaultEmbedding, metadata = {} } = call.request;

      // Check cache first
      const cached = await this.cache.getCachedEmbedding(text, model);
      if (cached) {
        const response = {
          id,
          vector: cached,
          dimensions: cached.length,
          model,
          success: true,
          processing_time: Date.now() - startTime,
          metadata: { ...metadata, cached: 'true' }
        };
        callback(null, response);
        return;
      }

      // Generate embedding
      const embedding = await this.modelManager.generateEmbedding(text, model);
      const processingTime = Date.now() - startTime;
      this.totalLatency += processingTime;

      // Cache result
      await this.cache.cacheEmbedding(text, model, embedding);

      const response = {
        id,
        vector: embedding,
        dimensions: embedding.length,
        model,
        success: true,
        processing_time: processingTime,
        metadata: { ...metadata, cached: 'false' }
      };

      callback(null, response);
    } catch (error) {
      logger.error('Embedding failed:', error);
      callback({
        code: 13, // INTERNAL
        message: error.message
      });
    }
  }

  async batchEmbed(call: any, callback: any): Promise<void> {
    const startTime = Date.now();
    
    try {
      const { requests, batch_id, normalize = true } = call.request;
      
      const results = [];
      let processed = 0;
      let failed = 0;

      for (const req of requests) {
        try {
          const { id, text, model = config.models.defaultEmbedding, metadata = {} } = req;
          
          // Check cache
          let embedding = await this.cache.getCachedEmbedding(text, model);
          let cached = true;

          if (!embedding) {
            embedding = await this.modelManager.generateEmbedding(text, model);
            await this.cache.cacheEmbedding(text, model, embedding);
            cached = false;
          }

          results.push({
            id,
            vector: embedding,
            dimensions: embedding.length,
            model,
            success: true,
            processing_time: Date.now() - startTime,
            metadata: { ...metadata, cached: cached.toString() }
          });

          processed++;
        } catch (error) {
          failed++;
          results.push({
            id: req.id,
            vector: [],
            dimensions: 0,
            model: req.model,
            success: false,
            processing_time: 0,
            metadata: { error: error.message }
          });
        }
      }

      callback(null, {
        embeddings: results,
        batch_id,
        total_processed: processed,
        total_failed: failed,
        total_processing_time: Date.now() - startTime,
        success: failed === 0
      });
    } catch (error) {
      logger.error('Batch embedding failed:', error);
      callback({
        code: 13,
        message: error.message
      });
    }
  }

  async healthCheck(call: any, callback: any): Promise<void> {
    const { detailed = false } = call.request;
    
    const models = {};
    for (const modelName of this.modelManager.getLoadedModels()) {
      models[modelName] = this.modelManager.isModelLoaded(modelName) ? 'loaded' : 'unloaded';
    }

    const avgLatency = this.requestCount > 0 ? this.totalLatency / this.requestCount : 0;

    const response = {
      healthy: true,
      status: 'OK',
      models,
      uptime_seconds: Math.floor(process.uptime()),
      stats: detailed ? {
        cpu_usage: process.cpuUsage().system / 1000000,
        memory_usage_mb: process.memoryUsage().rss / 1024 / 1024,
        gpu_usage: 0, // TODO: Get actual GPU usage
        gpu_memory_mb: 0,
        total_embeddings: this.requestCount,
        active_requests: 0,
        average_latency_ms: avgLatency
      } : undefined
    };

    callback(null, response);
  }

  getStats() {
    return {
      requestCount: this.requestCount,
      averageLatency: this.requestCount > 0 ? this.totalLatency / this.requestCount : 0
    };
  }
}

// HTTP API (REST fallback)
class HTTPServer {
  private app: express.Application;
  private embeddingService: EmbeddingService;

  constructor(embeddingService: EmbeddingService) {
    this.app = express();
    this.embeddingService = embeddingService;
    this.setupRoutes();
  }

  private setupRoutes(): void {
    this.app.use(express.json({ limit: '10mb' }));

    // Health check
    this.app.get('/health', async (req, res) => {
      const stats = this.embeddingService.getStats();
      res.json({
        status: 'healthy',
        uptime: process.uptime(),
        stats
      });
    });

    // Single embedding
    this.app.post('/embed', async (req, res) => {
      const schema = z.object({
        text: z.string(),
        model: z.string().optional(),
        id: z.string().optional()
      });

      try {
        const { text, model = config.models.defaultEmbedding, id = Date.now().toString() } = schema.parse(req.body);
        
        // Mock gRPC call for HTTP endpoint
        const call = { request: { id, text, model } };
        
        return new Promise((resolve, reject) => {
          this.embeddingService.embed(call, (error: any, response: any) => {
            if (error) {
              res.status(500).json({ error: error.message });
              reject(error);
            } else {
              res.json(response);
              resolve(response);
            }
          });
        });
      } catch (error) {
        res.status(400).json({ error: 'Invalid request format' });
      }
    });

    // Batch embedding
    this.app.post('/embed/batch', async (req, res) => {
      const schema = z.object({
        texts: z.array(z.string()),
        model: z.string().optional(),
        batch_id: z.string().optional()
      });

      try {
        const { texts, model = config.models.defaultEmbedding, batch_id = Date.now().toString() } = schema.parse(req.body);
        
        const requests = texts.map((text, i) => ({
          id: `${batch_id}_${i}`,
          text,
          model
        }));

        const call = { request: { requests, batch_id, normalize: true } };
        
        return new Promise((resolve, reject) => {
          this.embeddingService.batchEmbed(call, (error: any, response: any) => {
            if (error) {
              res.status(500).json({ error: error.message });
              reject(error);
            } else {
              res.json(response);
              resolve(response);
            }
          });
        });
      } catch (error) {
        res.status(400).json({ error: 'Invalid request format' });
      }
    });
  }

  start(): void {
    this.app.listen(config.http.port, () => {
      logger.info(`🌐 HTTP server listening on port ${config.http.port}`);
    });
  }
}

// Main Server
async function main() {
  try {
    logger.info('🚀 Starting YoRHa Embedding Service...');

    // Initialize embedding service
    const embeddingService = new EmbeddingService();
    await embeddingService.initialize();

    // Start HTTP server
    const httpServer = new HTTPServer(embeddingService);
    httpServer.start();

    // Start gRPC server
    const packageDefinition = loadSync('../../proto/embedding.proto', {
      keepCase: true,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true
    });

    const protoDescriptor = loadPackageDefinition(packageDefinition) as any;
    const embeddingProto = protoDescriptor.yorha.embedding;

    const grpcServer = new Server();
    grpcServer.addService(embeddingProto.EmbeddingService.service, {
      embed: embeddingService.embed.bind(embeddingService),
      batchEmbed: embeddingService.batchEmbed.bind(embeddingService),
      searchSimilar: (call: any, callback: any) => callback({ code: 12, message: 'Not implemented' }),
      healthCheck: embeddingService.healthCheck.bind(embeddingService),
      loadModel: (call: any, callback: any) => callback({ code: 12, message: 'Not implemented' }),
      listModels: (call: any, callback: any) => callback({ code: 12, message: 'Not implemented' })
    });

    grpcServer.bindAsync(
      `0.0.0.0:${config.grpc.port}`,
      ServerCredentials.createInsecure(),
      (error, port) => {
        if (error) {
          logger.error('❌ gRPC server failed to start:', error);
          return;
        }
        logger.info(`🚀 gRPC server listening on port ${port}`);
        grpcServer.start();
      }
    );

    // Graceful shutdown
    process.on('SIGINT', async () => {
      logger.info('🛑 Shutting down...');
      grpcServer.forceShutdown();
      process.exit(0);
    });

  } catch (error) {
    logger.error('❌ Failed to start embedding service:', error);
    process.exit(1);
  }
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

if (require.main === module) {
  main();
}