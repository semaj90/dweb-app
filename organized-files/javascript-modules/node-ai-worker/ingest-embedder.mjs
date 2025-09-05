#!/usr/bin/env node
// Node.js Ingest + Embedding Worker for Legal AI Pipeline
// Handles: MinIO download → OCR/PDF → chunk → embed → PostgreSQL + Neo4j sync
// Uses: llama.cpp for embeddings, RabbitMQ for job queue, Redis for cache

import amqp from 'amqplib';
import { LlamaModel, LlamaContext, getLlama } from 'node-llama-cpp';
import { Client as MinioClient } from 'minio';
import { Client } from 'pg';
import neo4j from 'neo4j-driver';
import Redis from 'ioredis';
import sharp from 'sharp';
import Tesseract from 'tesseract.js';
import pdf from 'pdf-parse';
import { ChunkSplitter } from './lib/chunking.mjs';
import { EntityExtractor } from './lib/entity-extraction.mjs';
import { VectorOperations } from './lib/vector-ops.mjs';

// Configuration
const CONFIG = {
  AMQP_URL: process.env.AMQP_URL || 'amqp://localhost',
  POSTGRES_URL: process.env.DATABASE_URL || 'postgresql://user:password@localhost:5432/legal_ai_db',
  NEO4J_URL: process.env.NEO4J_URL || 'bolt://localhost:7687',
  NEO4J_USER: process.env.NEO4J_USER || 'neo4j',
  NEO4J_PASSWORD: process.env.NEO4J_PASSWORD || 'password',
  REDIS_URL: process.env.REDIS_URL || 'redis://localhost:6379',
  MINIO_ENDPOINT: process.env.MINIO_ENDPOINT || 'localhost',
  MINIO_PORT: parseInt(process.env.MINIO_PORT || '9000'),
  MINIO_ACCESS_KEY: process.env.MINIO_ACCESS_KEY || 'minioadmin',
  MINIO_SECRET_KEY: process.env.MINIO_SECRET_KEY || 'minioadmin',
  MODELS_PATH: process.env.MODELS_PATH || './models',
  WORKER_ID: process.env.WORKER_ID || `worker-${Date.now()}`,
  MAX_CHUNK_SIZE: parseInt(process.env.MAX_CHUNK_SIZE || '500'), // tokens
  CHUNK_OVERLAP: parseInt(process.env.CHUNK_OVERLAP || '50'),
  GPU_LAYERS: parseInt(process.env.GPU_LAYERS || '35'), // RTX 3060 Ti
  EMBEDDING_DIMENSIONS: parseInt(process.env.EMBEDDING_DIMENSIONS || '384'),
};

console.log(`🚀 Starting Legal AI Ingest Worker (${CONFIG.WORKER_ID})`);

// Global clients
let llamaModel = null;
let pgClient = null;
let neo4jDriver = null;
let redisClient = null;
let minioClient = null;
let amqpChannel = null;

// Initialize all services
async function initializeServices() {
  console.log('🔧 Initializing services...');

  try {
    // Initialize llama.cpp with embedding model
    console.log('🦙 Loading llama.cpp embedding model...');
    const llama = await getLlama();
    llamaModel = await llama.loadModel({
      modelPath: `${CONFIG.MODELS_PATH}/nomic-embed-text-v1.5.f16.gguf`,
      gpuLayers: CONFIG.GPU_LAYERS,
      contextSize: 2048,
      batchSize: 512,
    });
    console.log('✅ Embedding model loaded');

    // PostgreSQL client
    pgClient = new Client({ connectionString: CONFIG.POSTGRES_URL });
    await pgClient.connect();
    console.log('✅ PostgreSQL connected');

    // Neo4j driver
    neo4jDriver = neo4j.driver(
      CONFIG.NEO4J_URL,
      neo4j.auth.basic(CONFIG.NEO4J_USER, CONFIG.NEO4J_PASSWORD)
    );
    await neo4jDriver.verifyConnectivity();
    console.log('✅ Neo4j connected');

    // Redis client
    redisClient = new Redis(CONFIG.REDIS_URL);
    await redisClient.ping();
    console.log('✅ Redis connected');

    // MinIO client
    minioClient = new MinioClient({
      endPoint: CONFIG.MINIO_ENDPOINT,
      port: CONFIG.MINIO_PORT,
      useSSL: false,
      accessKey: CONFIG.MINIO_ACCESS_KEY,
      secretKey: CONFIG.MINIO_SECRET_KEY,
    });
    console.log('✅ MinIO connected');

    // RabbitMQ connection
    const amqpConn = await amqp.connect(CONFIG.AMQP_URL);
    amqpChannel = await amqpConn.createChannel();
    
    // Declare queues
    await amqpChannel.assertQueue('legal.document.ingest', { durable: true });
    await amqpChannel.assertQueue('legal.document.chunk', { durable: true });
    await amqpChannel.assertQueue('legal.document.embed', { durable: true });
    await amqpChannel.assertQueue('legal.neo4j.sync', { durable: true });
    await amqpChannel.assertQueue('legal.document.failed', { durable: true }); // DLQ
    
    console.log('✅ RabbitMQ queues ready');

  } catch (error) {
    console.error('❌ Service initialization failed:', error);
    process.exit(1);
  }
}

// Generate embeddings using llama.cpp
async function generateEmbedding(text, model = 'nomic-embed-text') {
  try {
    if (!llamaModel) {
      throw new Error('Embedding model not loaded');
    }

    console.log(`🔢 Generating embedding for ${text.length} chars...`);
    const startTime = Date.now();

    // Create context and encode
    const context = await llamaModel.createContext({ contextSize: 2048 });
    const embedding = await context.encode(text);
    
    // Convert to normalized array
    const vector = Array.from(embedding.slice(0, CONFIG.EMBEDDING_DIMENSIONS));
    const normalizedVector = VectorOperations.normalize(vector);
    
    const duration = Date.now() - startTime;
    console.log(`✅ Embedding generated in ${duration}ms (${normalizedVector.length}D)`);

    // Cache in Redis
    const cacheKey = `embed:${model}:${VectorOperations.hash(text)}`;
    await redisClient.setex(cacheKey, 3600, JSON.stringify(normalizedVector)); // 1 hour cache

    return normalizedVector;

  } catch (error) {
    console.error('❌ Embedding generation failed:', error);
    throw error;
  }
}

// Extract text from various file formats
async function extractText(buffer, mimeType, filename) {
  console.log(`📄 Extracting text from ${filename} (${mimeType})`);
  
  try {
    switch (mimeType) {
      case 'application/pdf':
        const pdfData = await pdf(buffer);
        return {
          text: pdfData.text,
          metadata: {
            pages: pdfData.numpages,
            info: pdfData.info,
          }
        };

      case 'image/png':
      case 'image/jpeg':
      case 'image/jpg':
      case 'image/tiff':
        // Preprocess image with Sharp
        const optimizedImage = await sharp(buffer)
          .greyscale()
          .normalize()
          .sharpen()
          .png()
          .toBuffer();

        // OCR with Tesseract
        const ocrResult = await Tesseract.recognize(optimizedImage, 'eng', {
          logger: info => console.log(`📖 OCR: ${info.status} ${info.progress || ''}`),
        });

        return {
          text: ocrResult.data.text,
          metadata: {
            confidence: ocrResult.data.confidence / 100,
            blocks: ocrResult.data.blocks.length,
            ocrEngine: 'tesseract'
          }
        };

      case 'text/plain':
        return {
          text: buffer.toString('utf-8'),
          metadata: { encoding: 'utf-8' }
        };

      default:
        throw new Error(`Unsupported file type: ${mimeType}`);
    }
  } catch (error) {
    console.error(`❌ Text extraction failed for ${filename}:`, error);
    throw error;
  }
}

// Download file from MinIO
async function downloadFromMinio(sourceUri) {
  console.log(`⬇️ Downloading from MinIO: ${sourceUri}`);
  
  try {
    // Parse minio://bucket/key format
    const url = new URL(sourceUri);
    const bucket = url.hostname;
    const key = url.pathname.substring(1); // Remove leading /

    const stream = await minioClient.getObject(bucket, key);
    const chunks = [];
    
    return new Promise((resolve, reject) => {
      stream.on('data', chunk => chunks.push(chunk));
      stream.on('end', () => resolve(Buffer.concat(chunks)));
      stream.on('error', reject);
    });

  } catch (error) {
    console.error(`❌ MinIO download failed for ${sourceUri}:`, error);
    throw error;
  }
}

// Process ingest jobs: MinIO download → text extraction → chunking
async function processIngestJob(job) {
  const { documentId, sourceUri, filename, mimeType } = job;
  console.log(`📋 Processing ingest job: ${documentId} (${filename})`);

  try {
    // Update job status
    await pgClient.query(
      'UPDATE processing_jobs SET status = $1, started_at = NOW(), worker_node = $2 WHERE document_id = $3 AND job_type = $4',
      ['processing', CONFIG.WORKER_ID, documentId, 'ingest']
    );

    // Download from MinIO
    const buffer = await downloadFromMinio(sourceUri);
    console.log(`✅ Downloaded ${buffer.length} bytes`);

    // Extract text
    const { text, metadata } = await extractText(buffer, mimeType, filename);
    console.log(`✅ Extracted ${text.length} characters`);

    // Update document with extracted text
    await pgClient.query(
      `UPDATE documents SET 
        extracted_text = $1, 
        metadata = metadata || $2::jsonb,
        processing_status = 'extracted',
        updated_at = NOW()
      WHERE id = $3`,
      [text, JSON.stringify(metadata), documentId]
    );

    // Create chunking job
    const chunkJob = {
      documentId,
      text,
      metadata
    };

    await amqpChannel.sendToQueue(
      'legal.document.chunk',
      Buffer.from(JSON.stringify(chunkJob)),
      { persistent: true }
    );

    // Mark ingest job complete
    await pgClient.query(
      'UPDATE processing_jobs SET status = $1, completed_at = NOW() WHERE document_id = $2 AND job_type = $3',
      ['completed', documentId, 'ingest']
    );

    console.log(`✅ Ingest completed for ${documentId}`);

  } catch (error) {
    console.error(`❌ Ingest failed for ${documentId}:`, error);
    
    // Mark job as failed and create retry job
    await pgClient.query(
      'UPDATE processing_jobs SET status = $1, failed_at = NOW(), payload = payload || $2::jsonb WHERE document_id = $3 AND job_type = $4',
      ['failed', JSON.stringify({ error: error.message }), documentId, 'ingest']
    );

    // Send to DLQ for retry
    await amqpChannel.sendToQueue(
      'legal.document.failed',
      Buffer.from(JSON.stringify({ ...job, error: error.message })),
      { persistent: true }
    );
  }
}

// Process chunking jobs: text → overlapping chunks
async function processChunkJob(job) {
  const { documentId, text, metadata } = job;
  console.log(`🔪 Processing chunk job: ${documentId}`);

  try {
    // Initialize chunker
    const chunker = new ChunkSplitter({
      chunkSize: CONFIG.MAX_CHUNK_SIZE,
      overlap: CONFIG.CHUNK_OVERLAP,
      preserveStructure: true
    });

    // Split into chunks
    const chunks = await chunker.splitText(text);
    console.log(`✅ Created ${chunks.length} chunks`);

    // Insert chunks into database
    const chunkInserts = chunks.map(async (chunk, index) => {
      const chunkId = crypto.randomUUID();
      
      await pgClient.query(
        `INSERT INTO document_chunks (
          id, document_id, chunk_index, text, tokens, 
          start_offset, end_offset, metadata, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())`,
        [
          chunkId,
          documentId,
          index,
          chunk.text,
          chunk.tokens,
          chunk.startOffset,
          chunk.endOffset,
          JSON.stringify(chunk.metadata || {})
        ]
      );

      // Queue for embedding
      await amqpChannel.sendToQueue(
        'legal.document.embed',
        Buffer.from(JSON.stringify({ chunkId, documentId, text: chunk.text })),
        { persistent: true }
      );
    });

    await Promise.all(chunkInserts);

    // Update document status
    await pgClient.query(
      `UPDATE documents SET 
        processing_status = 'chunked', 
        updated_at = NOW() 
      WHERE id = $1`,
      [documentId]
    );

    console.log(`✅ Chunking completed for ${documentId}`);

  } catch (error) {
    console.error(`❌ Chunking failed for ${documentId}:`, error);
    throw error;
  }
}

// Process embedding jobs: chunk → vector → PostgreSQL + Redis cache
async function processEmbedJob(job) {
  const { chunkId, documentId, text } = job;
  console.log(`🔢 Processing embed job: ${chunkId}`);

  try {
    // Check Redis cache first
    const cacheKey = `embed:nomic:${VectorOperations.hash(text)}`;
    let embedding = await redisClient.get(cacheKey);
    
    if (embedding) {
      embedding = JSON.parse(embedding);
      console.log('✅ Cache hit for embedding');
    } else {
      embedding = await generateEmbedding(text);
    }

    // Store embedding in database
    await pgClient.query(
      `UPDATE document_chunks SET 
        embedding = $1::vector,
        embedding_model = $2,
        updated_at = NOW()
      WHERE id = $3`,
      [VectorOperations.toVector(embedding), 'nomic-embed-text', chunkId]
    );

    // Extract entities for Neo4j
    const entities = await EntityExtractor.extractEntities(text);
    
    if (entities.length > 0) {
      await amqpChannel.sendToQueue(
        'legal.neo4j.sync',
        Buffer.from(JSON.stringify({ 
          chunkId, 
          documentId, 
          entities,
          embedding 
        })),
        { persistent: true }
      );
    }

    console.log(`✅ Embedding completed for ${chunkId} (${entities.length} entities)`);

  } catch (error) {
    console.error(`❌ Embedding failed for ${chunkId}:`, error);
    throw error;
  }
}

// Process Neo4j sync jobs: entities → graph nodes
async function processNeo4jSyncJob(job) {
  const { chunkId, documentId, entities, embedding } = job;
  console.log(`🌐 Processing Neo4j sync: ${chunkId} (${entities.length} entities)`);

  try {
    const session = neo4jDriver.session();

    try {
      for (const entity of entities) {
        // Create or update entity node
        const result = await session.run(
          `MERGE (e:Entity {name: $name, type: $type})
           ON CREATE SET 
             e.id = randomUUID(),
             e.confidence = $confidence,
             e.createdAt = datetime(),
             e.documentIds = [$documentId],
             e.chunkIds = [$chunkId]
           ON MATCH SET 
             e.documentIds = CASE 
               WHEN $documentId IN e.documentIds THEN e.documentIds 
               ELSE e.documentIds + [$documentId] 
             END,
             e.chunkIds = CASE 
               WHEN $chunkId IN e.chunkIds THEN e.chunkIds 
               ELSE e.chunkIds + [$chunkId] 
             END,
             e.updatedAt = datetime()
           RETURN e.id as nodeId`,
          {
            name: entity.text,
            type: entity.label,
            confidence: entity.confidence,
            documentId,
            chunkId
          }
        );

        const nodeId = result.records[0]?.get('nodeId');

        if (nodeId) {
          // Update PostgreSQL with Neo4j node ID
          await pgClient.query(
            `INSERT INTO entity_nodes (
              neo4j_id, entity_type, name, normalized_name, 
              confidence, embedding, document_ids, chunk_ids,
              last_synced_at, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, NOW(), NOW(), NOW())
            ON CONFLICT (entity_type, normalized_name) 
            DO UPDATE SET
              document_ids = CASE 
                WHEN entity_nodes.document_ids ? $9 THEN entity_nodes.document_ids 
                ELSE entity_nodes.document_ids || $10::jsonb 
              END,
              chunk_ids = CASE 
                WHEN entity_nodes.chunk_ids ? $11 THEN entity_nodes.chunk_ids 
                ELSE entity_nodes.chunk_ids || $12::jsonb 
              END,
              last_synced_at = NOW(),
              updated_at = NOW()`,
            [
              nodeId,
              entity.label,
              entity.text,
              entity.text.toLowerCase().trim(),
              entity.confidence,
              VectorOperations.toVector(embedding),
              JSON.stringify([documentId]),
              JSON.stringify([chunkId]),
              documentId,
              JSON.stringify([documentId]),
              chunkId,
              JSON.stringify([chunkId])
            ]
          );
        }
      }

    } finally {
      await session.close();
    }

    // Check if all chunks for document are processed
    const result = await pgClient.query(
      'SELECT COUNT(*) as total, COUNT(embedding) as embedded FROM document_chunks WHERE document_id = $1',
      [documentId]
    );

    const { total, embedded } = result.rows[0];
    
    if (parseInt(total) === parseInt(embedded)) {
      // All chunks embedded, mark document as completed
      await pgClient.query(
        `UPDATE documents SET 
          processing_status = 'completed', 
          processed_at = NOW(),
          updated_at = NOW()
        WHERE id = $1`,
        [documentId]
      );
      console.log(`🎉 Document ${documentId} fully processed!`);
    }

    console.log(`✅ Neo4j sync completed for ${chunkId}`);

  } catch (error) {
    console.error(`❌ Neo4j sync failed for ${chunkId}:`, error);
    throw error;
  }
}

// Main worker loop
async function startWorker() {
  console.log(`👷 Starting worker queues...`);

  // Set prefetch count for fair distribution
  await amqpChannel.prefetch(1);

  // Consume ingest jobs
  await amqpChannel.consume('legal.document.ingest', async (msg) => {
    if (!msg) return;
    
    try {
      const job = JSON.parse(msg.content.toString());
      await processIngestJob(job);
      amqpChannel.ack(msg);
    } catch (error) {
      console.error('Ingest job failed:', error);
      amqpChannel.nack(msg, false, false); // Send to DLQ
    }
  }, { noAck: false });

  // Consume chunk jobs
  await amqpChannel.consume('legal.document.chunk', async (msg) => {
    if (!msg) return;
    
    try {
      const job = JSON.parse(msg.content.toString());
      await processChunkJob(job);
      amqpChannel.ack(msg);
    } catch (error) {
      console.error('Chunk job failed:', error);
      amqpChannel.nack(msg, false, false);
    }
  }, { noAck: false });

  // Consume embedding jobs
  await amqpChannel.consume('legal.document.embed', async (msg) => {
    if (!msg) return;
    
    try {
      const job = JSON.parse(msg.content.toString());
      await processEmbedJob(job);
      amqpChannel.ack(msg);
    } catch (error) {
      console.error('Embed job failed:', error);
      amqpChannel.nack(msg, false, false);
    }
  }, { noAck: false });

  // Consume Neo4j sync jobs
  await amqpChannel.consume('legal.neo4j.sync', async (msg) => {
    if (!msg) return;
    
    try {
      const job = JSON.parse(msg.content.toString());
      await processNeo4jSyncJob(job);
      amqpChannel.ack(msg);
    } catch (error) {
      console.error('Neo4j sync job failed:', error);
      amqpChannel.nack(msg, false, false);
    }
  }, { noAck: false });

  console.log('🚀 Worker started - waiting for jobs...');
}

// Health monitoring
async function healthCheck() {
  return {
    worker: CONFIG.WORKER_ID,
    status: 'healthy',
    services: {
      llama: llamaModel ? 'connected' : 'disconnected',
      postgres: pgClient ? 'connected' : 'disconnected',
      neo4j: neo4jDriver ? 'connected' : 'disconnected',
      redis: redisClient ? 'connected' : 'disconnected',
      minio: minioClient ? 'connected' : 'disconnected',
      rabbitmq: amqpChannel ? 'connected' : 'disconnected',
    },
    memory: process.memoryUsage(),
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  };
}

// Graceful shutdown
async function shutdown() {
  console.log('🛑 Shutting down worker...');
  
  try {
    if (amqpChannel) await amqpChannel.close();
    if (pgClient) await pgClient.end();
    if (neo4jDriver) await neo4jDriver.close();
    if (redisClient) await redisClient.quit();
    
    console.log('✅ Worker stopped gracefully');
    process.exit(0);
  } catch (error) {
    console.error('❌ Shutdown error:', error);
    process.exit(1);
  }
}

// Signal handlers
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

// Error handlers
process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught exception:', error);
  shutdown();
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled rejection at:', promise, 'reason:', reason);
  shutdown();
});

// Start the worker
(async () => {
  try {
    await initializeServices();
    await startWorker();
    
    // Health check endpoint (if you want to serve HTTP)
    if (process.env.HEALTH_PORT) {
      const express = (await import('express')).default;
      const app = express();
      
      app.get('/health', async (req, res) => {
        res.json(await healthCheck());
      });
      
      app.listen(process.env.HEALTH_PORT, () => {
        console.log(`🔍 Health endpoint: http://localhost:${process.env.HEALTH_PORT}/health`);
      });
    }
    
    console.log('🎉 Legal AI Ingest Worker ready!');

  } catch (error) {
    console.error('❌ Worker startup failed:', error);
    process.exit(1);
  }
})();